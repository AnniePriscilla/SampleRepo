"""
==============================================================================
E-COMMERCE ORDER SUPPORT AGENT
==============================================================================
An intent-routed support agent for tracking customer orders, shipment /
delivery status, and supplier information.

The agent inspects an incoming customer query and decides whether it needs:
  1. FULL-TEXT SEARCH  -> exact / structured lookups (order id, tracking
     number, email, carrier, exact status keyword).
  2. SEMANTIC SEARCH    -> fuzzy, descriptive, natural-language queries that
     do not contain a hard identifier ("the leather bag order from the
     Italian supplier that's stuck somewhere").
  3. HYBRID             -> query contains both an identifier AND descriptive
     language -> both tools are run and results are merged.

Design goal: this file must run stand-alone inside a restricted sandbox such
as Vocareum with NO internet access and NO API keys configured. It therefore
ships with local, dependency-light fallbacks:
  - Full-text search  -> SQLite FTS5 (built into Python's stdlib sqlite3)
  - Semantic search   -> Pinecone if `pinecone-client` + PINECONE_API_KEY are
                         available, otherwise an in-memory TF-IDF + cosine
                         similarity index (scikit-learn) that mimics the same
                         interface, so the rest of the code never changes.
  - Embeddings        -> OpenAI `text-embedding-3-small` if OPENAI_API_KEY is
                         set, otherwise the same TF-IDF vectorizer used by
                         the mock Pinecone index.

Run this file directly to see sample queries executed end-to-end and the
full test suite:
    python3 ecommerce_support_agent.py
==============================================================================
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ------------------------------------------------------------------------
# Shared identifier regexes. Defined once here so both the Intent Router
# Agent and the Full-Text Search Tool extract candidate identifiers the
# same way.
# ------------------------------------------------------------------------
ORDER_ID_RE = re.compile(r"\bORD-?\d{3,}\b", re.IGNORECASE)
TRACKING_RE = re.compile(r"\bTRK[0-9]{6,}\b", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")

# ------------------------------------------------------------------------
# Optional real backends. Import failures are swallowed on purpose: the
# module must keep working in an offline sandbox.
# ------------------------------------------------------------------------
try:
    import pinecone  # type: ignore
    _PINECONE_AVAILABLE = True
except Exception:
    _PINECONE_AVAILABLE = False

try:
    import openai  # type: ignore
    _OPENAI_AVAILABLE = True
except Exception:
    _OPENAI_AVAILABLE = False


PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_ENV = os.environ.get("PINECONE_ENVIRONMENT", "us-east-1")
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "ecommerce-orders-semantic")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

USE_REAL_PINECONE = bool(_PINECONE_AVAILABLE and PINECONE_API_KEY)
USE_REAL_OPENAI_EMBEDDINGS = bool(_OPENAI_AVAILABLE and OPENAI_API_KEY)


# ==========================================================================
# 1. DATASET
# ==========================================================================
# `semantic_description` is the field that gets embedded for Pinecone. It is
# a natural-language rollup of the product, supplier, and shipment narrative
# so that vague / conceptual customer language still matches.

TEST_ORDERS: List[Dict[str, Any]] = [
    {
        "order_id": "ORD-10001",
        "customer_name": "Ananya Rao",
        "customer_email": "ananya.rao@example.com",
        "product_name": "Genuine Leather Messenger Bag",
        "product_category": "Fashion Accessories",
        "supplier_name": "Milano Leather Co.",
        "supplier_location": "Florence, Italy",
        "order_date": "2026-08-10",
        "shipment_status": "In Transit",
        "tracking_number": "TRK9081726354",
        "carrier": "DHL",
        "estimated_delivery_date": "2026-09-08",
        "actual_delivery_date": None,
        "semantic_description": (
            "A handcrafted genuine leather messenger bag imported from an "
            "artisan supplier in Florence, Italy. The shipment is currently "
            "in transit via DHL after clearing customs, moving from the "
            "European distribution hub toward the local delivery facility."
        ),
    },
    {
        "order_id": "ORD-10002",
        "customer_name": "Rahul Mehta",
        "customer_email": "rahul.mehta@example.com",
        "product_name": "Wireless Noise Cancelling Headphones",
        "product_category": "Electronics",
        "supplier_name": "SoundWave Electronics Ltd.",
        "supplier_location": "Shenzhen, China",
        "order_date": "2026-08-20",
        "shipment_status": "Delayed",
        "tracking_number": "TRK1122334455",
        "carrier": "FedEx",
        "estimated_delivery_date": "2026-08-30",
        "actual_delivery_date": None,
        "semantic_description": (
            "Premium wireless noise cancelling headphones sourced from an "
            "electronics manufacturer in Shenzhen, China. The shipment has "
            "been sitting at the origin sorting facility longer than "
            "expected and is now flagged as delayed past its original "
            "delivery estimate."
        ),
    },
    {
        "order_id": "ORD-10003",
        "customer_name": "Priya Nair",
        "customer_email": "priya.nair@example.com",
        "product_name": "Organic Cotton Bedsheet Set",
        "product_category": "Home & Living",
        "supplier_name": "GreenWeave Textiles",
        "supplier_location": "Coimbatore, India",
        "order_date": "2026-08-25",
        "shipment_status": "Delivered",
        "tracking_number": "TRK5566778899",
        "carrier": "BlueDart",
        "estimated_delivery_date": "2026-08-30",
        "actual_delivery_date": "2026-08-29",
        "semantic_description": (
            "A soft organic cotton bedsheet set manufactured by a "
            "sustainable textile supplier in Coimbatore, India. The order "
            "was delivered a day ahead of schedule by BlueDart with no "
            "reported issues."
        ),
    },
    {
        "order_id": "ORD-10004",
        "customer_name": "Vikram Singh",
        "customer_email": "vikram.singh@example.com",
        "product_name": "Stainless Steel Cookware Set",
        "product_category": "Kitchen & Dining",
        "supplier_name": "HomeChef Metalworks",
        "supplier_location": "Ludhiana, India",
        "order_date": "2026-08-28",
        "shipment_status": "Out for Delivery",
        "tracking_number": "TRK6677889900",
        "carrier": "Delhivery",
        "estimated_delivery_date": "2026-09-06",
        "actual_delivery_date": None,
        "semantic_description": (
            "A durable stainless steel cookware set produced by a metalware "
            "supplier in Ludhiana, India. The package left the last-mile "
            "hub this morning and is out for delivery with the local "
            "Delhivery courier."
        ),
    },
    {
        "order_id": "ORD-10005",
        "customer_name": "Sneha Iyer",
        "customer_email": "sneha.iyer@example.com",
        "product_name": "Kids Educational Building Blocks",
        "product_category": "Toys",
        "supplier_name": "BrightMinds Toys Inc.",
        "supplier_location": "Hanoi, Vietnam",
        "order_date": "2026-08-15",
        "shipment_status": "Returned",
        "tracking_number": "TRK7788990011",
        "carrier": "FedEx",
        "estimated_delivery_date": "2026-08-25",
        "actual_delivery_date": None,
        "semantic_description": (
            "A set of educational wooden building blocks for children, "
            "manufactured by a toy supplier in Hanoi, Vietnam. The customer "
            "refused delivery due to a damaged outer box, and the shipment "
            "is now marked as returned to the origin warehouse."
        ),
    },
    {
        "order_id": "ORD-10006",
        "customer_name": "Arjun Reddy",
        "customer_email": "arjun.reddy@example.com",
        "product_name": "Bluetooth Fitness Tracker Watch",
        "product_category": "Electronics",
        "supplier_name": "PulseTech Wearables",
        "supplier_location": "Shenzhen, China",
        "order_date": "2026-08-22",
        "shipment_status": "In Transit",
        "tracking_number": "TRK2233445566",
        "carrier": "DHL",
        "estimated_delivery_date": "2026-09-07",
        "actual_delivery_date": None,
        "semantic_description": (
            "A Bluetooth-enabled fitness tracker watch from a wearables "
            "manufacturer based in Shenzhen, China. The parcel has left "
            "the origin country and is currently airborne en route to the "
            "destination customs facility."
        ),
    },
    {
        "order_id": "ORD-10007",
        "customer_name": "Meera Pillai",
        "customer_email": "meera.pillai@example.com",
        "product_name": "Handwoven Silk Saree",
        "product_category": "Fashion",
        "supplier_name": "Kanchipuram Weavers Guild",
        "supplier_location": "Kanchipuram, India",
        "order_date": "2026-08-18",
        "shipment_status": "Delivered",
        "tracking_number": "TRK3344556677",
        "carrier": "BlueDart",
        "estimated_delivery_date": "2026-08-24",
        "actual_delivery_date": "2026-08-24",
        "semantic_description": (
            "A handwoven silk saree crafted by a traditional weavers' "
            "guild in Kanchipuram, India, known for authentic silk "
            "textiles. Delivered on schedule with no delays."
        ),
    },
    {
        "order_id": "ORD-10008",
        "customer_name": "Karthik Subramanian",
        "customer_email": "karthik.s@example.com",
        "product_name": "Espresso Coffee Machine",
        "product_category": "Kitchen & Dining",
        "supplier_name": "Milano Appliance Works",
        "supplier_location": "Milan, Italy",
        "order_date": "2026-08-29",
        "shipment_status": "Processing",
        "tracking_number": None,
        "carrier": None,
        "estimated_delivery_date": "2026-09-15",
        "actual_delivery_date": None,
        "semantic_description": (
            "A premium espresso coffee machine sourced from an appliance "
            "manufacturer in Milan, Italy. The order is still being "
            "processed at the supplier's warehouse and has not yet been "
            "handed to a shipping carrier."
        ),
    },
    {
        "order_id": "ORD-10009",
        "customer_name": "Divya Krishnan",
        "customer_email": "divya.krishnan@example.com",
        "product_name": "Ceramic Dinnerware Set",
        "product_category": "Kitchen & Dining",
        "supplier_name": "TerraCotta Ceramics",
        "supplier_location": "Jaipur, India",
        "order_date": "2026-08-12",
        "shipment_status": "Cancelled",
        "tracking_number": None,
        "carrier": None,
        "estimated_delivery_date": None,
        "actual_delivery_date": None,
        "semantic_description": (
            "A hand-painted ceramic dinnerware set from a pottery supplier "
            "in Jaipur, India. The order was cancelled before dispatch "
            "because the supplier reported the item was out of stock."
        ),
    },
    {
        "order_id": "ORD-10010",
        "customer_name": "Rohit Sharma",
        "customer_email": "rohit.sharma@example.com",
        "product_name": "Gaming Laptop 16-inch",
        "product_category": "Electronics",
        "supplier_name": "NextGen Computing Corp.",
        "supplier_location": "Taipei, Taiwan",
        "order_date": "2026-08-30",
        "shipment_status": "Shipped",
        "tracking_number": "TRK4455667788",
        "carrier": "FedEx",
        "estimated_delivery_date": "2026-09-10",
        "actual_delivery_date": None,
        "semantic_description": (
            "A high-performance 16-inch gaming laptop manufactured by a "
            "computing hardware supplier in Taipei, Taiwan. The shipment "
            "has just left the supplier's facility and is booked on an "
            "international FedEx freight lane."
        ),
    },
    {
        "order_id": "ORD-10011",
        "customer_name": "Ishaan Kapoor",
        "customer_email": "ishaan.kapoor@example.com",
        "product_name": "Yoga Mat and Accessories Kit",
        "product_category": "Sports & Fitness",
        "supplier_name": "ZenFit Manufacturing",
        "supplier_location": "Bengaluru, India",
        "order_date": "2026-09-01",
        "shipment_status": "In Transit",
        "tracking_number": "TRK8899001122",
        "carrier": "Delhivery",
        "estimated_delivery_date": "2026-09-09",
        "actual_delivery_date": None,
        "semantic_description": (
            "A yoga mat bundled with fitness accessories, produced by a "
            "sports-goods manufacturer in Bengaluru, India. The shipment "
            "is moving between regional sorting hubs on its way to the "
            "customer's city."
        ),
    },
    {
        "order_id": "ORD-10012",
        "customer_name": "Neha Joshi",
        "customer_email": "neha.joshi@example.com",
        "product_name": "Designer Sunglasses",
        "product_category": "Fashion Accessories",
        "supplier_name": "Milano Leather Co.",
        "supplier_location": "Florence, Italy",
        "order_date": "2026-08-27",
        "shipment_status": "Delayed",
        "tracking_number": "TRK9900112233",
        "carrier": "DHL",
        "estimated_delivery_date": "2026-09-02",
        "actual_delivery_date": None,
        "semantic_description": (
            "A pair of designer sunglasses from the same Italian "
            "accessories supplier in Florence that also makes leather "
            "goods. The shipment has been held up at customs longer than "
            "expected, pushing delivery past the original estimate."
        ),
    },
    {
        "order_id": "ORD-10013",
        "customer_name": "Aditya Verma",
        "customer_email": "aditya.verma@example.com",
        "product_name": "Smart Home Security Camera",
        "product_category": "Electronics",
        "supplier_name": "SoundWave Electronics Ltd.",
        "supplier_location": "Shenzhen, China",
        "order_date": "2026-08-19",
        "shipment_status": "Delivered",
        "tracking_number": "TRK0011223344",
        "carrier": "FedEx",
        "estimated_delivery_date": "2026-08-28",
        "actual_delivery_date": "2026-08-27",
        "semantic_description": (
            "A Wi-Fi enabled smart home security camera from an "
            "electronics manufacturer in Shenzhen, China. Delivered a day "
            "early with no exceptions recorded during transit."
        ),
    },
    {
        "order_id": "ORD-10014",
        "customer_name": "Pooja Desai",
        "customer_email": "pooja.desai@example.com",
        "product_name": "Handmade Wooden Furniture Set",
        "product_category": "Home & Living",
        "supplier_name": "OakCraft Furniture",
        "supplier_location": "Saharanpur, India",
        "order_date": "2026-08-05",
        "shipment_status": "In Transit",
        "tracking_number": "TRK1234567890",
        "carrier": "Delhivery",
        "estimated_delivery_date": "2026-09-12",
        "actual_delivery_date": None,
        "semantic_description": (
            "A handmade wooden furniture set carved by a heritage "
            "woodworking supplier in Saharanpur, India. Due to the bulky "
            "size, the shipment is moving slowly through surface freight "
            "rather than air, and is currently between two regional hubs."
        ),
    },
    {
        "order_id": "ORD-10015",
        "customer_name": "Kavya Menon",
        "customer_email": "kavya.menon@example.com",
        "product_name": "Skincare Gift Hamper",
        "product_category": "Beauty & Personal Care",
        "supplier_name": "PureGlow Cosmetics",
        "supplier_location": "Seoul, South Korea",
        "order_date": "2026-08-31",
        "shipment_status": "Shipped",
        "tracking_number": "TRK2345678901",
        "carrier": "DHL",
        "estimated_delivery_date": "2026-09-11",
        "actual_delivery_date": None,
        "semantic_description": (
            "A curated skincare gift hamper sourced from a cosmetics "
            "supplier in Seoul, South Korea, popular for K-beauty "
            "products. The parcel has just been handed to DHL for "
            "international transit."
        ),
    },
]


# ==========================================================================
# 2. FULL-TEXT SEARCH TOOL  (SQLite FTS5)
# ==========================================================================
class FullTextSearchTool:
    """
    Deterministic keyword / exact-identifier search over structured order
    fields: order_id, tracking_number, customer_name, customer_email,
    supplier_name, carrier, shipment_status.

    Backed by SQLite's FTS5 virtual table -> no external server required,
    safe to run inside Vocareum or any offline sandbox.
    """

    name = "full_text_search"
    description = (
        "Exact / structured lookup tool. Use for queries that contain an "
        "order id, tracking number, email address, carrier name, or an "
        "exact shipment-status keyword."
    )

    SEARCHABLE_FIELDS = [
        "order_id", "customer_name", "customer_email", "product_name",
        "supplier_name", "tracking_number", "carrier", "shipment_status",
    ]

    def __init__(self, orders: List[Dict[str, Any]]):
        self._orders_by_id = {o["order_id"]: o for o in orders}
        self._conn = sqlite3.connect(":memory:")
        self._build_index(orders)

    def _build_index(self, orders: List[Dict[str, Any]]) -> None:
        cols = ", ".join(self.SEARCHABLE_FIELDS)
        self._conn.execute(f"CREATE VIRTUAL TABLE orders_fts USING fts5({cols})")
        rows = [
            tuple(str(o.get(f) or "") for f in self.SEARCHABLE_FIELDS)
            for o in orders
        ]
        placeholders = ", ".join(["?"] * len(self.SEARCHABLE_FIELDS))
        self._conn.executemany(
            f"INSERT INTO orders_fts ({cols}) VALUES ({placeholders})", rows
        )
        self._conn.commit()

    @staticmethod
    def _sanitize_fts_query(raw_query: str) -> str:
        """Turn free text into a safe OR-joined FTS5 MATCH expression."""
        tokens = re.findall(r"[A-Za-z0-9\-_@.]+", raw_query)
        tokens = [t for t in tokens if len(t) > 1]
        if not tokens:
            return '""'
        return " OR ".join(f'"{t}"*' for t in tokens)

    def search(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Input : {"query": str, "top_k": int}
        Output: {"search_type": "full_text", "matches": [ {order fields..., "match_score": float} ]}
        """
        # 1) Extract candidate hard identifiers (order id / tracking number /
        # email) from anywhere inside the free-text query, then look for an
        # exact (case-insensitive) field match. This is what lets a query
        # like "What is the status of ORD-10001?" resolve to a single order
        # instead of falling through to fuzzy keyword ranking.
        candidates = set()
        candidates.update(m.group(0) for m in ORDER_ID_RE.finditer(query))
        candidates.update(m.group(0) for m in TRACKING_RE.finditer(query))
        candidates.update(m.group(0) for m in EMAIL_RE.finditer(query))

        exact_hits = []
        if candidates:
            lowered_candidates = {c.lower() for c in candidates}
            for order in self._orders_by_id.values():
                for field_name in ("order_id", "tracking_number", "customer_email"):
                    value = order.get(field_name)
                    if value and value.lower() in lowered_candidates:
                        exact_hits.append({**order, "match_score": 1.0})
                        break
        if exact_hits:
            return {"search_type": "full_text", "matches": exact_hits[:top_k]}

        # 2) Fall back to FTS5 ranked keyword search.
        fts_query = self._sanitize_fts_query(query)
        cur = self._conn.execute(
            "SELECT order_id, bm25(orders_fts) AS rank FROM orders_fts "
            "WHERE orders_fts MATCH ? ORDER BY rank LIMIT ?",
            (fts_query, top_k),
        )
        results = cur.fetchall()
        matches = []
        if results:
            worst_rank = max(r[1] for r in results) or 1.0
            for order_id, rank in results:
                # bm25() in SQLite returns *lower is better*; convert to a
                # 0-1 "higher is better" score for a consistent API.
                score = 1.0 - (rank / worst_rank if worst_rank else 0)
                score = round(max(min(score, 1.0), 0.05), 4)
                matches.append({**self._orders_by_id[order_id], "match_score": score})
        return {"search_type": "full_text", "matches": matches}


# ==========================================================================
# 3. SEMANTIC SEARCH TOOL  (Pinecone, with local TF-IDF fallback)
# ==========================================================================
class _MockPineconeIndex:
    """
    Drop-in stand-in for a Pinecone index when no API key / network is
    available. Uses TF-IDF vectors + cosine similarity over
    `semantic_description`, which is a reasonable local proxy for a real
    embedding model for demo / test purposes.
    """

    def __init__(self, orders: List[Dict[str, Any]]):
        self._orders = orders
        corpus = [o["semantic_description"] for o in orders]
        self._vectorizer = TfidfVectorizer(stop_words="english")
        self._matrix = self._vectorizer.fit_transform(corpus)

    def query(self, query_text: str, top_k: int, filter: Optional[Dict[str, Any]] = None):
        query_vec = self._vectorizer.transform([query_text])
        scores = cosine_similarity(query_vec, self._matrix).flatten()
        ranked = np.argsort(-scores)
        results = []
        for idx in ranked:
            order = self._orders[idx]
            if filter and not self._passes_filter(order, filter):
                continue
            results.append({
                "id": order["order_id"],
                "score": round(float(scores[idx]), 4),
                "metadata": order,
            })
            if len(results) >= top_k:
                break
        return results

    @staticmethod
    def _passes_filter(order: Dict[str, Any], filter: Dict[str, Any]) -> bool:
        for key, value in filter.items():
            if order.get(key) != value:
                return False
        return True


class SemanticSearchTool:
    """
    Fuzzy / conceptual search over the vector database. Handles natural
    language queries that describe a product, supplier, or situation rather
    than citing an exact identifier.

    Uses a real Pinecone index when `PINECONE_API_KEY` (and the
    `pinecone-client` package) are available; otherwise falls back to an
    in-process TF-IDF cosine-similarity index with an identical `.query()`
    interface, so calling code never has to branch on which backend is
    active.
    """

    name = "semantic_search"
    description = (
        "Fuzzy / conceptual lookup tool. Use for natural-language queries "
        "describing a product, supplier characteristic, or shipment "
        "situation without a hard identifier (e.g. 'the leather bag from "
        "the Italian supplier that's stuck somewhere')."
    )

    EMBEDDING_MODEL = "text-embedding-3-small"  # used only if OpenAI is available
    VECTOR_DIM_OPENAI = 1536

    def __init__(self, orders: List[Dict[str, Any]]):
        self._orders_by_id = {o["order_id"]: o for o in orders}
        self._orders = orders
        self._backend = "pinecone" if USE_REAL_PINECONE else "mock_tfidf"
        if self._backend == "pinecone":
            self._init_pinecone(orders)
        else:
            self._index = _MockPineconeIndex(orders)

    # -- Real Pinecone path (only exercised when creds + package exist) ----
    def _init_pinecone(self, orders: List[Dict[str, Any]]) -> None:
        pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
        existing = [i["name"] for i in pc.list_indexes()]
        dim = self.VECTOR_DIM_OPENAI if USE_REAL_OPENAI_EMBEDDINGS else 384
        if PINECONE_INDEX_NAME not in existing:
            pc.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=dim,
                metric="cosine",
                spec=pinecone.ServerlessSpec(cloud="aws", region=PINECONE_ENV),
            )
        self._index = pc.Index(PINECONE_INDEX_NAME)
        vectors = []
        for order in orders:
            vectors.append({
                "id": order["order_id"],
                "values": self._embed(order["semantic_description"]),
                "metadata": {k: (v if v is not None else "") for k, v in order.items()},
            })
        self._index.upsert(vectors=vectors)

    def _embed(self, text: str) -> List[float]:
        if USE_REAL_OPENAI_EMBEDDINGS:
            client = openai.OpenAI(api_key=OPENAI_API_KEY)
            resp = client.embeddings.create(model=self.EMBEDDING_MODEL, input=text)
            return resp.data[0].embedding
        # Deterministic offline fallback: hash-based pseudo-embedding so the
        # real-Pinecone code path still runs without an OpenAI key.
        rng = np.random.default_rng(abs(hash(text)) % (2**32))
        return rng.random(384).tolist()

    # -- Public API ----------------------------------------------------
    def search(self, query: str, top_k: int = 3, filter: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Input : {"query": str, "top_k": int, "filter": dict|None}
        Output: {"search_type": "semantic", "matches": [ {order fields..., "match_score": float} ]}
        """
        if self._backend == "pinecone":
            embedding = self._embed(query)
            raw = self._index.query(
                vector=embedding, top_k=top_k, include_metadata=True, filter=filter
            )
            matches = [
                {**m["metadata"], "match_score": round(float(m["score"]), 4)}
                for m in raw["matches"]
            ]
        else:
            raw = self._index.query(query, top_k=top_k, filter=filter)
            matches = [
                {**m["metadata"], "match_score": m["score"]} for m in raw
            ]
        return {"search_type": "semantic", "matches": matches}


# ==========================================================================
# 4. INTENT ROUTER AGENT
# ==========================================================================
STATUS_KEYWORDS = {
    "processing", "shipped", "in transit", "out for delivery",
    "delivered", "delayed", "returned", "cancelled",
}
DESCRIPTIVE_MARKERS = (
    "similar to", "kind of", "sort of", "something like", "the one with",
    "supplier that", "made by", "that smells", "looking for a", "not sure",
    "package with", "order with a", "stuck", "handmade", "handcrafted",
    "any update on my", "made from", "sourced from", "supplier in",
)


@dataclass
class IntentDecision:
    intent: str  # "structured_lookup" | "conceptual_query" | "hybrid"
    matched_identifiers: List[str] = field(default_factory=list)
    reasoning: str = ""
    confidence: float = 0.0


class IntentRouterAgent:
    """
    Decides whether a customer query should go to full-text search,
    semantic search, or both (hybrid).

    Strategy: fast, deterministic, rule-based classification first
    (regex + keyword matching on hard identifiers). This keeps latency and
    cost low and the behaviour auditable/testable. An optional LLM-based
    fallback (`classify_with_llm`) is provided for genuinely ambiguous
    queries, used only when an LLM API key is configured.
    """

    role = (
        "Intent Router Agent: reads the raw customer query and classifies "
        "it as a structured lookup, a conceptual/semantic query, or a "
        "hybrid of both, then hands off to the correct tool(s)."
    )

    def classify(self, query_text: str) -> IntentDecision:
        text = query_text.strip()

        # "Hard" identifiers are unambiguous and always deserve exact
        # matching. "Soft" signals (a bare status word like "shipped"
        # appearing inside a natural sentence) are weaker and must not
        # override clearly descriptive language.
        hard_ids = []
        if ORDER_ID_RE.search(text):
            hard_ids.append("order_id")
        if TRACKING_RE.search(text):
            hard_ids.append("tracking_number")
        if EMAIL_RE.search(text):
            hard_ids.append("email")

        has_status_keyword = any(kw in text.lower() for kw in STATUS_KEYWORDS)
        has_descriptive_language = any(m in text.lower() for m in DESCRIPTIVE_MARKERS)
        word_count = len(text.split())
        is_long_natural_language = word_count >= 8 and has_descriptive_language

        # 1) Hard identifier + descriptive language -> run both tools.
        if hard_ids and has_descriptive_language:
            return IntentDecision(
                intent="hybrid",
                matched_identifiers=hard_ids,
                reasoning=(
                    "Query contains a hard identifier "
                    f"({', '.join(hard_ids)}) AND descriptive natural "
                    "language, so both search tools are used and results "
                    "are merged."
                ),
                confidence=0.85,
            )

        # 2) Hard identifier alone -> deterministic full-text search.
        if hard_ids:
            return IntentDecision(
                intent="structured_lookup",
                matched_identifiers=hard_ids,
                reasoning=(
                    "Query contains a hard identifier "
                    f"({', '.join(hard_ids)}) -> deterministic full-text "
                    "search is more precise than semantic similarity here."
                ),
                confidence=0.95,
            )

        # 3) No hard identifier, but clearly descriptive / conceptual
        #    language -> semantic search, even if a status word also
        #    appears in the sentence (e.g. "has it shipped yet" inside a
        #    longer descriptive question).
        if is_long_natural_language:
            return IntentDecision(
                intent="conceptual_query",
                matched_identifiers=[],
                reasoning=(
                    "No exact identifier found; query describes the "
                    "product/supplier/situation in natural language, so "
                    "semantic (vector) search is required to capture "
                    "meaning rather than keywords."
                ),
                confidence=0.8,
            )

        # 4) No hard identifier, no descriptive language, but a bare status
        #    keyword drives a short, direct query (e.g. "Show me anything
        #    that is Delayed") -> treat the status word as a structured
        #    filter and use full-text search.
        if has_status_keyword:
            return IntentDecision(
                intent="structured_lookup",
                matched_identifiers=["status_keyword"],
                reasoning=(
                    "Short query built around an exact shipment-status "
                    "keyword -> full-text search filters on that field "
                    "directly."
                ),
                confidence=0.7,
            )

        # 5) No hard identifier, no descriptive marker, no status keyword.
        #    Fall back to semantic search since it degrades more gracefully
        #    than an empty full-text match on an out-of-vocabulary query.
        return IntentDecision(
            intent="conceptual_query",
            matched_identifiers=[],
            reasoning="Ambiguous query with no hard identifiers or status keywords; defaulting to semantic search.",
            confidence=0.5,
        )


# ==========================================================================
# 5. ORCHESTRATOR AGENT
# ==========================================================================
class EcommerceSupportAgent:
    """
    Top-level agent that a customer-facing channel (chat widget, IVR
    transcript, email parser, etc.) calls with one query at a time.

    Pipeline:
        customer query -> IntentRouterAgent -> tool(s) -> response composer
    """

    role = (
        "Orchestrator Agent: owns the end-to-end conversation turn. "
        "Delegates intent classification to the Intent Router Agent, "
        "invokes the Full-Text Search Tool and/or Semantic Search Tool "
        "based on that decision, then composes a single structured "
        "response plus a natural-language summary for the customer."
    )

    def __init__(self, orders: List[Dict[str, Any]]):
        self.router = IntentRouterAgent()
        self.full_text_tool = FullTextSearchTool(orders)
        self.semantic_tool = SemanticSearchTool(orders)

    def handle_query(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Input JSON:
        {
          "query_id": str,
          "customer_id": str | null,
          "channel": "chat" | "email" | "voice",
          "query_text": str,
          "timestamp": str (ISO8601)
        }

        Output JSON:
        {
          "query_id": str,
          "intent": "structured_lookup" | "conceptual_query" | "hybrid",
          "tool_used": "full_text_search" | "semantic_search" | "both",
          "matched_identifiers": [str],
          "results": [ {order fields..., match_score} ],
          "confidence": float,
          "response_text": str,
          "timestamp": str (ISO8601)
        }
        """
        query_text = request["query_text"]
        decision = self.router.classify(query_text)

        if decision.intent == "structured_lookup":
            tool_used = "full_text_search"
            results = self.full_text_tool.search(query_text)["matches"]
        elif decision.intent == "conceptual_query":
            tool_used = "semantic_search"
            results = self.semantic_tool.search(query_text)["matches"]
        else:  # hybrid
            tool_used = "both"
            ft_results = self.full_text_tool.search(query_text)["matches"]
            sem_results = self.semantic_tool.search(query_text)["matches"]
            results = self._merge_results(ft_results, sem_results)

        return {
            "query_id": request.get("query_id", str(uuid.uuid4())),
            "intent": decision.intent,
            "tool_used": tool_used,
            "matched_identifiers": decision.matched_identifiers,
            "results": results,
            "confidence": decision.confidence,
            "response_text": self._compose_response_text(query_text, results),
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }

    @staticmethod
    def _merge_results(ft_results: List[Dict[str, Any]], sem_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        merged: Dict[str, Dict[str, Any]] = {}
        for r in ft_results + sem_results:
            oid = r["order_id"]
            if oid not in merged or r["match_score"] > merged[oid]["match_score"]:
                merged[oid] = r
        return sorted(merged.values(), key=lambda r: -r["match_score"])

    @staticmethod
    def _compose_response_text(query_text: str, results: List[Dict[str, Any]]) -> str:
        if not results:
            return (
                "I couldn't find an order matching that request. Could you "
                "share your order ID, tracking number, or the email used "
                "for the purchase?"
            )
        top = results[0]
        parts = [f"Order {top['order_id']} ({top['product_name']}) is currently '{top['shipment_status']}'."]
        if top.get("supplier_name"):
            parts.append(f"Supplied by {top['supplier_name']} ({top.get('supplier_location', 'n/a')}).")
        if top.get("tracking_number"):
            parts.append(f"Tracking number: {top['tracking_number']} via {top.get('carrier', 'n/a')}.")
        if top.get("estimated_delivery_date"):
            parts.append(f"Estimated delivery: {top['estimated_delivery_date']}.")
        return " ".join(parts)


# ==========================================================================
# 6. SAMPLE RUNS
# ==========================================================================
def run_sample_queries(agent: EcommerceSupportAgent) -> None:
    samples = [
        {
            "query_id": "Q-001",
            "customer_id": "CUST-501",
            "channel": "chat",
            "query_text": "What's the status of order ORD-10002?",
            "timestamp": "2026-09-06T10:00:00Z",
        },
        {
            "query_id": "Q-002",
            "customer_id": "CUST-502",
            "channel": "chat",
            "query_text": "I'm looking for the handmade leather bag order from the Italian supplier, is it still stuck somewhere?",
            "timestamp": "2026-09-06T10:05:00Z",
        },
        {
            "query_id": "Q-003",
            "customer_id": "CUST-503",
            "channel": "email",
            "query_text": "Can you check tracking number TRK6677889900 and also tell me if it's the cookware set that's out for delivery?",
            "timestamp": "2026-09-06T10:10:00Z",
        },
    ]
    print("\n" + "=" * 78)
    print("SAMPLE QUERY RUNS")
    print("=" * 78)
    for s in samples:
        result = agent.handle_query(s)
        print(f"\n--- INPUT ---\n{json.dumps(s, indent=2)}")
        print(f"--- OUTPUT ---\n{json.dumps(result, indent=2, default=str)}")


# ==========================================================================
# 7. TEST CASES
# ==========================================================================
TEST_CASES: List[Dict[str, Any]] = [
    {
        "name": "exact_order_id_lookup",
        "query_text": "What is the status of ORD-10001?",
        "expected_intent": "structured_lookup",
        "expected_tool": "full_text_search",
        "expected_top_order_id": "ORD-10001",
    },
    {
        "name": "exact_tracking_number_lookup",
        "query_text": "Track my shipment TRK5566778899 please",
        "expected_intent": "structured_lookup",
        "expected_tool": "full_text_search",
        "expected_top_order_id": "ORD-10003",
    },
    {
        "name": "email_lookup",
        "query_text": "Show me the order for rohit.sharma@example.com",
        "expected_intent": "structured_lookup",
        "expected_tool": "full_text_search",
        "expected_top_order_id": "ORD-10010",
    },
    {
        "name": "semantic_supplier_description",
        "query_text": "I ordered something handcrafted from a leather supplier in Florence, has it shipped yet?",
        "expected_intent": "conceptual_query",
        "expected_tool": "semantic_search",
        "expected_top_order_id": "ORD-10001",
    },
    {
        "name": "semantic_situation_description",
        "query_text": "My kids toy order arrived damaged and I sent it back, what's happening with it now?",
        "expected_intent": "conceptual_query",
        "expected_tool": "semantic_search",
        "expected_top_order_id": "ORD-10005",
    },
    {
        "name": "semantic_vague_supplier_query",
        "query_text": "Which of my orders is coming from a Korean skincare supplier?",
        "expected_intent": "conceptual_query",
        "expected_tool": "semantic_search",
        "expected_top_order_id": "ORD-10015",
    },
    {
        "name": "hybrid_identifier_plus_description",
        "query_text": "Can you check TRK9900112233, it's the sunglasses order that's stuck, any update?",
        "expected_intent": "hybrid",
        "expected_tool": "both",
        "expected_top_order_id": "ORD-10012",
    },
    {
        "name": "status_keyword_only",
        "query_text": "Show me anything that is Delayed",
        "expected_intent": "structured_lookup",
        "expected_tool": "full_text_search",
        "expected_top_order_id": None,  # multiple valid matches
    },
]


def run_test_suite(agent: EcommerceSupportAgent) -> None:
    print("\n" + "=" * 78)
    print("TEST SUITE")
    print("=" * 78)
    passed, failed = 0, 0
    for case in TEST_CASES:
        request = {
            "query_id": case["name"],
            "customer_id": "TEST",
            "channel": "chat",
            "query_text": case["query_text"],
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }
        result = agent.handle_query(request)

        intent_ok = result["intent"] == case["expected_intent"]
        tool_ok = result["tool_used"] == case["expected_tool"]
        top_id = result["results"][0]["order_id"] if result["results"] else None
        top_id_ok = (
            case["expected_top_order_id"] is None
            or top_id == case["expected_top_order_id"]
        )

        ok = intent_ok and tool_ok and top_id_ok
        passed += ok
        failed += not ok

        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {case['name']}")
        print(f"    query           : {case['query_text']}")
        print(f"    expected intent : {case['expected_intent']:<18} got: {result['intent']}")
        print(f"    expected tool   : {case['expected_tool']:<18} got: {result['tool_used']}")
        print(f"    expected top id : {case['expected_top_order_id']}   got: {top_id}")

    print(f"\n{passed}/{len(TEST_CASES)} test cases passed, {failed} failed.")
    assert failed == 0, f"{failed} test case(s) failed"


# ==========================================================================
# 8. ENTRY POINT
# ==========================================================================
if __name__ == "__main__":
    print(f"Backend in use -> full_text: sqlite-fts5 | semantic: "
          f"{'pinecone' if USE_REAL_PINECONE else 'mock_tfidf (offline fallback)'}")

    agent = EcommerceSupportAgent(TEST_ORDERS)
    run_sample_queries(agent)
    run_test_suite(agent)
