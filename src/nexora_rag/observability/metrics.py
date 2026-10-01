"""Prometheus metrics (Step W): request outcomes, latency, cache hit ratio."""
from prometheus_client import Counter, Histogram, make_asgi_app

REQUESTS = Counter(
    "rag_requests_total", "RAG chat requests", ["outcome"]  # answered | refused
)
LATENCY = Histogram(
    "rag_latency_seconds",
    "End-to-end RAG latency",
    buckets=(0.5, 1, 2, 3, 5, 8, 13, 21, 34),
)
CACHE_LOOKUPS = Counter(
    "rag_cache_lookups_total", "Answer cache lookups", ["result"]  # hit | miss
)

metrics_app = make_asgi_app()