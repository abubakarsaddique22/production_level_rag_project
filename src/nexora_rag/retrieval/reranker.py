"""
Cross-encoder reranking (Step N).

Dense + BM25 (hybrid.py) retrieve a wide candidate pool cheaply but
coarsely -- they score the query and each chunk SEPARATELY then compare
vectors / keyword overlap. A cross-encoder reads the query and each
candidate chunk TOGETHER in one forward pass, so it can notice a specific
phrase buried inside an otherwise off-topic-looking chunk -- exactly the
q019 case: "submission days" living inside a chunk whose dominant topic
looks like "per diem and accommodation".

This precision costs latency, so it only ever runs on a SMALL candidate
set (top ~30 from hybrid search), never on the whole corpus.
"""

from __future__ import annotations

from sentence_transformers import CrossEncoder

from ..core.exceptions import VectorStoreError
from ..core.logging import get_logger
from .hybrid import HybridRetriever

log = get_logger(__name__)

DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker:
    """Thin wrapper around a sentence-transformers CrossEncoder."""

    def __init__(self, model_name: str = DEFAULT_RERANKER_MODEL):
        self.model_name = model_name
        log.info("loading_reranker_model", extra={"model": model_name})
        try:
            self._model = CrossEncoder(model_name)
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(
                f"Could not load reranker model '{model_name}': {exc}"
            ) from exc

    def rerank(self, query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
        """candidates: dicts with at least a "content" key (as returned by
        HybridRetriever.search()). Returns the same dicts re-sorted by
        cross-encoder score, with a new "rerank_score" field, truncated
        to top_k."""
        if not candidates:
            return []

        pairs = [(query, c["content"]) for c in candidates]
        scores = self._model.predict(pairs)

        scored = list(zip(candidates, scores))
        scored.sort(key=lambda pair: pair[1], reverse=True)

        return [{**candidate, "rerank_score": float(score)} for candidate, score in scored[:top_k]]


class RerankingRetriever:
    """Full Step M+N pipeline: hybrid retrieve a wide candidate pool
    (default 30), then cross-encoder rerank down to the final top_k."""

    def __init__(
        self,
        hybrid_retriever: HybridRetriever | None = None,
        reranker: Reranker | None = None,
        candidate_pool_size: int = 30,
    ):
        self.hybrid_retriever = hybrid_retriever or HybridRetriever()
        self.reranker = reranker or Reranker()
        self.candidate_pool_size = candidate_pool_size

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        candidates = self.hybrid_retriever.search(query, top_k=self.candidate_pool_size)
        return self.reranker.rerank(query, candidates, top_k=top_k)


if __name__ == "__main__":
    retriever = RerankingRetriever()

    test_query = "How many calendar days does an employee have to submit an expense claim?"
    results = retriever.search(test_query, top_k=5)

    print(f"Query: {test_query!r}\n")
    for r in results:
        print(f"  {r['rerank_score']:.4f}  {r.get('chunk_id')}  -> {r['content'][:80]}")