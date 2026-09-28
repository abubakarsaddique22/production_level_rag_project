"""
Hybrid retrieval: dense (Qdrant) + sparse (BM25) combined with
Reciprocal Rank Fusion (RRF).

Dense search ranks by meaning; BM25 ranks by literal keyword overlap.
Their raw scores are on completely different scales (cosine similarity
0-1 vs an unbounded BM25 score), so they can't just be averaged. RRF
sidesteps this by only looking at RANK POSITION in each list, not the
raw score:

    fused_score(chunk) = sum, over every ranking that contains it,
                          of  1 / (k + rank_in_that_ranking)

k=60 is the standard RRF constant from the original paper
(Cormack et al., 2009, SIGIR).
"""

from __future__ import annotations

from ..core.config import settings
from ..core.logging import get_logger
from .sparse import SparseIndex
from .vector_store import VectorStore

log = get_logger(__name__)


def reciprocal_rank_fusion(
    rankings: list[list[str]],
    k: int = 60,
) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)

    return sorted(scores.items(), key=lambda pair: pair[1], reverse=True)


class HybridRetriever:
    """Combines VectorStore (dense) and SparseIndex (BM25) via RRF."""

    def __init__(
        self,
        vector_store: VectorStore | None = None,
        sparse_index: SparseIndex | None = None,
        rrf_k: int | None = None,
        candidates_per_source: int = 30,
    ):
        self.vector_store = vector_store or VectorStore()
        self.sparse_index = sparse_index or SparseIndex()
        self.rrf_k = rrf_k or settings.rrf_k
        self.candidates_per_source = candidates_per_source

    def search(
        self,
        query: str,
        top_k: int = 5,
        departments: list[str] | None = None,
    ) -> list[dict]:
        dense_results = self.vector_store.search_by_text(
            query,
            top_k=self.candidates_per_source,
            department_filter=departments,
        )
        dense_ranking = [r.payload["chunk_id"] for r in dense_results]
        dense_payload_by_id = {r.payload["chunk_id"]: r.payload for r in dense_results}

        sparse_results = self.sparse_index.search(
            query,
            top_k=self.candidates_per_source,
            departments=departments,
        )
        sparse_ranking = [chunk_id for chunk_id, _score in sparse_results]

        fused = reciprocal_rank_fusion([dense_ranking, sparse_ranking], k=self.rrf_k)

        output: list[dict] = []
        for chunk_id, fused_score in fused:
            payload = dense_payload_by_id.get(chunk_id)

            if payload is None:
                chunk = self.sparse_index.get_chunk(chunk_id)
                payload = {
                    "chunk_id": chunk_id,
                    "content": chunk["content"] if chunk else "",
                    **(chunk.get("metadata", {}) if chunk else {}),
                }

            # Safety net: never return a chunk outside the user's departments,
            # even if a filter upstream failed.
            if departments is not None and payload.get("department") not in departments:
                log.warning(
                    "rbac_safety_net_dropped_chunk",
                    extra={"chunk_id": chunk_id, "department": payload.get("department")},
                )
                continue

            output.append({**payload, "fused_score": fused_score})
            if len(output) >= top_k:
                break

        return output

if __name__ == "__main__":
    retriever = HybridRetriever()

    test_query = "How many calendar days does an employee have to submit an expense claim?"
    results = retriever.search(test_query, top_k=5)

    print(f"Query: {test_query!r}\n")
    for r in results:
        print(
            f"  {r['fused_score']:.4f}  {r.get('doc_id')}  page {r.get('page')}  "
            f"-> {r['content'][:80]}"
        )