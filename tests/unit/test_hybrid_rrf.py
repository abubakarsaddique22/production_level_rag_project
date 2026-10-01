from types import SimpleNamespace

import pytest

from nexora_rag.retrieval.hybrid import HybridRetriever, reciprocal_rank_fusion

# ---------- reciprocal_rank_fusion ----------


def test_single_ranking_keeps_its_order():
    fused = reciprocal_rank_fusion([["a", "b", "c"]])
    assert [cid for cid, _ in fused] == ["a", "b", "c"]


def test_chunk_found_by_both_rankings_wins():
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["c", "d"]], k=60)
    ids = [cid for cid, _ in fused]
    assert ids[0] == "c"  # rank 3 in one list + rank 1 in the other
    assert ids[1] == "a"
    scores = dict(fused)
    assert scores["c"] == pytest.approx(1 / 63 + 1 / 61)


def test_empty_rankings():
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []


def test_scores_are_sorted_best_first():
    scores = [s for _, s in reciprocal_rank_fusion([["a", "b", "c"], ["b", "c", "a"]])]
    assert scores == sorted(scores, reverse=True)


# ---------- HybridRetriever with fake backends (no Qdrant, no models) ----------


def point(chunk_id, department, content="dense text"):
    return SimpleNamespace(
        payload={"chunk_id": chunk_id, "department": department, "content": content}
    )


class FakeVectorStore:
    def __init__(self, results):
        self.results = results
        self.department_filters = []

    def search_by_text(self, query, top_k, department_filter=None):
        self.department_filters.append(department_filter)
        return self.results


class FakeSparseIndex:
    def __init__(self, ranked, chunks=None):
        self.ranked = ranked
        self.chunks = chunks or {}
        self.department_filters = []

    def search(self, query, top_k, departments=None):
        self.department_filters.append(departments)
        return self.ranked

    def get_chunk(self, chunk_id):
        return self.chunks.get(chunk_id)


def sparse_chunk(content, department):
    return {"content": content, "metadata": {"department": department, "doc_id": "NX-S"}}


def test_fusion_order_and_sparse_only_chunk_is_built_from_sparse_index():
    dense = FakeVectorStore([point("A", "HR"), point("B", "HR")])
    sparse = FakeSparseIndex(
        ranked=[("B", 2.0), ("C", 1.0)],
        chunks={"C": sparse_chunk("keyword text", "HR")},
    )
    retriever = HybridRetriever(vector_store=dense, sparse_index=sparse)

    results = retriever.search("question", top_k=3, departments=["HR"])

    assert [r["chunk_id"] for r in results] == ["B", "A", "C"]
    assert all("fused_score" in r for r in results)
    assert results[2]["content"] == "keyword text"
    assert results[2]["department"] == "HR"


def test_departments_are_passed_to_both_backends():
    dense = FakeVectorStore([])
    sparse = FakeSparseIndex([])
    HybridRetriever(vector_store=dense, sparse_index=sparse).search(
        "q", departments=["HR", "Product"]
    )
    assert dense.department_filters == [["HR", "Product"]]
    assert sparse.department_filters == [["HR", "Product"]]


def test_safety_net_drops_dense_chunk_from_a_forbidden_department():
    dense = FakeVectorStore([point("F", "Finance")])  # upstream filter "failed"
    retriever = HybridRetriever(vector_store=dense, sparse_index=FakeSparseIndex([]))
    assert retriever.search("q", departments=["HR"]) == []


def test_safety_net_drops_sparse_chunk_from_a_forbidden_department():
    sparse = FakeSparseIndex(
        ranked=[("S", 1.0)], chunks={"S": sparse_chunk("secret", "Finance")}
    )
    retriever = HybridRetriever(vector_store=FakeVectorStore([]), sparse_index=sparse)
    assert retriever.search("q", departments=["HR"]) == []


def test_top_k_limits_the_output():
    dense = FakeVectorStore([point(f"c{i}", "HR") for i in range(10)])
    retriever = HybridRetriever(vector_store=dense, sparse_index=FakeSparseIndex([]))
    assert len(retriever.search("q", top_k=3, departments=["HR"])) == 3


def test_no_departments_means_no_filtering():
    dense = FakeVectorStore([point("F", "Finance")])
    retriever = HybridRetriever(vector_store=dense, sparse_index=FakeSparseIndex([]))
    assert [r["chunk_id"] for r in retriever.search("q", departments=None)] == ["F"]
