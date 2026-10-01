from types import SimpleNamespace

import pytest

from nexora_rag.evaluation.metrics import (
    compute_exact_match_metrics,
    hit_at_k,
    recall_at_k,
    reciprocal_rank,
)

# ---------- the three metric functions ----------

def test_hit_at_k():
    assert hit_at_k(["a", "b"], ["b"]) == 1
    assert hit_at_k(["a", "b"], ["x", "b"]) == 1
    assert hit_at_k(["a", "b"], ["x"]) == 0
    assert hit_at_k([], ["x"]) == 0


def test_reciprocal_rank():
    assert reciprocal_rank(["a", "b", "c"], ["a"]) == 1.0
    assert reciprocal_rank(["a", "b", "c"], ["b"]) == 0.5
    assert reciprocal_rank(["a", "b", "c"], ["c", "b"]) == 0.5   # first relevant one counts
    assert reciprocal_rank(["a", "b"], ["x"]) == 0.0


def test_recall_at_k():
    assert recall_at_k(["a", "b"], ["a", "b"]) == 1.0
    assert recall_at_k(["a", "z"], ["a", "b"]) == 0.5
    assert recall_at_k(["z"], ["a", "b"]) == 0.0
    assert recall_at_k(["a"], []) == 0.0   # nothing relevant: defined as 0, not a crash


# ---------- compute_exact_match_metrics with a fake retriever ----------

class FakeRetriever:
    """Plain retriever (has .search, no .search_by_text), returns a fixed ranking per query."""

    def __init__(self, ranking_by_query):
        self.ranking_by_query = ranking_by_query

    def search(self, query, top_k=5):
        return [{"chunk_id": cid} for cid in self.ranking_by_query[query][:top_k]]


def item(item_id, query, relevant):
    return SimpleNamespace(
        id=item_id, query=query, category="factual", difficulty="easy", relevant_chunk_ids=relevant
    )


def test_summary_failures_and_skipped_questions():
    golden = [
        item("q1", "found", ["a"]),         # hit at rank 1
        item("q2", "missed", ["z"]),        # never retrieved
        item("q3", "unanswerable", []),     # no relevant chunk -> skipped, not a miss
    ]
    retriever = FakeRetriever({"found": ["a", "b", "c"], "missed": ["a", "b", "c"]})

    result = compute_exact_match_metrics(golden, retriever, top_k=3)

    summary = result["summary"]
    assert summary["hit@3"] == pytest.approx(0.5)
    assert summary["mrr"] == pytest.approx(0.5)
    assert summary["recall@3"] == pytest.approx(0.5)
    assert summary["n_questions"] == 2
    assert summary["n_skipped_unanswerable"] == 1
    assert [f["id"] for f in result["failures"]] == ["q2"]


def test_top_k_cuts_the_ranking():
    golden = [item("q1", "late", ["c"])]
    retriever = FakeRetriever({"late": ["a", "b", "c"]})

    assert compute_exact_match_metrics(golden, retriever, top_k=2)["summary"]["hit@2"] == 0
    assert compute_exact_match_metrics(golden, retriever, top_k=3)["summary"]["hit@3"] == 1
