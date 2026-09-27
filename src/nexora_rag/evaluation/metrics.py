# """
# Retrieval evaluation (Step J).

# Two complementary styles, both driven by the SAME golden set
# (data/eval/golden_dataset.json):

# 1. Exact-match metrics -- hit@k, MRR, recall@k.
#    Deterministic, free, no LLM calls. Uses `relevant_chunk_ids` to check
#    whether the CORRECT chunk(s) were retrieved, and how high they ranked.

# 2. DeepEval contextual metrics -- Contextual Precision, Contextual Recall,
#    Contextual Relevancy. LLM-judge based. Uses `relevant_contexts` /
#    `ground_truth_answer` to judge retrieval QUALITY more semantically
#    (e.g. "is the retrieved text actually relevant", not just "did the
#    exact same chunk_id come back").

# Generation isn't built yet (that's Step K/L), so `actual_output` in each
# DeepEval test case is set to the golden set's own ground_truth_answer as
# a stand-in. ContextualPrecision/Recall score retrieval_context against
# expected_output, not against actual_output, so this placeholder does not
# distort the retrieval score. Once the real pipeline exists, Step V reruns
# the SAME metric classes with the pipeline's real generated answer to also
# get Faithfulness/AnswerRelevancy (those genuinely need actual_output).

# Requires OPENAI_API_KEY in .env for the DeepEval half (its default judge
# model). If you don't have one yet, run with use_deepeval=False -- the
# exact-match metrics alone already tell you a lot.
# """

# from __future__ import annotations

# import json
# from pathlib import Path
# from typing import Any

# from ..core.logging import get_logger
# from ..retrieval.vector_store import VectorStore

# log = get_logger(__name__)

# GOLDEN_SET_PATH = Path("data/eval/golden_set.json")


# # ---------------------------------------------------------------------
# # 1. Golden set loading
# # ---------------------------------------------------------------------

# def load_golden_set(path: Path = GOLDEN_SET_PATH) -> list[dict]:
#     """Accepts either a JSON array or true JSONL (one object per line) --
#     handles both so you don't have to reformat your existing file."""
#     text = path.read_text(encoding="utf-8").strip()
#     if text.startswith("["):
#         return json.loads(text)
#     return [json.loads(line) for line in text.splitlines() if line.strip()]


# # ---------------------------------------------------------------------
# # 2. Exact-match retrieval metrics (hit@k, MRR, recall@k) -- free, no LLM
# # ---------------------------------------------------------------------

# def _retrieve_chunk_ids(store: VectorStore, query: str, top_k: int) -> list[str]:
#     results = store.search_by_text(query, top_k=top_k)
#     return [r.payload["chunk_id"] for r in results]


# def hit_at_k(retrieved_ids: list[str], relevant_ids: list[str]) -> int:
#     """1 if ANY relevant chunk is in the retrieved list, else 0."""
#     return int(any(rid in retrieved_ids for rid in relevant_ids))


# def reciprocal_rank(retrieved_ids: list[str], relevant_ids: list[str]) -> float:
#     """1 / rank of the FIRST relevant chunk found (0 if none found)."""
#     for rank, cid in enumerate(retrieved_ids, start=1):
#         if cid in relevant_ids:
#             return 1.0 / rank
#     return 0.0


# def recall_at_k(retrieved_ids: list[str], relevant_ids: list[str]) -> float:
#     """Fraction of ALL relevant chunks that were retrieved."""
#     if not relevant_ids:
#         return 0.0
#     found = sum(1 for rid in relevant_ids if rid in retrieved_ids)
#     return found / len(relevant_ids)


# def compute_exact_match_metrics(
#     golden_set: list[dict],
#     store: VectorStore,
#     top_k: int = 5,
# ) -> dict[str, Any]:
#     """Runs every golden-set question through the retriever and computes
#     hit@k / MRR / recall@k, plus a per-question breakdown -- Step V's
#     "improve" loop needs to see WHICH questions failed, not just one
#     aggregate number."""
#     per_question = []

#     for item in golden_set:
#         query = item["query"]
#         relevant_ids = item.get("relevant_chunk_ids", [])

#         retrieved_ids = _retrieve_chunk_ids(store, query, top_k)

#         per_question.append({
#             "id": item["id"],
#             "query": query,
#             "category": item.get("category"),
#             "difficulty": item.get("difficulty"),
#             "hit": hit_at_k(retrieved_ids, relevant_ids),
#             "reciprocal_rank": reciprocal_rank(retrieved_ids, relevant_ids),
#             "recall": recall_at_k(retrieved_ids, relevant_ids),
#             "retrieved_ids": retrieved_ids,
#             "relevant_ids": relevant_ids,
#         })

#     n = len(per_question) or 1
#     summary = {
#         f"hit@{top_k}": sum(p["hit"] for p in per_question) / n,
#         "mrr": sum(p["reciprocal_rank"] for p in per_question) / n,
#         f"recall@{top_k}": sum(p["recall"] for p in per_question) / n,
#         "n_questions": len(per_question),
#         "top_k": top_k,
#     }

#     failures = [p for p in per_question if p["hit"] == 0]

#     return {"summary": summary, "per_question": per_question, "failures": failures}


# # ---------------------------------------------------------------------
# # 3. DeepEval contextual metrics -- LLM-judge based
# # ---------------------------------------------------------------------

# def compute_contextual_metrics(
#     golden_set: list[dict],
#     store: VectorStore,
#     top_k: int = 5,
# ) -> dict[str, Any]:
#     """Runs DeepEval's ContextualPrecision / ContextualRecall /
#     ContextualRelevancy metrics against the retrieved context.
#     See module docstring for the actual_output placeholder explanation.
#     """
#     from deepeval.metrics import (
#         ContextualPrecisionMetric,
#         ContextualRecallMetric,
#         ContextualRelevancyMetric,
#     )
#     from deepeval.test_case import LLMTestCase

#     precision_metric = ContextualPrecisionMetric(threshold=0.7)
#     recall_metric = ContextualRecallMetric(threshold=0.7)
#     relevancy_metric = ContextualRelevancyMetric(threshold=0.7)

#     per_question = []

#     for item in golden_set:
#         query = item["query"]
#         expected_output = item["ground_truth_answer"]

#         results = store.search_by_text(query, top_k=top_k)
#         retrieval_context = [r.payload["content"] for r in results]

#         test_case = LLMTestCase(
#             input=query,
#             actual_output=expected_output,  # placeholder -- see module docstring
#             expected_output=expected_output,
#             retrieval_context=retrieval_context,
#         )

#         precision_metric.measure(test_case)
#         recall_metric.measure(test_case)
#         relevancy_metric.measure(test_case)

#         per_question.append({
#             "id": item["id"],
#             "query": query,
#             "contextual_precision": precision_metric.score,
#             "contextual_recall": recall_metric.score,
#             "contextual_relevancy": relevancy_metric.score,
#         })

#         log.info(
#             "deepeval_scored_question",
#             extra={
#                 "id": item["id"],
#                 "precision": round(precision_metric.score, 3),
#                 "recall": round(recall_metric.score, 3),
#                 "relevancy": round(relevancy_metric.score, 3),
#             },
#         )

#     n = len(per_question) or 1
#     summary = {
#         "avg_contextual_precision": sum(p["contextual_precision"] for p in per_question) / n,
#         "avg_contextual_recall": sum(p["contextual_recall"] for p in per_question) / n,
#         "avg_contextual_relevancy": sum(p["contextual_relevancy"] for p in per_question) / n,
#         "n_questions": len(per_question),
#         "top_k": top_k,
#     }

#     return {"summary": summary, "per_question": per_question}


# # ---------------------------------------------------------------------
# # 4. Entry point
# # ---------------------------------------------------------------------

# def run_retrieval_eval(
#     golden_set_path: Path = GOLDEN_SET_PATH,
#     top_k: int = 5,
#     use_deepeval: bool = True,
# ) -> dict[str, Any]:
#     golden_set = load_golden_set(golden_set_path)
#     store = VectorStore()

#     print(f"Loaded {len(golden_set)} golden-set questions from {golden_set_path}")
#     print(f"\n{'=' * 60}\nEXACT-MATCH METRICS (hit@k, MRR, recall@k)\n{'=' * 60}")

#     exact = compute_exact_match_metrics(golden_set, store, top_k=top_k)
#     for k, v in exact["summary"].items():
#         print(f"  {k:15s}: {v}")

#     if exact["failures"]:
#         print(f"\n  {len(exact['failures'])} question(s) with ZERO hit:")
#         for f in exact["failures"]:
#             print(f"    - [{f['id']}] {f['query']}")

#     result: dict[str, Any] = {"exact_match": exact}

#     if use_deepeval:
#         print(f"\n{'=' * 60}\nDEEPEVAL CONTEXTUAL METRICS (LLM-judge)\n{'=' * 60}")
#         contextual = compute_contextual_metrics(golden_set, store, top_k=top_k)
#         for k, v in contextual["summary"].items():
#             print(f"  {k:28s}: {v}")
#         result["contextual"] = contextual

#     return result


# if __name__ == "__main__":
#     run_retrieval_eval()








"""
Retrieval evaluation (Step J / Step M comparison).

Two complementary evaluation STYLES, both driven by the SAME golden set
(data/eval/golden_dataset.json):

1. Exact-match metrics -- hit@k, MRR, recall@k.
   Deterministic, free, no LLM calls. Uses `relevant_chunk_ids` to check
   whether the CORRECT chunk(s) were retrieved, and how high they ranked.
   Works with EITHER retriever (VectorStore dense-only, or
   HybridRetriever dense+BM25+RRF) via duck typing -- see _run_retriever.

2. DeepEval contextual metrics -- Contextual Precision, Contextual Recall,
   Contextual Relevancy. LLM-judge based, requires OPENAI_API_KEY.
   Generation isn't built yet (Step K/L), so `actual_output` is set to
   the golden set's own ground_truth_answer as a stand-in -- see
   compute_contextual_metrics docstring for why this doesn't distort the
   retrieval-only score.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..core.logging import get_logger
from ..retrieval.hybrid import HybridRetriever
from ..retrieval.vector_store import VectorStore
from ..retrieval.reranker import RerankingRetriever

log = get_logger(__name__)

GOLDEN_SET_PATH = Path("data/eval/golden_dataset.json")


# ---------------------------------------------------------------------
# 1. Golden set loading
# ---------------------------------------------------------------------

def load_golden_set(path: Path = GOLDEN_SET_PATH) -> list[dict]:
    """Accepts either a JSON array or true JSONL (one object per line)."""
    text = path.read_text(encoding="utf-8").strip()
    if text.startswith("["):
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


# ---------------------------------------------------------------------
# 2. Exact-match retrieval metrics (hit@k, MRR, recall@k) -- free, no LLM
# ---------------------------------------------------------------------

def _run_retriever(retriever, query: str, top_k: int) -> list[str]:
    """Returns a plain list of chunk_ids, best-first.

    Works with EITHER retriever type without the caller needing to know
    which one it is:
      - VectorStore        -> .search_by_text() -> list of ScoredPoint
                               (chunk_id lives in .payload["chunk_id"])
      - HybridRetriever     -> .search()         -> list of plain dicts
                               (chunk_id lives in ["chunk_id"])
    """
    if hasattr(retriever, "search_by_text"):
        results = retriever.search_by_text(query, top_k=top_k)
        return [r.payload["chunk_id"] for r in results]

    results = retriever.search(query, top_k=top_k)
    return [r["chunk_id"] for r in results]


def hit_at_k(retrieved_ids: list[str], relevant_ids: list[str]) -> int:
    """1 if ANY relevant chunk is in the retrieved list, else 0."""
    return int(any(rid in retrieved_ids for rid in relevant_ids))


def reciprocal_rank(retrieved_ids: list[str], relevant_ids: list[str]) -> float:
    """1 / rank of the FIRST relevant chunk found (0 if none found)."""
    for rank, cid in enumerate(retrieved_ids, start=1):
        if cid in relevant_ids:
            return 1.0 / rank
    return 0.0


def recall_at_k(retrieved_ids: list[str], relevant_ids: list[str]) -> float:
    """Fraction of ALL relevant chunks that were retrieved."""
    if not relevant_ids:
        return 0.0
    found = sum(1 for rid in relevant_ids if rid in retrieved_ids)
    return found / len(relevant_ids)


def compute_exact_match_metrics(
    golden_set: list[dict],
    retriever,
    top_k: int = 5,
) -> dict[str, Any]:
    """retriever: a VectorStore (dense-only) or a HybridRetriever
    (dense+BM25+RRF) -- see _run_retriever for how each is called."""
    per_question = []

    for item in golden_set:
        query = item["query"]
        relevant_ids = item.get("relevant_chunk_ids", [])

        retrieved_ids = _run_retriever(retriever, query, top_k)

        per_question.append({
            "id": item["id"],
            "query": query,
            "category": item.get("category"),
            "difficulty": item.get("difficulty"),
            "hit": hit_at_k(retrieved_ids, relevant_ids),
            "reciprocal_rank": reciprocal_rank(retrieved_ids, relevant_ids),
            "recall": recall_at_k(retrieved_ids, relevant_ids),
            "retrieved_ids": retrieved_ids,
            "relevant_ids": relevant_ids,
        })

    n = len(per_question) or 1
    summary = {
        f"hit@{top_k}": sum(p["hit"] for p in per_question) / n,
        "mrr": sum(p["reciprocal_rank"] for p in per_question) / n,
        f"recall@{top_k}": sum(p["recall"] for p in per_question) / n,
        "n_questions": len(per_question),
        "top_k": top_k,
    }

    failures = [p for p in per_question if p["hit"] == 0]

    return {"summary": summary, "per_question": per_question, "failures": failures}


# ---------------------------------------------------------------------
# 3. DeepEval contextual metrics -- LLM-judge based
# ---------------------------------------------------------------------

def compute_contextual_metrics(
    golden_set: list[dict],
    store: VectorStore,
    top_k: int = 5,
) -> dict[str, Any]:
    """Runs DeepEval's ContextualPrecision / ContextualRecall /
    ContextualRelevancy metrics. Requires OPENAI_API_KEY in .env.

    `actual_output` is set to ground_truth_answer as a placeholder since
    generation doesn't exist yet (Step K/L) -- ContextualPrecision/Recall
    score retrieval_context against expected_output, not actual_output,
    so this doesn't distort the retrieval-only score.
    """
    from deepeval.metrics import (
        ContextualPrecisionMetric,
        ContextualRecallMetric,
        ContextualRelevancyMetric,
    )
    from deepeval.test_case import LLMTestCase

    precision_metric = ContextualPrecisionMetric(threshold=0.7)
    recall_metric = ContextualRecallMetric(threshold=0.7)
    relevancy_metric = ContextualRelevancyMetric(threshold=0.7)

    per_question = []

    for item in golden_set:
        query = item["query"]
        expected_output = item["ground_truth_answer"]

        results = store.search_by_text(query, top_k=top_k)
        retrieval_context = [r.payload["content"] for r in results]

        test_case = LLMTestCase(
            input=query,
            actual_output=expected_output,
            expected_output=expected_output,
            retrieval_context=retrieval_context,
        )

        precision_metric.measure(test_case)
        recall_metric.measure(test_case)
        relevancy_metric.measure(test_case)

        per_question.append({
            "id": item["id"],
            "query": query,
            "contextual_precision": precision_metric.score,
            "contextual_recall": recall_metric.score,
            "contextual_relevancy": relevancy_metric.score,
        })

    n = len(per_question) or 1
    summary = {
        "avg_contextual_precision": sum(p["contextual_precision"] for p in per_question) / n,
        "avg_contextual_recall": sum(p["contextual_recall"] for p in per_question) / n,
        "avg_contextual_relevancy": sum(p["contextual_relevancy"] for p in per_question) / n,
        "n_questions": len(per_question),
        "top_k": top_k,
    }

    return {"summary": summary, "per_question": per_question}


# ---------------------------------------------------------------------
# 4. Single-retriever entry point (what you already ran for the baseline)
# ---------------------------------------------------------------------

def run_retrieval_eval(
    golden_set_path: Path = GOLDEN_SET_PATH,
    top_k: int = 5,
    use_deepeval: bool = True,
) -> dict[str, Any]:
    golden_set = load_golden_set(golden_set_path)
    store = VectorStore()

    print(f"Loaded {len(golden_set)} golden-set questions from {golden_set_path}")
    print(f"\n{'=' * 60}\nEXACT-MATCH METRICS (hit@k, MRR, recall@k)\n{'=' * 60}")

    exact = compute_exact_match_metrics(golden_set, store, top_k=top_k)
    for k, v in exact["summary"].items():
        print(f"  {k:15s}: {v}")

    if exact["failures"]:
        print(f"\n  {len(exact['failures'])} question(s) with ZERO hit:")
        for f in exact["failures"]:
            print(f"    - [{f['id']}] {f['query']}")

    result: dict[str, Any] = {"exact_match": exact}

    if use_deepeval:
        print(f"\n{'=' * 60}\nDEEPEVAL CONTEXTUAL METRICS (LLM-judge)\n{'=' * 60}")
        contextual = compute_contextual_metrics(golden_set, store, top_k=top_k)
        for k, v in contextual["summary"].items():
            print(f"  {k:28s}: {v}")
        result["contextual"] = contextual

    return result


# ---------------------------------------------------------------------
# 5. Dense vs Hybrid comparison -- THIS fills in baseline.md's next row
# ---------------------------------------------------------------------

def compare_retrievers(
    golden_set_path: Path = GOLDEN_SET_PATH,
    top_k: int = 5,
) -> dict[str, Any]:
    """Runs the SAME golden set through three progressively richer
    retrievers -- dense only, hybrid (dense+BM25+RRF), and hybrid+rerank
    (Step N cross-encoder) -- and prints a side-by-side comparison,
    including which specific questions got fixed, are still failing, or
    regressed at each stage.
    """
    golden_set = load_golden_set(golden_set_path)
    print(f"Loaded {len(golden_set)} golden-set questions\n")

    stages: dict[str, dict[str, Any]] = {}

    dense_store = VectorStore()
    print("=" * 60)
    print("STAGE 1: DENSE ONLY (baseline)")
    print("=" * 60)
    stages["dense"] = compute_exact_match_metrics(golden_set, dense_store, top_k=top_k)
    for k, v in stages["dense"]["summary"].items():
        print(f"  {k:15s}: {v}")

    hybrid_retriever = HybridRetriever(vector_store=dense_store)
    print(f"\n{'=' * 60}")
    print("STAGE 2: HYBRID (dense + BM25 + RRF)")
    print("=" * 60)
    stages["hybrid"] = compute_exact_match_metrics(golden_set, hybrid_retriever, top_k=top_k)
    for k, v in stages["hybrid"]["summary"].items():
        print(f"  {k:15s}: {v}")

    reranking_retriever = RerankingRetriever(hybrid_retriever=hybrid_retriever)
    print(f"\n{'=' * 60}")
    print("STAGE 3: HYBRID + RERANK (cross-encoder)")
    print("=" * 60)
    stages["rerank"] = compute_exact_match_metrics(golden_set, reranking_retriever, top_k=top_k)
    for k, v in stages["rerank"]["summary"].items():
        print(f"  {k:15s}: {v}")

    print(f"\n{'=' * 60}")
    print("DELTAS (each stage vs the one before it)")
    print("=" * 60)
    ordered = [("dense", "hybrid"), ("hybrid", "rerank")]
    for before, after in ordered:
        print(f"\n  {before} -> {after}:")
        for key in (f"hit@{top_k}", "mrr", f"recall@{top_k}"):
            delta = stages[after]["summary"][key] - stages[before]["summary"][key]
            print(f"    {key:15s}: {delta:+.3f}")

    failed_by_stage = {name: {f["id"] for f in data["failures"]} for name, data in stages.items()}

    print(f"\n{'=' * 60}")
    print("FAILURE MOVEMENT")
    print("=" * 60)
    for before, after in ordered:
        fixed = sorted(failed_by_stage[before] - failed_by_stage[after])
        still_failing = sorted(failed_by_stage[after])
        regressed = sorted(failed_by_stage[after] - failed_by_stage[before])
        print(f"\n  {before} -> {after}:")
        print(f"    Fixed          : {fixed or 'none'}")
        print(f"    Still failing  : {still_failing or 'none'}")
        if regressed:
            print(f"    REGRESSED      : {regressed}")

    return stages

if __name__ == "__main__":
    compare_retrievers()