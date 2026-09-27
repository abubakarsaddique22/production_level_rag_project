"""
Retrieval evaluation -- COMPONENT LEVEL (Step J baseline / Step M-N-O comparison).

Scope of this file, on purpose: RETRIEVAL ONLY.
    - Exact-match metrics -- hit@k, MRR, recall@k. Deterministic, free,
      no LLM calls. Uses `relevant_chunk_ids` to check whether the
      CORRECT chunk(s) were retrieved, and how high they ranked.
    - DeepEval contextual metrics -- Contextual Precision, Contextual
      Recall, Contextual Relevancy. LLM-judge based (DeepEval, not
      RAGAS). These still measure retrieval quality, just semantically
      ("is the retrieved text actually relevant") instead of by exact
      chunk_id match.

Generation-level metrics (Faithfulness, Answer Relevancy) are NOT here.
Those need a real `actual_output` from the generation pipeline, which
only exists once retrieval + generation run together end to end --
that lives in ragas_runner.py (despite the historical file name, it
also uses DeepEval; no RAGAS dependency in this project).

Golden-set loading + validation is centralized in dataset.py. This
file only consumes GoldenSetItem objects from there -- no duplicate
loading logic.

Requires OPENAI_API_KEY in .env for the DeepEval half (its default
judge model). If you don't have one yet, run with use_deepeval=False --
the exact-match metrics alone already tell you a lot.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..core.logging import get_logger
from ..retrieval.hybrid import HybridRetriever
from ..retrieval.reranker import RerankingRetriever
from ..retrieval.vector_store import VectorStore
from .dataset import GOLDEN_SET_PATH, GoldenSetItem, load_golden_set
from .judge import get_judge

log = get_logger(__name__)


# ---------------------------------------------------------------------
# 1. Exact-match retrieval metrics (hit@k, MRR, recall@k) -- free, no LLM
# ---------------------------------------------------------------------

def _run_retriever(retriever, query: str, top_k: int) -> list[str]:
    """Returns a plain list of chunk_ids, best-first.

    Works with EITHER retriever type without the caller needing to know
    which one it is:
      - VectorStore         -> .search_by_text() -> list of ScoredPoint
                                (chunk_id lives in .payload["chunk_id"])
      - HybridRetriever /
        RerankingRetriever   -> .search()         -> list of plain dicts
                                (chunk_id lives in ["chunk_id"])
    """
    if hasattr(retriever, "search_by_text"):
        results = retriever.search_by_text(query, top_k=top_k)
        return [r.payload["chunk_id"] for r in results]

    results = retriever.search(query, top_k=top_k)
    return [r["chunk_id"] for r in results]


def _run_retriever_with_content(retriever, query: str, top_k: int) -> list[str]:
    """Same duck-typing as _run_retriever, but returns the chunk TEXT
    instead of chunk_id -- needed for DeepEval contextual metrics, which
    judge the actual retrieved content, not the id."""
    if hasattr(retriever, "search_by_text"):
        results = retriever.search_by_text(query, top_k=top_k)
        return [r.payload["content"] for r in results]

    results = retriever.search(query, top_k=top_k)
    return [r["content"] for r in results]


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
    golden_set: list[GoldenSetItem],
    retriever,
    top_k: int = 5,
) -> dict[str, Any]:
    """retriever: a VectorStore (dense-only), HybridRetriever
    (dense+BM25+RRF), or RerankingRetriever (hybrid+cross-encoder) --
    see _run_retriever for how each is called.

    Skips items with no relevant_chunk_ids (unanswerable / access-
    control questions in the golden set) -- there's nothing to hit
    against, so counting them as misses would understate retrieval
    quality on the questions that ARE meant to retrieve something.
    """
    per_question = []
    skipped = 0

    for item in golden_set:
        if not item.relevant_chunk_ids:
            skipped += 1
            continue

        retrieved_ids = _run_retriever(retriever, item.query, top_k)

        per_question.append({
            "id": item.id,
            "query": item.query,
            "category": item.category,
            "difficulty": item.difficulty,
            "hit": hit_at_k(retrieved_ids, item.relevant_chunk_ids),
            "reciprocal_rank": reciprocal_rank(retrieved_ids, item.relevant_chunk_ids),
            "recall": recall_at_k(retrieved_ids, item.relevant_chunk_ids),
            "retrieved_ids": retrieved_ids,
            "relevant_ids": item.relevant_chunk_ids,
        })

    n = len(per_question) or 1
    summary = {
        f"hit@{top_k}": sum(p["hit"] for p in per_question) / n,
        "mrr": sum(p["reciprocal_rank"] for p in per_question) / n,
        f"recall@{top_k}": sum(p["recall"] for p in per_question) / n,
        "n_questions": len(per_question),
        "n_skipped_unanswerable": skipped,
        "top_k": top_k,
    }

    failures = [p for p in per_question if p["hit"] == 0]

    return {"summary": summary, "per_question": per_question, "failures": failures}


# ---------------------------------------------------------------------
# 2. DeepEval contextual metrics -- LLM-judge based (retrieval quality,
#    scored semantically instead of by exact chunk_id match)
#
#    Uses DeepEval's `evaluate()` harness (the framework's own
#    recommended entry point) instead of manually calling
#    `metric.measure()` in a loop:
#      - runs all test cases x all metrics in parallel (30 questions x
#        3 metrics = 90 judge calls -- a manual loop does these one at
#        a time, evaluate() batches them)
#      - `hyperparameters` tags the run (retriever type, top_k, ...) so
#        a dense-vs-hybrid-vs-rerank comparison is self-documenting in
#        the printed report, not just in our own summary dict
# ---------------------------------------------------------------------

def compute_contextual_metrics(
    golden_set: list[GoldenSetItem],
    retriever,
    top_k: int = 5,
    retriever_name: str = "dense",
) -> dict[str, Any]:
    """Runs DeepEval's ContextualPrecision / ContextualRecall /
    ContextualRelevancy metrics against the retrieved context, via
    `deepeval.evaluate()`.

    `actual_output` is set to `ground_truth_answer` as a placeholder,
    because generation doesn't run in this file -- see module
    docstring. ContextualPrecision/Recall score `retrieval_context`
    against `expected_output`, not `actual_output`, so this placeholder
    does not distort the retrieval-only score. ContextualRelevancy
    scores `retrieval_context` against `input` only, so it isn't
    affected by actual_output either.
    """
    from deepeval import evaluate
    from deepeval.evaluate.configs import AsyncConfig
    from deepeval.metrics import (
        ContextualPrecisionMetric,
        ContextualRecallMetric,
        ContextualRelevancyMetric,
    )
    from deepeval.test_case import LLMTestCase

    judge = get_judge()

    test_cases = []
    skipped = 0

    for item in golden_set:
        if not item.relevant_chunk_ids:
            skipped += 1
            continue

        retrieval_context = _run_retriever_with_content(retriever, item.query, top_k)

        test_cases.append(
            LLMTestCase(
                input=item.query,
                actual_output=item.ground_truth_answer,  # placeholder -- see docstring
                expected_output=item.ground_truth_answer,
                retrieval_context=retrieval_context,
                # id is not a real LLMTestCase field, so we keep a
                # parallel list of ids in the same order to zip results
                # back together below.
            )
        )

    ids_in_order = [item.id for item in golden_set if item.relevant_chunk_ids]
    queries_in_order = [item.query for item in golden_set if item.relevant_chunk_ids]

    metrics = [
        ContextualPrecisionMetric(threshold=0.7, model=judge, include_reason=True),
        ContextualRecallMetric(threshold=0.7, model=judge, include_reason=True),
        ContextualRelevancyMetric(threshold=0.7, model=judge, include_reason=True),
    ]

    eval_result = evaluate(
        test_cases=test_cases,
        metrics=metrics,
        # Ollama (esp. local, and even cloud under load) tends to choke
        # if you fire too many concurrent judge calls at once -- cap it
        # instead of DeepEval's default high concurrency.
        async_config=AsyncConfig(max_concurrent=2),
        hyperparameters={
            "retriever": retriever_name,
            "top_k": top_k,
            "judge_provider": "ollama",
            "judge_model": judge.get_model_name(),
            "golden_set": str(GOLDEN_SET_PATH),
        },
    )

    per_question = []
    for tc_id, query, test_result in zip(ids_in_order, queries_in_order, eval_result.test_results):
        scores = {m.name: m.score for m in test_result.metrics_data}
        per_question.append({
            "id": tc_id,
            "query": query,
            "contextual_precision": scores.get("Contextual Precision", 0.0),
            "contextual_recall": scores.get("Contextual Recall", 0.0),
            "contextual_relevancy": scores.get("Contextual Relevancy", 0.0),
        })

    n = len(per_question) or 1
    summary = {
        "avg_contextual_precision": sum(p["contextual_precision"] for p in per_question) / n,
        "avg_contextual_recall": sum(p["contextual_recall"] for p in per_question) / n,
        "avg_contextual_relevancy": sum(p["contextual_relevancy"] for p in per_question) / n,
        "n_questions": len(per_question),
        "n_skipped_unanswerable": skipped,
        "top_k": top_k,
        "retriever": retriever_name,
    }

    log.info("deepeval_contextual_summary", extra=summary)

    return {"summary": summary, "per_question": per_question}


# ---------------------------------------------------------------------
# 3. Single-retriever entry point (baseline run)
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
        print(f"  {k:24s}: {v}")

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
# 4. Dense vs Hybrid vs Rerank comparison -- fills baseline.md's next rows
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
        print(f"  {k:24s}: {v}")

    hybrid_retriever = HybridRetriever(vector_store=dense_store)
    print(f"\n{'=' * 60}")
    print("STAGE 2: HYBRID (dense + BM25 + RRF)")
    print("=" * 60)
    stages["hybrid"] = compute_exact_match_metrics(golden_set, hybrid_retriever, top_k=top_k)
    for k, v in stages["hybrid"]["summary"].items():
        print(f"  {k:24s}: {v}")

    reranking_retriever = RerankingRetriever(hybrid_retriever=hybrid_retriever)
    print(f"\n{'=' * 60}")
    print("STAGE 3: HYBRID + RERANK (cross-encoder)")
    print("=" * 60)
    stages["rerank"] = compute_exact_match_metrics(golden_set, reranking_retriever, top_k=top_k)
    for k, v in stages["rerank"]["summary"].items():
        print(f"  {k:24s}: {v}")

    print(f"\n{'=' * 60}")
    print("DELTAS (each stage vs the one before it)")
    print("=" * 60)
    ordered = [("dense", "hybrid"), ("hybrid", "rerank")]
    for before, after in ordered:
        print(f"\n  {before} -> {after}:")
        for key in (f"hit@{top_k}", "mrr", f"recall@{top_k}"):
            delta = stages[after]["summary"][key] - stages[before]["summary"][key]
            print(f"    {key:24s}: {delta:+.3f}")

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

    return {
        "stages": stages,
        "retrievers": {
            "dense": dense_store,
            "hybrid": hybrid_retriever,
            "rerank": reranking_retriever,
        },
    }


# ---------------------------------------------------------------------
# 5. Full run: exact-match across all 3 stages + DeepEval contextual on
#    the final (best) retriever -- this is what `python -m ... metrics`
#    actually runs end to end.
# ---------------------------------------------------------------------

def run_full_eval(
    golden_set_path: Path = GOLDEN_SET_PATH,
    top_k: int = 5,
) -> dict[str, Any]:
    comparison = compare_retrievers(golden_set_path, top_k=top_k)

    golden_set = load_golden_set(golden_set_path)
    rerank_retriever = comparison["retrievers"]["rerank"]

    print(f"\n{'=' * 60}")
    print("DEEPEVAL CONTEXTUAL METRICS (LLM-judge, on hybrid+rerank)")
    print("=" * 60)
    contextual = compute_contextual_metrics(
        golden_set, rerank_retriever, top_k=top_k, retriever_name="hybrid+rerank"
    )
    for k, v in contextual["summary"].items():
        print(f"  {k:28s}: {v}")

    return {**comparison, "contextual": contextual}


if __name__ == "__main__":
    run_full_eval()