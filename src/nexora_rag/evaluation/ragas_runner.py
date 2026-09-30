from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from ..api.deps import ROLE_DEPARTMENTS
from ..core.logging import get_logger
from ..generation.rag_service import RagService
from .dataset import GOLDEN_SET_PATH, load_golden_set
from .judge import get_judge

log = get_logger(__name__)


def run_generation_eval(
    golden_set_path: Path = GOLDEN_SET_PATH,
    top_k: int = 5,
    pause: float = 0.0,
) -> dict[str, Any]:
    """Runs every golden-set question through the REAL RagService and
    scores the real answers with DeepEval's Faithfulness and Answer
    Relevancy metrics.

    Skips items with no relevant_chunk_ids (unanswerable / access-control
    questions) -- same convention as metrics.py.

    pause: seconds to wait before each question (Groq free tier rate limit).
    """
    from deepeval import evaluate
    from deepeval.evaluate.configs import AsyncConfig
    from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
    from deepeval.test_case import LLMTestCase

    golden_set = load_golden_set(golden_set_path)
    judge = get_judge()
    service = RagService(top_k=top_k)
    departments = ROLE_DEPARTMENTS["admin"]  # all departments, like the other eval scripts

    test_cases = []
    ids_in_order = []
    queries_in_order = []
    real_answers = []
    skipped = 0

    print(f"Running {len(golden_set)} questions through the real RAG pipeline...\n")

    for item in golden_set:
        if not item.relevant_chunk_ids:
            skipped += 1
            continue

        time.sleep(pause)  # avoid Groq 429 between back-to-back questions
        response = service.answer(item.query, departments=departments)

        # Re-retrieve the full (untruncated) chunk content for the
        # faithfulness/relevancy judge -- response["sources"] only carries
        # a 200-char snippet per source. The pipeline is deterministic, so
        # this reconstructs the same context that generation actually saw.
        chunks = service.retriever.search(item.query, top_k=top_k)
        retrieval_context = [c["content"] for c in chunks]

        test_cases.append(
            LLMTestCase(
                input=item.query,
                actual_output=response["answer"],  # REAL generated answer
                expected_output=item.ground_truth_answer,
                retrieval_context=retrieval_context,
            )
        )
        ids_in_order.append(item.id)
        queries_in_order.append(item.query)
        real_answers.append(response["answer"])

        log.info(
            "generation_eval_case_ready",
            extra={"id": item.id, "answer_preview": response["answer"][:80]},
        )

    metrics = [
        FaithfulnessMetric(threshold=0.85, model=judge, include_reason=True),
        AnswerRelevancyMetric(threshold=0.85, model=judge, include_reason=True),
    ]

    eval_result = evaluate(
        test_cases=test_cases,
        metrics=metrics,
        async_config=AsyncConfig(max_concurrent=2),
        hyperparameters={
            "retriever": "hybrid+rerank",
            "top_k": top_k,
            "judge_provider": "ollama",
            "judge_model": judge.get_model_name(),
            "golden_set": str(golden_set_path),
        },
    )

    per_question = []
    for tc_id, query, answer, test_result in zip(
        ids_in_order, queries_in_order, real_answers, eval_result.test_results
    ):
        scores = {m.name: m.score for m in test_result.metrics_data}
        per_question.append({
            "id": tc_id,
            "query": query,
            "answer": answer,
            "faithfulness": scores.get("Faithfulness", 0.0),
            "answer_relevancy": scores.get("Answer Relevancy", 0.0),
        })

    n = len(per_question) or 1
    summary = {
        "avg_faithfulness": sum(p["faithfulness"] for p in per_question) / n,
        "avg_answer_relevancy": sum(p["answer_relevancy"] for p in per_question) / n,
        "n_questions": len(per_question),
        "n_skipped_unanswerable": skipped,
        "top_k": top_k,
    }

    low_faithfulness = [p for p in per_question if p["faithfulness"] < 0.85]
    low_relevancy = [p for p in per_question if p["answer_relevancy"] < 0.85]

    log.info("generation_eval_summary", extra=summary)

    return {
        "summary": summary,
        "per_question": per_question,
        "low_faithfulness": low_faithfulness,
        "low_relevancy": low_relevancy,
    }


if __name__ == "__main__":
    result = run_generation_eval()

    print(f"\n{'=' * 60}")
    print("GENERATION EVALUATION (real RagService answers)")
    print("=" * 60)
    for k, v in result["summary"].items():
        print(f"  {k:24s}: {v}")

    if result["low_faithfulness"]:
        print(f"\n  {len(result['low_faithfulness'])} question(s) below faithfulness threshold:")
        for p in result["low_faithfulness"]:
            print(f"    - [{p['id']}] {p['query']} (score={p['faithfulness']:.2f})")

    if result["low_relevancy"]:
        print(f"\n  {len(result['low_relevancy'])} question(s) below answer relevancy threshold:")
        for p in result["low_relevancy"]:
            print(f"    - [{p['id']}] {p['query']} (score={p['answer_relevancy']:.2f})")