"""
Show where the time goes in one RAG request (Step R).

Times each stage separately on a few golden-set questions:
query embedding, Qdrant dense search, BM25, reranking, LLM call.

Run from the project root:
    python scripts/latency_breakdown.py
"""

import sys
from pathlib import Path

# Make "src/" importable regardless of the current working directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import json
import time

from nexora_rag.api.deps import ROLE_DEPARTMENTS
from nexora_rag.generation.llm import ask_llm
from nexora_rag.generation.prompts import SYSTEM_PROMPT, build_user_message
from nexora_rag.generation.rag_service import RagService

GOLDEN_SET_PATH = Path("data/eval/golden_dataset.json")
N_QUESTIONS = 3


def load_questions(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8").strip()
    if text.startswith("["):  # JSON array
        records = json.loads(text)
    else:  # JSON Lines
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
    return [r["query"] for r in records]


def timed(fn, *args, **kwargs):
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, time.perf_counter() - start


def main() -> None:
    questions = load_questions(GOLDEN_SET_PATH)[:N_QUESTIONS]
    departments = ROLE_DEPARTMENTS["admin"]  # all departments

    service = RagService()
    service.answer("warm up", departments=departments)  # loads all models, not counted

    hybrid = service.retriever.hybrid_retriever
    reranker = service.retriever.reranker
    vector_store = hybrid.vector_store
    pool = service.retriever.candidate_pool_size

    totals: dict[str, float] = {}
    for i, question in enumerate(questions, start=1):
        stages: dict[str, float] = {}

        vectors, stages["embed"] = timed(vector_store._embedder.encode, [question])
        _, stages["qdrant"] = timed(
            vector_store.search, vectors[0], top_k=pool, department_filter=departments
        )
        _, stages["bm25"] = timed(
            hybrid.sparse_index.search, question, top_k=pool, departments=departments
        )

        candidates = hybrid.search(question, top_k=pool, departments=departments)  # not timed
        chunks, stages["rerank"] = timed(
            reranker.rerank, question, candidates, top_k=service.top_k
        )

        user_message = build_user_message(question, chunks)
        answer, stages["llm"] = timed(ask_llm, SYSTEM_PROMPT, user_message)

        line = " | ".join(f"{name} {sec:.2f}s" for name, sec in stages.items())
        print(f"Q{i}: {line} | total {sum(stages.values()):.2f}s")
        print(f"      prompt {len(user_message)} chars, answer {len(answer)} chars")

        for name, sec in stages.items():
            totals[name] = totals.get(name, 0.0) + sec

    grand_total = sum(totals.values())
    print("\nAverage per stage:")
    for name, sec in totals.items():
        print(f"  {name:<7} {sec / len(questions):6.2f}s  ({sec / grand_total * 100:4.1f}%)")


if __name__ == "__main__":
    main()