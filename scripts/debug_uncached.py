"""
Find out why some golden questions are not cached (Step R).

Lists the questions that have no cached answer, then runs the first few
of them by hand and prints the raw LLM answer next to the citations that
were found. Costs a few LLM calls.

Run from the project root:
    python scripts/debug_uncached.py
"""

import sys
from pathlib import Path

# Make "src/" importable regardless of the current working directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import json

from nexora_rag.api.deps import ROLE_DEPARTMENTS
from nexora_rag.core.cache import get_cached_answer, make_key
from nexora_rag.generation.citations import build_sources, validate_citations
from nexora_rag.generation.llm import ask_llm
from nexora_rag.generation.prompts import SYSTEM_PROMPT, build_user_message
from nexora_rag.generation.rag_service import RagService

GOLDEN_SET_PATH = Path("data/eval/golden_dataset.json")
N_TO_INSPECT = 3


def load_records(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8").strip()
    if text.startswith("["):  # JSON array
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]  # JSON Lines


def main() -> None:
    departments = ROLE_DEPARTMENTS["admin"]  # same departments as measure_latency.py
    records = load_records(GOLDEN_SET_PATH)

    uncached = [
        r for r in records
        if get_cached_answer(make_key(r["query"], departments)) is None
    ]
    print(f"Cached: {len(records) - len(uncached)}/{len(records)}")
    print("Not cached:", [r["id"] for r in uncached])

    service = RagService()
    for record in uncached[:N_TO_INSPECT]:
        question = record["query"]
        chunks = service.retriever.search(
            question, top_k=service.top_k, departments=departments
        )
        raw_answer = ask_llm(SYSTEM_PROMPT, build_user_message(question, chunks))
        result = validate_citations(raw_answer, chunks)
        sources = build_sources(chunks, result["valid"])

        print(f"\n--- {record['id']}: {question}")
        print("chunks retrieved:", len(chunks))
        print("raw answer:", raw_answer)
        print("valid citations:", result["valid"])
        print("sources found:", len(sources))


if __name__ == "__main__":
    main()