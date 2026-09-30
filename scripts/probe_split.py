"""Probe: can plain RAG answer the two halves of c013 when asked separately?

Usage (from project root):
    python scripts/probe_split.py
"""
import sys
from pathlib import Path

# Make "src/" importable regardless of the current working directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import time

from nexora_rag.api.deps import ROLE_DEPARTMENTS
from nexora_rag.generation.rag_service import RagService

QUESTIONS = [
    ("PART 1 (leave)", "When an employee resigns, how is unused annual leave paid in the final settlement, and at what rate per day?"),
    ("PART 2 (gratuity)", "How much gratuity does an employee get for each completed year of service, and after how many years of service does it apply?"),
    ("FULL c013", "A G3 employee with a monthly basic salary of PKR 120,000 resigns after 3 completed years of service and has 12 unused annual leave days. What is the total of leave encashment plus gratuity in the final settlement, in PKR?"),
]
PAUSE = 12


def show_sources(sources) -> str:
    """Print doc_id + page if the source is a dict, otherwise a short string."""
    parts = []
    for s in sources or []:
        if isinstance(s, dict):
            parts.append(f"{s.get('doc_id', '?')} p{s.get('page', '?')}")
        else:
            parts.append(str(s)[:60])
    return ", ".join(parts) or "(no sources)"


def main() -> None:
    departments = ROLE_DEPARTMENTS["admin"]  # all departments
    service = RagService()
    service.answer("warm up", departments=departments)  # not counted: loads models

    for name, question in QUESTIONS:
        time.sleep(PAUSE)
        result = service.answer(question, departments=departments)
        if isinstance(result, dict):
            answer, sources = result.get("answer", ""), result.get("sources")
        else:
            answer, sources = getattr(result, "answer", str(result)), getattr(result, "sources", None)
        print(f"\n===== {name} =====")
        print(f"Q: {question}\n")
        print(f"A: {answer}\n")
        print(f"SOURCES: {show_sources(sources)}")


if __name__ == "__main__":
    main()