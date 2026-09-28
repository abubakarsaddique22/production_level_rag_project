"""
Measure end-to-end RAG latency over the golden set (Step R).

Run from the project root:
    python scripts/measure_latency.py baseline
    python scripts/measure_latency.py with_cache

Appends one row per run to docs/latency_results.md.
"""

import sys
from pathlib import Path

# Make "src/" importable regardless of the current working directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import json
import time
from datetime import date

from nexora_rag.api.deps import ROLE_DEPARTMENTS
from nexora_rag.generation.rag_service import RagService

GOLDEN_SET_PATH = Path("data/eval/golden_dataset.json")
# r"C:\Users\abubakar\Downloads\nexora-rag\nexora-rag\data\eval\golden_dataset.json"
RESULTS_PATH = Path("docs/latency_results.md")


def load_questions(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8").strip()
    if text.startswith("["):  # JSON array
        records = json.loads(text)
    else:  # JSON Lines
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
    return [r["query"] for r in records]


def percentile(values: list[float], p: int) -> float:
    ordered = sorted(values)
    return ordered[round(p / 100 * (len(ordered) - 1))]


def main() -> None:
    label = sys.argv[1] if len(sys.argv) > 1 else "run"
    pause = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
    questions = load_questions(GOLDEN_SET_PATH)
    departments = ROLE_DEPARTMENTS["admin"]  # all departments

    service = RagService()
    service.answer("warm up", departments=departments)  # not counted: loads models

    times: list[float] = []
    for i, question in enumerate(questions, start=1):
        if i > 1:
            time.sleep(pause)  # outside the timed section
        start = time.perf_counter()
        try:
            service.answer(question, departments=departments)
        except Exception as exc:
            print(f"[{i}/{len(questions)}] FAILED: {exc}")
            continue
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        print(f"[{i}/{len(questions)}] {elapsed:.2f}s")

    if not times:
        print("No successful requests.")
        return

    p50 = percentile(times, 50)
    p95 = percentile(times, 95)
    mean = sum(times) / len(times)
    print(f"\n{label}: n={len(times)}  p50={p50:.2f}s  p95={p95:.2f}s  mean={mean:.2f}s")

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not RESULTS_PATH.exists():
        RESULTS_PATH.write_text(
            "| Run | Date | N | p50 (s) | p95 (s) | Mean (s) |\n"
            "|---|---|---|---|---|---|\n",
            encoding="utf-8",
        )
    with RESULTS_PATH.open("a", encoding="utf-8") as f:
        f.write(f"| {label} | {date.today()} | {len(times)} | {p50:.2f} | {p95:.2f} | {mean:.2f} |\n")


if __name__ == "__main__":
    main()