"""Run the full evaluation for one configuration and log it.

What it runs (all on data/eval/golden_dataset.json):
  1. retrieval: dense vs hybrid vs hybrid+rerank (hit@k, MRR, recall@k)
  2. DeepEval contextual precision / recall / relevancy (on hybrid+rerank)
  3. DeepEval faithfulness / answer relevancy on real RagService answers

It appends ONE row to docs/eval/results.csv and saves the details to
docs/eval/runs/<label>.json.

Usage (from project root):
    python scripts/run_eval.py <label> [top_k] [pause_seconds] [model_name]

Example:
    python scripts/run_eval.py baseline_k5 5 12 gpt-oss-120b

Flush Redis before every run, otherwise cached answers from an older
configuration are reused:
    docker compose exec redis redis-cli FLUSHALL
"""
import sys
from pathlib import Path

# Make "src/" importable regardless of the current working directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import csv
import json
from datetime import date

from nexora_rag.evaluation.metrics import run_full_eval
from nexora_rag.evaluation.ragas_runner import run_generation_eval

RESULTS_CSV = Path("docs/eval/results.csv")
RUNS_DIR = Path("docs/eval/runs")

COLUMNS = [
    "label", "date", "model", "top_k", "n_questions", "n_skipped",
    "dense_hit", "dense_mrr", "dense_recall",
    "hybrid_hit", "hybrid_mrr", "hybrid_recall",
    "rerank_hit", "rerank_mrr", "rerank_recall",
    "ctx_precision", "ctx_recall", "ctx_relevancy",
    "faithfulness", "answer_relevancy",
]


def label_exists(label: str) -> bool:
    """A label is used once, so a saved baseline is never overwritten by accident."""
    if not RESULTS_CSV.exists():
        return False
    with RESULTS_CSV.open(encoding="utf-8", newline="") as f:
        return any(r["label"] == label for r in csv.DictReader(f))


def stage_columns(name: str, summary: dict, top_k: int) -> dict:
    return {
        f"{name}_hit": round(summary[f"hit@{top_k}"], 3),
        f"{name}_mrr": round(summary["mrr"], 3),
        f"{name}_recall": round(summary[f"recall@{top_k}"], 3),
    }


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        return
    label = sys.argv[1]
    top_k = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    pause = float(sys.argv[3]) if len(sys.argv) > 3 else 12.0
    model = sys.argv[4] if len(sys.argv) > 4 else "unknown"

    if label_exists(label):
        print(f"Label '{label}' already exists in {RESULTS_CSV}. Use a new label.")
        return

    retrieval = run_full_eval(top_k=top_k)  # 3 retrieval stages + DeepEval contextual
    generation = run_generation_eval(top_k=top_k, pause=pause)

    stages = retrieval["stages"]
    rerank = stages["rerank"]["summary"]
    ctx = retrieval["contextual"]["summary"]
    gen = generation["summary"]

    row = {
        "label": label,
        "date": str(date.today()),
        "model": model,
        "top_k": top_k,
        "n_questions": rerank["n_questions"],
        "n_skipped": rerank["n_skipped_unanswerable"],
        **stage_columns("dense", stages["dense"]["summary"], top_k),
        **stage_columns("hybrid", stages["hybrid"]["summary"], top_k),
        **stage_columns("rerank", rerank, top_k),
        "ctx_precision": round(ctx["avg_contextual_precision"], 3),
        "ctx_recall": round(ctx["avg_contextual_recall"], 3),
        "ctx_relevancy": round(ctx["avg_contextual_relevancy"], 3),
        "faithfulness": round(gen["avg_faithfulness"], 3),
        "answer_relevancy": round(gen["avg_answer_relevancy"], 3),
    }

    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    new_file = not RESULTS_CSV.exists()
    with RESULTS_CSV.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)

    # details for the report: failing questions and per-question scores
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    details = {
        "row": row,
        "retrieval_failures": {
            name: [f["id"] for f in data["failures"]] for name, data in stages.items()
        },
        "contextual": retrieval["contextual"]["per_question"],
        "generation": generation["per_question"],
    }
    (RUNS_DIR / f"{label}.json").write_text(
        json.dumps(details, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\nSaved row to {RESULTS_CSV} and details to {RUNS_DIR / (label + '.json')}")
    for key in COLUMNS:
        print(f"  {key:18s}: {row[key]}")


if __name__ == "__main__":
    main()