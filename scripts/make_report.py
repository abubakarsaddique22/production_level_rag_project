"""Step V: build docs/eval/report.md and charts from the eval results.

Usage (from the repo root):
    python scripts/make_report.py              # label final_k3
    python scripts/make_report.py final_k3     # any other label
"""
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save PNG files without opening a window
import matplotlib.pyplot as plt

EVAL_DIR = Path("docs/eval")
CSV_PATH = EVAL_DIR / "results.csv"
CHART_DIR = EVAL_DIR / "charts"
REPORT_PATH = EVAL_DIR / "report.md"


def load_rows():
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_run(label):
    with open(EVAL_DIR / "runs" / f"{label}.json", encoding="utf-8") as f:
        return json.load(f)


def chart_retrieval(row, path):
    """Compare the three retrievers on hit, MRR and recall."""
    names = ["dense", "hybrid", "rerank"]
    metrics = [("hit", "hit@k"), ("mrr", "MRR"), ("recall", "recall@k")]
    width = 0.25
    plt.figure(figsize=(7, 4))
    for i, (key, label) in enumerate(metrics):
        values = [float(row[f"{n}_{key}"]) for n in names]
        xs = [j + i * width for j in range(len(names))]
        bars = plt.bar(xs, values, width, label=label)
        for b, v in zip(bars, values):
            plt.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
    plt.xticks([j + width for j in range(len(names))], ["Dense", "Hybrid", "Hybrid + Rerank"])
    plt.ylim(0, 1.12)
    plt.title(f"Retrieval comparison (top_k={row['top_k']})")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def chart_generation(row, path):
    """Generation and DeepEval metrics."""
    keys = ["ctx_precision", "ctx_recall", "ctx_relevancy", "faithfulness", "answer_relevancy"]
    names = ["Ctx precision", "Ctx recall", "Ctx relevancy", "Faithfulness", "Answer relevancy"]
    values = [float(row[k]) for k in keys]
    plt.figure(figsize=(7, 4))
    bars = plt.bar(names, values, color="#4c72b0")
    for b, v in zip(bars, values):
        plt.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
    plt.ylim(0, 1.12)
    plt.xticks(rotation=20)
    plt.title("DeepEval metrics")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for r in rows:
        lines.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(lines)


def build_report(row, run):
    f = lambda key: f"{float(row[key]):.3f}"

    retrieval = table(
        ["Retriever", "hit@k", "MRR", "recall@k"],
        [
            ["Dense", f("dense_hit"), f("dense_mrr"), f("dense_recall")],
            ["Hybrid (BM25 + RRF)", f("hybrid_hit"), f("hybrid_mrr"), f("hybrid_recall")],
            ["Hybrid + Rerank", f("rerank_hit"), f("rerank_mrr"), f("rerank_recall")],
        ],
    )

    fails = run["retrieval_failures"]
    failures = table(
        ["Retriever", "Failed questions"],
        [[name, ", ".join(ids) if ids else "none"] for name, ids in fails.items()],
    )

    generation = table(
        ["Metric", "Score"],
        [
            ["Contextual precision", f("ctx_precision")],
            ["Contextual recall", f("ctx_recall")],
            ["Contextual relevancy", f("ctx_relevancy")],
            ["Faithfulness", f("faithfulness")],
            ["Answer relevancy", f("answer_relevancy")],
        ],
    )

    # weakest generation answers
    weak = [g for g in run["generation"] if g["faithfulness"] < 1 or g["answer_relevancy"] < 0.8]
    weak_table = table(
        ["Question", "Faithfulness", "Answer relevancy"],
        [[g["id"], f"{g['faithfulness']:.2f}", f"{g['answer_relevancy']:.2f}"] for g in weak],
    )

    # questions with the lowest contextual relevancy
    low_rel = sorted(run["contextual"], key=lambda r: r["contextual_relevancy"])[:5]
    low_rel_table = table(
        ["Question", "Query", "Relevancy"],
        [[r["id"], r["query"], f"{r['contextual_relevancy']:.3f}"] for r in low_rel],
    )

    history = table(
        ["Label", "Date", "top_k", "Rerank hit", "Rerank MRR", "Faithfulness", "Answer rel."],
        [
            [r["label"], r["date"], r["top_k"], r["rerank_hit"], r["rerank_mrr"],
             r["faithfulness"], r["answer_relevancy"]]
            for r in load_rows()
        ],
    )

    return f"""# Evaluation Report: {row['label']}

- Date: {row['date']}
- Answer model: `{row['model']}`
- top_k: {row['top_k']}
- Golden set: {row['n_questions']} questions ({row['n_skipped']} skipped)

## 1. Retrieval

Retrieval-only metrics (no LLM involved). Each question is matched exactly against its `relevant_chunk_ids`.

{retrieval}

![Retrieval comparison](charts/retrieval.png)

Questions each retriever got wrong:

{failures}

After reranking, no question fails (hit@{row['top_k']} = {f('rerank_hit')}).
Hybrid search fixed q005 compared to dense, and the reranker fixed q004 and q019.

## 2. Generation and DeepEval

{generation}

![DeepEval metrics](charts/generation.png)

The judge model is self-hosted on Ollama.

### Weakest generation answers

{weak_table}

### Lowest contextual relevancy

{low_rel_table}

## 3. Limitations

- **Contextual relevancy is very low ({f('ctx_relevancy')}).** Chunks are large page-level text, so they contain
  a lot of content unrelated to the question next to the relevant part. This cause has not been verified yet.
- **Judge noise.** DeepEval scores varied by roughly 0.02 to 0.03 between two runs
  (precision 0.911 vs 0.928, recall 1.0 vs 0.967), so small differences should not be read as improvements.
- **Small golden set.** There are only {row['n_questions']} questions and all of them are answerable
  (no unanswerable, adversarial or multi-document questions), so refusal accuracy is not measured here.
- Retrieval metrics are deterministic, but generation metrics depend on an LLM judge and can change a little on every run.

## 4. Run history

{history}
"""


def main():
    label = sys.argv[1] if len(sys.argv) > 1 else "final_k3"

    rows = load_rows()
    row = next((r for r in rows if r["label"] == label), None)
    if row is None:
        sys.exit(f"Label '{label}' not found in results.csv")
    run = load_run(label)

    CHART_DIR.mkdir(parents=True, exist_ok=True)
    chart_retrieval(row, CHART_DIR / "retrieval.png")
    chart_generation(row, CHART_DIR / "generation.png")

    REPORT_PATH.write_text(build_report(row, run), encoding="utf-8")
    print(f"Report written: {REPORT_PATH}")


if __name__ == "__main__":
    main()