"""Step V: compare two eval runs from results.csv and catch regressions.

Usage (from the repo root):
    python scripts/check_regression.py final_k3 new_label

Exit code 0 = all fine, 1 = a metric dropped by more than its tolerance.
"""
import csv
import sys
from pathlib import Path

CSV_PATH = Path("docs/eval/results.csv")

# (column, tolerance, strict)
# strict=True  -> a drop larger than the tolerance is a REGRESSION (exit code 1)
# strict=False -> only shows a WARN (very noisy metric)
METRICS = [
    ("rerank_hit", 0.02, True),
    ("rerank_mrr", 0.02, True),
    ("rerank_recall", 0.02, True),
    ("hybrid_hit", 0.02, True),
    ("dense_hit", 0.02, True),
    ("ctx_precision", 0.05, True),
    ("ctx_recall", 0.05, True),
    ("faithfulness", 0.05, True),
    ("answer_relevancy", 0.05, True),
    ("ctx_relevancy", 0.05, False),
]


def load_row(label):
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["label"] == label]
    if not rows:
        sys.exit(f"Label '{label}' not found in results.csv")
    return rows[-1]  # if a label was run twice, use the last row


def main():
    if len(sys.argv) != 3:
        sys.exit("Usage: python scripts/check_regression.py <baseline_label> <new_label>")

    base = load_row(sys.argv[1])
    new = load_row(sys.argv[2])

    # warn before an unfair comparison
    for key in ("top_k", "model", "n_questions"):
        if base[key] != new[key]:
            print(f"WARNING: {key} differs (baseline={base[key]}, new={new[key]})")

    print(f"\nBaseline: {base['label']}   New: {new['label']}\n")
    print(f"{'metric':<18}{'baseline':>10}{'new':>10}{'diff':>9}  status")
    print("-" * 56)

    regressions = 0
    for col, tol, strict in METRICS:
        b, n = float(base[col]), float(new[col])
        diff = n - b
        if diff < -tol:
            status = "REGRESSION" if strict else "WARN"
            if strict:
                regressions += 1
        else:
            status = "OK"
        print(f"{col:<18}{b:>10.3f}{n:>10.3f}{diff:>+9.3f}  {status}")

    print()
    if regressions:
        print(f"Found regression in {regressions} metric(s).")
        sys.exit(1)
    print("No regression found.")


if __name__ == "__main__":
    main()