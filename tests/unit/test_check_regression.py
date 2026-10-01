import csv
import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "check_regression.py"


@pytest.fixture
def script():
    spec = importlib.util.spec_from_file_location("check_regression", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_results(path, script, rows):
    """rows: {label: {metric_column: value}}; every metric column defaults to 0.9."""
    columns = ["label", "top_k", "model", "n_questions"] + [m[0] for m in script.METRICS]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for label, overrides in rows.items():
            row = {"label": label, "top_k": 3, "model": "m", "n_questions": 30}
            row.update({m[0]: 0.9 for m in script.METRICS})
            row.update(overrides)
            writer.writerow(row)


def run(script, monkeypatch, tmp_path, rows, base="base", new="new"):
    results = tmp_path / "results.csv"
    write_results(results, script, rows)
    monkeypatch.setattr(script, "CSV_PATH", results)
    monkeypatch.setattr(sys, "argv", ["check_regression.py", base, new])
    script.main()


def test_identical_runs_pass(script, monkeypatch, tmp_path, capsys):
    run(script, monkeypatch, tmp_path, {"base": {}, "new": {}})
    assert "No regression found" in capsys.readouterr().out


def test_small_drop_inside_tolerance_passes(script, monkeypatch, tmp_path):
    run(script, monkeypatch, tmp_path, {"base": {}, "new": {"rerank_hit": 0.89}})  # -0.01, tol 0.02


def test_drop_beyond_tolerance_exits_with_code_1(script, monkeypatch, tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        run(script, monkeypatch, tmp_path, {"base": {}, "new": {"rerank_hit": 0.80}})
    assert exc.value.code == 1
    assert "REGRESSION" in capsys.readouterr().out


def test_improvement_passes(script, monkeypatch, tmp_path):
    run(script, monkeypatch, tmp_path, {"base": {}, "new": {"faithfulness": 1.0}})


def test_noisy_contextual_relevancy_only_warns(script, monkeypatch, tmp_path, capsys):
    run(script, monkeypatch, tmp_path, {"base": {}, "new": {"ctx_relevancy": 0.1}})
    out = capsys.readouterr().out
    assert "WARN" in out
    assert "No regression found" in out


def test_unknown_label_stops_with_a_message(script, monkeypatch, tmp_path):
    with pytest.raises(SystemExit) as exc:
        run(script, monkeypatch, tmp_path, {"base": {}}, new="missing")
    assert "missing" in str(exc.value.code)


def test_different_top_k_prints_a_warning(script, monkeypatch, tmp_path, capsys):
    run(script, monkeypatch, tmp_path, {"base": {}, "new": {"top_k": 5}})
    assert "top_k differs" in capsys.readouterr().out


def test_wrong_number_of_arguments_stops(script, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["check_regression.py", "only-one"])
    with pytest.raises(SystemExit):
        script.main()
