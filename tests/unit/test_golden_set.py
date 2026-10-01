import json
import re
from pathlib import Path

import pytest

from nexora_rag.core.config import settings
from nexora_rag.evaluation.dataset import load_golden_set

ROOT = Path(__file__).resolve().parents[2]
GOLDEN = ROOT / "data" / "eval" / "golden_dataset.json"
PROCESSED = Path(settings.data_processed_dir)

# chunk id = <doc_id>-p<page>-c<index>-<12 hex chars of the content hash>
CHUNK_ID_RE = re.compile(r"^.+-p\d+-c\d+-[0-9a-f]{12}$")


def make_record(i, **overrides):
    record = {
        "id": f"q{i:03d}",
        "query": "What is X?",
        "ground_truth_answer": "X is Y.",
        "relevant_chunk_ids": ["NX-TEST-001-p1-c0-abcdef123456"],
    }
    record.update(overrides)
    return record


def write(tmp_path, records, name="golden.json"):
    path = tmp_path / name
    path.write_text(json.dumps(records), encoding="utf-8")
    return path


# ---------- the validation logic (small fake files) ----------

def test_valid_array_loads(tmp_path):
    items = load_golden_set(write(tmp_path, [make_record(1), make_record(2)]))
    assert [i.id for i in items] == ["q001", "q002"]


def test_jsonl_is_accepted(tmp_path):
    path = tmp_path / "golden.jsonl"
    path.write_text("\n".join(json.dumps(make_record(i)) for i in (1, 2)), encoding="utf-8")
    assert len(load_golden_set(path)) == 2


def test_duplicate_ids_are_rejected(tmp_path):
    path = write(tmp_path, [make_record(1), make_record(1)])
    with pytest.raises(ValueError, match="Duplicate"):
        load_golden_set(path)


def test_missing_required_field_is_rejected(tmp_path):
    bad = make_record(1)
    del bad["ground_truth_answer"]
    with pytest.raises(ValueError, match="validation failed"):
        load_golden_set(write(tmp_path, [bad]))


def test_empty_file_is_rejected(tmp_path):
    path = tmp_path / "golden.json"
    path.write_text("  \n", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        load_golden_set(path)


def test_unanswerable_question_without_chunk_ids_is_allowed(tmp_path):
    items = load_golden_set(write(tmp_path, [make_record(1, relevant_chunk_ids=[])]))
    assert items[0].relevant_chunk_ids == []


# ---------- the real golden set (runs in CI too) ----------

@pytest.fixture(scope="module")
def items():
    return load_golden_set(GOLDEN)


def test_real_golden_set_loads_with_at_least_30_questions(items):
    assert len(items) >= 30


def test_real_questions_and_answers_are_not_blank(items):
    for item in items:
        assert item.query.strip(), item.id
        assert item.ground_truth_answer.strip(), item.id


def test_real_chunk_ids_have_the_expected_shape(items):
    for item in items:
        for chunk_id in item.relevant_chunk_ids:
            assert CHUNK_ID_RE.match(chunk_id), f"{item.id}: {chunk_id}"


# ---------- local only: data/processed is not in git ----------

@pytest.mark.skipif(not PROCESSED.exists(), reason="data/processed is not in git, so CI skips this")
def test_every_golden_chunk_id_exists_in_the_processed_chunks(items):
    existing = set()
    for chunks_file in PROCESSED.glob("*/chunks.json"):
        existing.update(c["chunk_id"] for c in json.loads(chunks_file.read_text(encoding="utf-8")))

    wanted = {cid for item in items for cid in item.relevant_chunk_ids}
    missing = wanted - existing
    assert not missing, f"golden chunk ids not found in processed chunks: {sorted(missing)}"