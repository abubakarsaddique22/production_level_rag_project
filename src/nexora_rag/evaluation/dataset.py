"""
Golden set loading and validation (Step D / Step V support).

Single source of truth for reading data/eval/golden_dataset.json.
Both metrics.py (exact-match + DeepEval contextual metrics) and
ragas_runner.py (end-to-end generation metrics) should import
`load_golden_set` from here instead of re-implementing loading logic.

Every record is validated against `GoldenSetItem` on load, so a
malformed or incomplete record fails loudly at load time instead of
silently causing a KeyError deep inside a metric loop later.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator

from ..core.logging import get_logger

log = get_logger(__name__)

GOLDEN_SET_PATH = Path("data/eval/golden_dataset.json")


class GoldenSetItem(BaseModel):
    """One golden-set question, matching golden_dataset.json's record shape."""

    id: str
    query: str
    ground_truth_answer: str
    relevant_chunk_ids: list[str] = Field(default_factory=list)
    relevant_contexts: list[str] = Field(default_factory=list)
    category: str | None = None
    difficulty: str | None = None
    metadata: dict[str, Any] | None = None

    @field_validator("relevant_chunk_ids")
    @classmethod
    def warn_if_no_relevant_chunks(cls, v: list[str], info) -> list[str]:
        # Not a hard failure -- some questions are intentionally
        # "unanswerable" / "access control" cases with no relevant chunk.
        # But it's worth logging so it's not mistaken for a data bug.
        return v


class GoldenSet(BaseModel):
    """The full set, with a duplicate-id check across all items."""

    items: list[GoldenSetItem]

    @field_validator("items")
    @classmethod
    def no_duplicate_ids(cls, items: list[GoldenSetItem]) -> list[GoldenSetItem]:
        seen: dict[str, int] = {}
        for i, item in enumerate(items):
            if item.id in seen:
                raise ValueError(
                    f"Duplicate golden-set id '{item.id}' at index {i} "
                    f"(first seen at index {seen[item.id]})"
                )
            seen[item.id] = i
        return items


def _read_raw(path: Path) -> list[dict]:
    """Accepts either a JSON array or true JSONL (one object per line)."""
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Golden set file is empty: {path}")
    if text.startswith("["):
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def load_golden_set(path: Path = GOLDEN_SET_PATH) -> list[GoldenSetItem]:
    """Load + validate the golden set. Raises on the first bad record,
    with the record's index and id (if available) so it's easy to find
    and fix in the source file.

    Returns a list of GoldenSetItem (attribute access: item.query,
    item.relevant_chunk_ids, ...), not raw dicts.
    """
    raw_records = _read_raw(path)

    try:
        golden_set = GoldenSet(items=raw_records)
    except Exception as exc:
        raise ValueError(
            f"Golden set validation failed for {path}: {exc}"
        ) from exc

    n_no_relevant = sum(1 for i in golden_set.items if not i.relevant_chunk_ids)
    if n_no_relevant:
        log.info(
            "golden_set_loaded_with_unanswerable_items",
            extra={
                "path": str(path),
                "total": len(golden_set.items),
                "no_relevant_chunk_ids": n_no_relevant,
            },
        )

    log.info(
        "golden_set_loaded",
        extra={"path": str(path), "n_questions": len(golden_set.items)},
    )

    return golden_set.items


if __name__ == "__main__":
    items = load_golden_set()
    print(f"Loaded {len(items)} golden-set questions from {GOLDEN_SET_PATH}")
    categories = sorted({i.category for i in items if i.category})
    print(f"Categories: {categories}")
    n_unanswerable = sum(1 for i in items if not i.relevant_chunk_ids)
    print(f"Questions with no relevant_chunk_ids (unanswerable/access-control): {n_unanswerable}")