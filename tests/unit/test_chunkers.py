import json

from nexora_rag.core.config import settings
from nexora_rag.ingestion.chunkers import (
    chunk_document,
    create_chunk_id,
    detect_chunk_type,
)


def make_page(page=1, content="Short page.", doc_id="NX-TEST-001"):
    return {
        "doc_id": doc_id,
        "title": "Test Policy",
        "department": "HR",
        "version": "1.0",
        "effective_date": "2026-01-01",
        "page": page,
        "source_pdf": "test.pdf",
        "image_count": 0,
        "table_count": 0,
        "content_hash": "sha256:page-level-hash",
        "chunk_type_hint": "prose",
        "content": content,
    }


def write_pages(tmp_path, pages):
    (tmp_path / "clean_pages.json").write_text(json.dumps(pages), encoding="utf-8")
    return tmp_path


# ---------- chunk id ----------

def test_chunk_id_is_deterministic():
    assert create_chunk_id("NX-1", 2, 0, "text") == create_chunk_id("NX-1", 2, 0, "text")


def test_chunk_id_changes_with_content_page_and_index():
    base = create_chunk_id("NX-1", 2, 0, "text")
    assert create_chunk_id("NX-1", 2, 0, "other text") != base
    assert create_chunk_id("NX-1", 3, 0, "text") != base
    assert create_chunk_id("NX-1", 2, 1, "text") != base


# ---------- chunk type ----------

def test_detect_chunk_type():
    assert detect_chunk_type("plain prose") == "prose"
    assert detect_chunk_type("Tables:\n| a | b |") == "table"
    assert detect_chunk_type("Image OCR: some text") == "image_ocr"
    assert detect_chunk_type("Tables: x\nImage OCR: y") == "mixed"


# ---------- chunk_document ----------

def test_missing_clean_pages_returns_empty(tmp_path):
    assert chunk_document(tmp_path) == []


def test_short_page_is_one_chunk_with_full_metadata(tmp_path):
    write_pages(tmp_path, [make_page(content="Paid maternity leave is 90 calendar days.")])

    chunks = chunk_document(tmp_path)

    assert len(chunks) == 1
    chunk = chunks[0]
    meta = chunk["metadata"]
    assert chunk["content"] == "Paid maternity leave is 90 calendar days."
    assert meta["doc_id"] == "NX-TEST-001"
    assert meta["department"] == "HR"
    assert meta["page"] == 1
    assert meta["chunk_index"] == 0
    assert meta["chunk_type"] == "prose"
    assert meta["chunk_size"] == len(chunk["content"])
    # per-chunk hash replaces the page hash, the page hash is kept for reference
    assert meta["content_hash"].startswith("sha256:")
    assert meta["content_hash"] != "sha256:page-level-hash"
    assert meta["page_content_hash"] == "sha256:page-level-hash"
    assert chunk["chunk_id"].startswith("NX-TEST-001-p1-c0-")


def test_long_page_is_split_and_every_chunk_fits_the_limit(tmp_path):
    limit = settings.chunk_tokens * 4  # create_splitter uses ~4 characters per token
    sentences = [f"Sentence number {i} explains one rule of the leave policy." for i in range(400)]
    write_pages(tmp_path, [make_page(content=" ".join(sentences))])

    chunks = chunk_document(tmp_path)

    assert len(chunks) > 1
    assert all(len(c["content"]) <= limit for c in chunks)
    assert [c["metadata"]["chunk_index"] for c in chunks] == list(range(len(chunks)))
    ids = [c["chunk_id"] for c in chunks]
    assert len(ids) == len(set(ids)), "chunk ids must be unique"


def test_chunk_index_restarts_for_each_page(tmp_path):
    write_pages(tmp_path, [make_page(page=1, content="Page one."), make_page(page=2, content="Page two.")])

    chunks = chunk_document(tmp_path)

    assert [(c["metadata"]["page"], c["metadata"]["chunk_index"]) for c in chunks] == [(1, 0), (2, 0)]


def test_blank_page_produces_no_chunks(tmp_path):
    write_pages(tmp_path, [make_page(content="  \n  ")])
    assert chunk_document(tmp_path) == []


def test_table_text_is_marked_as_table_chunk(tmp_path):
    write_pages(tmp_path, [make_page(content="Tables:\n| Grade | Cap |\n| G4 | 25000 |")])

    chunks = chunk_document(tmp_path)

    assert chunks[0]["metadata"]["chunk_type"] == "table"
