from nexora_rag.generation.citations import (
    build_sources,
    extract_citations,
    validate_citations,
)

CHUNKS = [
    {"doc_id": "NX-HR-001", "title": "Leave Policy", "page": 3, "content": "Annual leave is 20 days."},
    {"doc_id": "NX-HR-001", "title": "Leave Policy", "page": 4, "content": "Maternity leave is 90 days."},
]


# ---------- extract_citations ----------

def test_extract_keeps_order_and_removes_duplicates():
    assert extract_citations("A [2] then B [1] and again [2].") == [2, 1]


def test_extract_without_citations():
    assert extract_citations("No sources here.") == []


# ---------- validate_citations ----------

def test_valid_citations_are_kept():
    result = validate_citations("Maternity leave is 90 days [2].", CHUNKS)
    assert result["used"] == [2]
    assert result["valid"] == [2]
    assert result["invalid"] == []
    assert result["clean_answer"] == "Maternity leave is 90 days [2]."


def test_hallucinated_citation_is_stripped():
    result = validate_citations("Leave is 90 days [2]. See also [5].", CHUNKS)
    assert result["valid"] == [2]
    assert result["invalid"] == [5]
    assert "[5]" not in result["clean_answer"]
    assert "[2]" in result["clean_answer"]


def test_citation_zero_and_negative_index_are_invalid():
    result = validate_citations("Wrong [0].", CHUNKS)
    assert result["valid"] == []
    assert result["invalid"] == [0]


def test_no_chunks_makes_every_citation_invalid():
    result = validate_citations("Answer [1].", [])
    assert result["valid"] == []
    assert result["invalid"] == [1]


def test_stray_fullwidth_markers_are_removed():
    result = validate_citations("Answer\u3010123\u2020L1-L4\u3011 text.", CHUNKS)
    assert "\u3010" not in result["clean_answer"]
    assert result["valid"] == []


# ---------- build_sources ----------

def test_sources_follow_citation_order_and_numbering():
    sources = build_sources(CHUNKS, [2, 1])
    assert [s["id"] for s in sources] == [2, 1]
    assert sources[0]["page"] == 4          # citation [2] -> chunks[1]
    assert sources[0]["doc_id"] == "NX-HR-001"
    assert sources[1]["title"] == "Leave Policy"


def test_snippet_is_cut_to_200_characters():
    long_chunk = [{"doc_id": "X", "title": "T", "page": 1, "content": "a" * 500}]
    assert len(build_sources(long_chunk, [1])[0]["snippet"]) == 200


def test_no_valid_citations_means_no_sources():
    assert build_sources(CHUNKS, []) == []
