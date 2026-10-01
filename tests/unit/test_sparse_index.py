import json

from nexora_rag.retrieval.sparse import SparseIndex

CHUNKS = {
    "doc_hr": [
        {
            "chunk_id": "c-hr",
            "content": "Paid maternity leave is ninety calendar days for every employee.",
            "metadata": {"department": "HR"},
        }
    ],
    "doc_fin": [
        {
            "chunk_id": "c-fin",
            "content": "Procurement approval tiers are defined in NX-FIN-002 for purchases.",
            "metadata": {"department": "Finance"},
        }
    ],
    "doc_eng": [
        {
            "chunk_id": "c-eng",
            "content": "Roll back the canary release when the error rate stays high.",
            "metadata": {"department": "Engineering"},
        }
    ],
    "doc_prod": [
        {
            "chunk_id": "c-prod",
            "content": "The dashboard exports reports as csv or pdf files.",
            "metadata": {"department": "Product"},
        }
    ],
}


def build_index(tmp_path):
    for folder, chunks in CHUNKS.items():
        directory = tmp_path / folder
        directory.mkdir()
        (directory / "chunks.json").write_text(json.dumps(chunks), encoding="utf-8")
    return SparseIndex(processed_dir=tmp_path)


def test_keyword_query_finds_the_matching_chunk(tmp_path):
    index = build_index(tmp_path)
    ranked = index.search("maternity leave days", top_k=3)
    assert ranked[0][0] == "c-hr"


def test_exact_document_id_is_found(tmp_path):
    # the reason hybrid search exists: IDs like NX-FIN-002 are weak for dense search
    index = build_index(tmp_path)
    assert index.search("NX-FIN-002", top_k=3)[0][0] == "c-fin"


def test_department_filter_hides_other_departments(tmp_path):
    index = build_index(tmp_path)
    ranked = index.search("approval tiers purchases", top_k=5, departments=["HR"])
    assert "c-fin" not in [chunk_id for chunk_id, _ in ranked]


def test_empty_department_list_returns_nothing(tmp_path):
    index = build_index(tmp_path)
    assert index.search("maternity leave", top_k=5, departments=[]) == []


def test_top_k_limits_results(tmp_path):
    index = build_index(tmp_path)
    assert len(index.search("leave", top_k=2)) == 2


def test_empty_corpus_returns_no_results(tmp_path):
    index = SparseIndex(processed_dir=tmp_path)
    assert index.search("anything") == []


def test_get_chunk(tmp_path):
    index = build_index(tmp_path)
    assert index.get_chunk("c-hr")["metadata"]["department"] == "HR"
    assert index.get_chunk("missing") is None
