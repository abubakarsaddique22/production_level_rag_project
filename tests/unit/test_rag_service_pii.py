from src.nexora_rag.generation import rag_service
from src.nexora_rag.generation.rag_service import RagService


class FakeRetriever:
    def search(self, query, top_k, departments):
        return [{"title": "HR", "doc_id": "hr", "page": 1, "content": "text"}]


def test_answer_and_snippets_are_masked_before_caching(monkeypatch):
    saved = {}

    monkeypatch.setattr(rag_service, "get_cached_answer", lambda key: None)
    monkeypatch.setattr(
        rag_service, "set_cached_answer",
        lambda key, answer, sources: saved.update(answer=answer, sources=sources),
    )
    monkeypatch.setattr(rag_service, "ask_llm", lambda system, user: "raw")
    monkeypatch.setattr(
        rag_service, "validate_citations",
        lambda raw, chunks: {"valid": [1], "clean_answer": "Email hr@nexora.com [1]."},
    )
    monkeypatch.setattr(
        rag_service, "build_sources",
        lambda chunks, valid: [{"id": 1, "snippet": "Call 0300-1234567"}],
    )

    result = RagService(retriever=FakeRetriever()).answer(
        "Who do I contact?", departments=["HR"]
    )

    assert "hr@nexora.com" not in result["answer"]
    assert "[1]" in result["answer"]
    assert "0300-1234567" not in result["sources"][0]["snippet"]
    assert "hr@nexora.com" not in saved["answer"]