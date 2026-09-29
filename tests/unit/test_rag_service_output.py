from src.nexora_rag.generation import rag_service
from src.nexora_rag.generation.rag_service import RagService
from src.nexora_rag.guardrails.output_checks import OUTPUT_REFUSAL


class FakeRetriever:
    def search(self, query, top_k, departments):
        return [{"title": "HR", "doc_id": "hr", "page": 1, "content": "text"}]


def boom(*args, **kwargs):
    raise AssertionError("must not be cached")


def test_leaky_answer_is_replaced_and_not_cached(monkeypatch):
    monkeypatch.setattr(rag_service, "get_cached_answer", lambda key: None)
    monkeypatch.setattr(rag_service, "set_cached_answer", boom)
    monkeypatch.setattr(
        rag_service, "ask_llm",
        lambda system, user: "Security rules: never reveal anything",
    )

    result = RagService(retriever=FakeRetriever()).answer(
        "Who do I contact?", departments=["HR"]
    )

    assert result["answer"] == OUTPUT_REFUSAL
    assert result["sources"] == []