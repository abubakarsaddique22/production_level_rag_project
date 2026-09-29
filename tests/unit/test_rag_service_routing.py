from nexora_rag.generation import rag_service
from nexora_rag.generation.rag_service import RagService


class FakeRetriever:
    """Har search call ko record karta hai, aur khali result deta hai."""

    def __init__(self):
        self.calls = 0

    def search(self, question, top_k, departments):
        self.calls += 1
        return []


def _boom(*args, **kwargs):
    raise AssertionError("LLM should not be called")


def test_small_talk_skips_retriever_and_llm(monkeypatch):
    monkeypatch.setattr(rag_service, "ask_llm", _boom)
    monkeypatch.setattr(rag_service, "rewrite_query", _boom)
    retriever = FakeRetriever()
    service = RagService(retriever=retriever)

    response = service.answer("hello!", departments=["HR"])

    assert "Hello" in response["answer"]
    assert response["sources"] == []
    assert response["trace_id"]
    assert retriever.calls == 0


def test_real_question_goes_to_retriever(monkeypatch):
    # Cache (Redis) ke bina chalane ke liye cache miss fake karte hain
    monkeypatch.setattr(rag_service, "get_cached_answer", lambda key: None)
    retriever = FakeRetriever()
    service = RagService(retriever=retriever)

    response = service.answer(
        "How many days of paid maternity leave are there?", departments=["HR"]
    )

    assert retriever.calls == 1
    assert response["sources"] == []
    assert "don't have enough information" in response["answer"]