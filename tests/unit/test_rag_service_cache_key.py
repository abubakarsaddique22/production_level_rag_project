from src.nexora_rag.core.cache import make_key
from src.nexora_rag.generation import rag_service
from src.nexora_rag.generation.rag_service import RagService


class BoomRetriever:
    def search(self, *args, **kwargs):
        raise AssertionError("retriever must not be called")


def test_cache_key_uses_standalone_question(monkeypatch):
    seen = {}

    def fake_rewrite(question, history):
        return "What is the leave policy for contractors?"

    def fake_get(key):
        seen["key"] = key
        return {"answer": "cached", "sources": [{"id": 1}]}

    monkeypatch.setattr(rag_service, "rewrite_query", fake_rewrite)
    monkeypatch.setattr(rag_service, "get_cached_answer", fake_get)

    service = RagService(retriever=BoomRetriever())
    history = [{"role": "user", "content": "What is the leave policy?"}]
    service.answer("and for contractors?", departments=["HR"], history=history)

    expected = make_key("What is the leave policy for contractors?", ["HR"])
    assert seen["key"] == expected