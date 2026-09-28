from nexora_rag.retrieval import rewrite
from nexora_rag.retrieval.rewrite import rewrite_query

HISTORY = [
    {"role": "user", "content": "What is the hotel cap for a G5 employee?"},
    {"role": "assistant", "content": "PKR 30,000 per night [1]."},
]


def test_no_history_returns_question_without_calling_llm(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("LLM must not be called")

    monkeypatch.setattr(rewrite, "ask_llm", fail)
    assert rewrite_query("and for G4?", []) == "and for G4?"


def test_follow_up_is_rewritten_using_history(monkeypatch):
    seen = {}

    def fake_ask_llm(system_prompt, user_message, **kwargs):
        seen["user_message"] = user_message
        return '"What is the hotel cap for a G4 employee?"'

    monkeypatch.setattr(rewrite, "ask_llm", fake_ask_llm)
    result = rewrite_query("and for G4?", HISTORY)

    assert result == "What is the hotel cap for a G4 employee?"
    assert "G5 employee" in seen["user_message"]
    assert "and for G4?" in seen["user_message"]


def test_llm_failure_falls_back_to_original_question(monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("rate limited")

    monkeypatch.setattr(rewrite, "ask_llm", broken)
    assert rewrite_query("and for G4?", HISTORY) == "and for G4?"


def test_empty_llm_answer_falls_back_to_original_question(monkeypatch):
    monkeypatch.setattr(rewrite, "ask_llm", lambda *args, **kwargs: "   ")
    assert rewrite_query("and for G4?", HISTORY) == "and for G4?"