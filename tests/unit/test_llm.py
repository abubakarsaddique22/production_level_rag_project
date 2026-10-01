import pytest

from nexora_rag.core.exceptions import LLMError
from nexora_rag.generation import llm


class FakeResponse:
    def __init__(self, content):
        self.content = content


def make_fake_chat(results):
    """results: one item per call. An Exception is raised, anything else is returned."""
    calls = []

    class FakeChat:
        def __init__(self, **kwargs):
            pass

        def invoke(self, messages):
            calls.append(messages)
            result = results.pop(0)
            if isinstance(result, Exception):
                raise result
            return FakeResponse(result)

    FakeChat.calls = calls
    return FakeChat


def test_returns_the_answer_text(monkeypatch):
    chat = make_fake_chat(["Hello"])
    monkeypatch.setattr(llm, "ChatGroq", chat)
    assert llm.ask_llm("system", "user") == "Hello"
    assert len(chat.calls) == 1


def test_system_and_user_messages_are_sent(monkeypatch):
    chat = make_fake_chat(["ok"])
    monkeypatch.setattr(llm, "ChatGroq", chat)
    llm.ask_llm("be brief", "what is 2+2?")
    system, human = chat.calls[0]
    assert system.content == "be brief"
    assert human.content == "what is 2+2?"


def test_retries_once_then_succeeds(monkeypatch):
    chat = make_fake_chat([RuntimeError("429"), "second try"])
    monkeypatch.setattr(llm, "ChatGroq", chat)
    assert llm.ask_llm("s", "u") == "second try"
    assert len(chat.calls) == 2


def test_two_failures_raise_llm_error(monkeypatch):
    chat = make_fake_chat([RuntimeError("429"), RuntimeError("429")])
    monkeypatch.setattr(llm, "ChatGroq", chat)
    with pytest.raises(LLMError):
        llm.ask_llm("s", "u")
    assert len(chat.calls) == 2