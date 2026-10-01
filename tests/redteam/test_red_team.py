import json
from pathlib import Path

import pytest

from nexora_rag.generation import rag_service
from nexora_rag.generation.prompts import build_user_message
from nexora_rag.generation.rag_service import RagService
from nexora_rag.guardrails.input_checks import check_input
from nexora_rag.guardrails.output_checks import OUTPUT_REFUSAL

DATA = json.loads((Path(__file__).parent / "attacks.json").read_text(encoding="utf-8"))


def attack_id(attack):
    return attack["id"]


class FakeRetriever:
    def __init__(self, chunks=None):
        default = [{"title": "HR", "doc_id": "hr", "page": 1, "content": "text"}]
        self.chunks = default if chunks is None else chunks
        self.seen_departments = None

    def search(self, query, top_k, departments):
        self.seen_departments = departments
        return self.chunks


def run_service(monkeypatch, retriever, llm_output="", question="Who do I contact?", departments=None):
    """RagService ko fake LLM/cache ke saath chalata hai. saved = jo cache mein gaya."""
    saved = {}
    monkeypatch.setattr(rag_service, "get_cached_answer", lambda key: None)
    monkeypatch.setattr(
        rag_service, "set_cached_answer",
        lambda key, answer, sources: saved.update(answer=answer),
    )
    monkeypatch.setattr(rag_service, "ask_llm", lambda system, user: llm_output)
    monkeypatch.setattr(
        rag_service, "validate_citations",
        lambda raw, chunks: {"valid": [1], "clean_answer": raw},
    )
    monkeypatch.setattr(
        rag_service, "build_sources",
        lambda chunks, valid: [{"id": 1, "snippet": "x"}],
    )
    result = RagService(retriever=retriever).answer(
        question, departments=departments or ["HR"]
    )
    return result, saved


@pytest.mark.parametrize("attack", DATA["input_attacks"], ids=attack_id)
def test_input_attacks_are_blocked(attack):
    assert check_input(attack["payload"]) == (False, attack["expected_reason"])


@pytest.mark.parametrize("attack", DATA["document_attacks"], ids=attack_id)
def test_poisoned_document_cannot_break_out(attack):
    chunk = {
        "title": attack.get("title", "Leave Policy"),
        "page": 1,
        "content": attack["content"],
    }
    message = build_user_message("How many days?", [chunk])
    assert message.count("<document ") == 1
    assert message.count("</document>") == 1


@pytest.mark.parametrize("attack", DATA["output_attacks"], ids=attack_id)
def test_bad_model_output_is_stopped(monkeypatch, attack):
    result, saved = run_service(monkeypatch, FakeRetriever(), attack["llm_output"])

    if attack["expected"] == "refused":
        assert result["answer"] == OUTPUT_REFUSAL
        assert result["sources"] == []
        assert saved == {}
    else:
        for secret in attack["must_not_contain"]:
            assert secret not in result["answer"]
            assert secret not in saved["answer"]


@pytest.mark.parametrize("attack", DATA["rbac_attacks"], ids=attack_id)
def test_user_cannot_widen_their_departments(monkeypatch, attack):
    retriever = FakeRetriever(chunks=[])
    result, _ = run_service(
        monkeypatch, retriever, question=attack["question"], departments=["HR"]
    )
    assert retriever.seen_departments == ["HR"]
    assert result["sources"] == []


@pytest.mark.xfail(strict=True, reason="known gap: rule-based guard misses paraphrases and other languages")
@pytest.mark.parametrize("attack", DATA["known_gaps"], ids=attack_id)
def test_known_gaps(attack):
    assert check_input(attack["payload"]) == (False, "jailbreak")