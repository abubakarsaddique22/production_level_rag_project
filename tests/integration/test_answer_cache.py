"""
Answer cache tests (Step R).

Uses a fake retriever and a fake LLM (no models, no Groq calls), but the
real Redis cache. Redis must be running:  docker compose up -d redis

Run from the project root:
    pytest tests/integration/test_answer_cache.py -v
"""

import uuid

import pytest

from nexora_rag.core.cache import make_key
from nexora_rag.generation import rag_service
from nexora_rag.generation.rag_service import RagService

FINANCE = ["HR", "Finance", "Product"]
EMPLOYEE = ["HR", "Product"]


class FakeRetriever:
    def __init__(self, chunks):
        self.chunks = chunks
        self.calls = []  # the departments of every search

    def search(self, query, top_k=5, departments=None):
        self.calls.append(departments)
        return self.chunks


def make_chunk():
    return {
        "chunk_id": "NX-TEST-001-p1-c0",
        "content": "Maternity leave is 90 calendar days.",
        "doc_id": "NX-TEST-001",
        "title": "Test Policy",
        "department": "HR",
        "version": "1.0",
        "effective_date": "2026-01-01",
        "page": 1,
        "chunk_type": "prose",
        "chunk_index": 0,
    }


def new_question():
    # unique text, so an old cached answer can never interfere
    return f"test question {uuid.uuid4()}"


@pytest.fixture
def llm_calls(monkeypatch):
    calls = []

    def fake_ask_llm(system_prompt, user_message, **kwargs):
        calls.append(user_message)
        return "The answer is 90 days [1]."

    monkeypatch.setattr(rag_service, "ask_llm", fake_ask_llm)
    return calls


def test_same_question_and_role_hits_cache(llm_calls):
    retriever = FakeRetriever([make_chunk()])
    service = RagService(retriever=retriever, top_k=3)
    question = new_question()

    first = service.answer(question, departments=EMPLOYEE)
    second = service.answer(question, departments=EMPLOYEE)

    assert first["sources"], "test setup: the fake answer must have sources"
    assert second["answer"] == first["answer"]
    assert second["sources"] == first["sources"]
    assert len(llm_calls) == 1
    assert len(retriever.calls) == 1


def test_cache_is_not_shared_between_roles(llm_calls):
    retriever = FakeRetriever([make_chunk()])
    service = RagService(retriever=retriever, top_k=3)
    question = new_question()

    service.answer(question, departments=FINANCE)
    service.answer(question, departments=EMPLOYEE)

    assert len(llm_calls) == 2
    assert retriever.calls == [FINANCE, EMPLOYEE]


def test_refusals_are_not_cached(llm_calls):
    retriever = FakeRetriever([])  # nothing retrieved -> refusal
    service = RagService(retriever=retriever, top_k=3)
    question = new_question()

    first = service.answer(question, departments=EMPLOYEE)
    service.answer(question, departments=EMPLOYEE)

    assert "don't have enough information" in first["answer"]
    assert len(retriever.calls) == 2


def test_key_ignores_case_spaces_and_department_order():
    assert make_key("Hello  World", ["HR", "Product"]) == make_key(
        " hello world ", ["Product", "HR"]
    )


def test_key_differs_for_different_departments():
    assert make_key("hello", ["HR"]) != make_key("hello", ["HR", "Finance"])