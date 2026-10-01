from types import SimpleNamespace

import pytest

from nexora_rag.agents.tools import calculator, kb_search, lookup_ticket


# ---------- calculator ----------

def test_calculator_basic():
    assert calculator("30000 * 5").result == 150000.0


def test_calculator_commas_and_brackets():
    assert calculator("(2,000 + 3,000) * 4").result == 20000.0


def test_calculator_negative_number():
    assert calculator("-5 + 10").result == 5.0


def test_calculator_rejects_power():
    with pytest.raises(ValueError):
        calculator("9 ** 9 ** 9")


def test_calculator_rejects_code():
    with pytest.raises(ValueError):
        calculator("__import__('os').system('dir')")


def test_calculator_divide_by_zero():
    with pytest.raises(ZeroDivisionError):
        calculator("1 / 0")


def test_calculator_rejects_empty_and_too_long():
    with pytest.raises(ValueError):
        calculator("")
    with pytest.raises(ValueError):
        calculator("1+" * 100 + "1")


# ---------- ticket lookup ----------

def test_ticket_found_ignores_case_and_spaces():
    out = lookup_ticket("  tck-101 ")
    assert out.found is True
    assert out.status == "open"


def test_ticket_not_found():
    out = lookup_ticket("TCK-999")
    assert out.found is False
    assert out.status is None


# ---------- kb_search (fake service) ----------

def make_service(result):
    calls = []

    def answer(question, departments, user_id=None, history=None):
        calls.append({"question": question, "departments": departments, "user_id": user_id})
        return result

    return SimpleNamespace(answer=answer, calls=calls)


def test_kb_search_found_and_passes_departments():
    service = make_service({
        "answer": "PKR 30,000 per night [1].",
        "sources": [{"id": 1, "doc_id": "NX-FIN-001"}],
        "trace_id": "t1",
        "latency_ms": 10,
    })
    out = kb_search(service, "hotel cap for G5?", departments=["Finance"], user_id="u1")
    assert out.found is True
    assert out.sources[0]["doc_id"] == "NX-FIN-001"
    assert service.calls[0]["departments"] == ["Finance"]
    assert service.calls[0]["user_id"] == "u1"


def test_kb_search_no_sources_means_not_found():
    service = make_service({
        "answer": "I don't have enough information to answer that.",
        "sources": [],
        "trace_id": "t2",
        "latency_ms": 5,
    })
    out = kb_search(service, "CEO salary?", departments=["HR"])
    assert out.found is False