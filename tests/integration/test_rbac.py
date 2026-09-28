"""
RBAC tests (Step Q): no role may ever retrieve chunks from a
department it is not allowed to see.

Uses the real retriever (Qdrant + BM25), but no login, Postgres or LLM.
Run from the project root:
    pytest tests/integration/test_rbac.py -v
"""

import pytest
import yaml

from nexora_rag.api.deps import ROLE_DEPARTMENTS
from nexora_rag.core.config import settings
from nexora_rag.retrieval.hybrid import HybridRetriever

QUESTIONS = [
    "How many days of maternity leave are paid?",
    "What are the procurement approval tiers?",
    "How do I roll back a failed deployment?",
    "What is the hotel cap per night for a G5 employee?",
    "What are the API error codes?",
]


@pytest.fixture(scope="module")
def retriever():
    return HybridRetriever()


@pytest.mark.parametrize("role", list(ROLE_DEPARTMENTS))
@pytest.mark.parametrize("question", QUESTIONS)
def test_no_chunks_outside_allowed_departments(retriever, role, question):
    allowed = ROLE_DEPARTMENTS[role]
    results = retriever.search(question, top_k=5, departments=allowed)
    for r in results:
        assert r["department"] in allowed


def test_employee_cannot_retrieve_finance_or_engineering(retriever):
    results = retriever.search(
        "What are the procurement approval tiers?",
        top_k=5,
        departments=ROLE_DEPARTMENTS["employee"],
    )
    assert all(r["department"] not in {"Finance", "Engineering"} for r in results)


def test_empty_departments_returns_nothing(retriever):
    assert retriever.search("maternity leave", top_k=5, departments=[]) == []


def test_role_departments_match_registry():
    with open(settings.document_registry_path, encoding="utf-8") as f:
        matrix = yaml.safe_load(f)["access_matrix"]

    expected: dict[str, set[str]] = {}
    for department, roles in matrix.items():
        for role in roles:
            expected.setdefault(role, set()).add(department)

    actual = {role: set(depts) for role, depts in ROLE_DEPARTMENTS.items()}
    assert actual == expected