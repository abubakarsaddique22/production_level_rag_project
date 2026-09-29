import uuid

import pytest
from fastapi.testclient import TestClient

from nexora_rag.api.deps import get_rag_service
from nexora_rag.api.main import app
from nexora_rag.core.security import create_access_token
from nexora_rag.db.session import get_db


class FakeRagService:
    def answer(self, question, departments, user_id=None, history=None):
        return {"answer": "ok", "sources": [], "trace_id": "t", "latency_ms": 1}


class _FakeResult:
    """Empty query result: load_history() gets no messages."""

    def scalars(self):
        return self

    def all(self):
        return []


class FakeDB:
    """Stands in for the database session (no Postgres needed in this test)."""

    def add(self, obj):
        if getattr(obj, "id", None) is None:
            obj.id = str(uuid.uuid4())

    async def flush(self):
        pass

    async def commit(self):
        pass

    async def execute(self, *args, **kwargs):
        return _FakeResult()


async def fake_get_db():
    yield FakeDB()


@pytest.fixture
def client():
    app.dependency_overrides[get_rag_service] = lambda: FakeRagService()
    app.dependency_overrides[get_db] = fake_get_db
    yield TestClient(app)
    app.dependency_overrides.clear()


def new_user_headers(role="employee"):
    token = create_access_token(user_id=str(uuid.uuid4()), role=role)
    return {"Authorization": f"Bearer {token}"}


def ask(client, headers):
    return client.post("/v1/chat", json={"question": "test"}, headers=headers)


def test_blocked_after_20_requests(client):
    headers = new_user_headers()
    codes = [ask(client, headers).status_code for _ in range(21)]
    assert codes[:20] == [200] * 20
    assert codes[20] == 429


def test_limit_is_per_user(client):
    user_a, user_b = new_user_headers(), new_user_headers()
    for _ in range(21):
        ask(client, user_a)
    assert ask(client, user_b).status_code == 200


def test_chat_requires_token(client):
    response = client.post("/v1/chat", json={"question": "test"})
    assert response.status_code in (401, 403)