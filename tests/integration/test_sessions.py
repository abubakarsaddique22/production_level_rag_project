"""
Chat session tests (Step S): messages are stored per session and one user
can never use another user's session.

Uses a fake RagService (no LLM, no models) with the real Postgres and Redis:
    docker compose up -d postgres redis

Run from the project root:
    pytest tests/integration/test_sessions.py -v
"""

import uuid

import httpx
import pytest
from sqlalchemy import delete, select

from nexora_rag.api.deps import get_rag_service
from nexora_rag.api.main import app
from nexora_rag.core.security import create_access_token
from nexora_rag.db.models import ChatSession, Feedback, Message, User
from nexora_rag.db.session import async_session_factory, engine

pytestmark = pytest.mark.anyio


class FakeRagService:
    def answer(self, question, departments, user_id=None):
        return {
            "answer": f"answer to: {question}",
            "sources": [
                {"id": 1, "doc_id": "NX-TEST-001", "title": "Test", "page": 1, "snippet": "s"}
            ],
            "trace_id": f"trace-{uuid.uuid4()}",
            "latency_ms": 1,
        }


@pytest.fixture
async def client():
    app.dependency_overrides[get_rag_service] = lambda: FakeRagService()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture
async def make_user():
    user_ids: list[str] = []

    async def _make(role: str = "employee") -> dict:
        user = User(
            email=f"test-{uuid.uuid4()}@nexora.test",
            hashed_password="not-a-real-hash",
            role=role,
        )
        async with async_session_factory() as db:
            db.add(user)
            await db.commit()
        user_ids.append(user.id)
        token = create_access_token(user_id=user.id, role=role)
        return {"Authorization": f"Bearer {token}"}

    yield _make

    # cleanup: remove everything these test users created
    async with async_session_factory() as db:
        for user_id in user_ids:
            own_sessions = select(ChatSession.id).where(ChatSession.user_id == user_id)
            await db.execute(delete(Message).where(Message.session_id.in_(own_sessions)))
            await db.execute(delete(ChatSession).where(ChatSession.user_id == user_id))
            await db.execute(delete(Feedback).where(Feedback.user_id == user_id))
            await db.execute(delete(User).where(User.id == user_id))
        await db.commit()
    await engine.dispose()  # pooled connections belong to this test's event loop


async def ask(client, headers, question, session_id=None):
    body = {"question": question}
    if session_id:
        body["session_id"] = session_id
    return await client.post("/v1/chat", json=body, headers=headers)


async def load_messages(session_id):
    async with async_session_factory() as db:
        result = await db.execute(
            select(Message).where(Message.session_id == session_id).order_by(Message.id)
        )
        return result.scalars().all()


async def test_new_chat_creates_session_and_saves_messages(client, make_user):
    headers = await make_user()

    response = await ask(client, headers, "hello")
    assert response.status_code == 200
    body = response.json()

    messages = await load_messages(body["session_id"])
    assert [m.role for m in messages] == ["user", "assistant"]
    assert messages[0].content == "hello"
    assert messages[1].content == body["answer"]
    assert messages[1].trace_id == body["trace_id"]
    assert messages[1].sources[0]["doc_id"] == "NX-TEST-001"


async def test_follow_up_stays_in_the_same_session(client, make_user):
    headers = await make_user()
    first = await ask(client, headers, "first question")
    session_id = first.json()["session_id"]

    second = await ask(client, headers, "second question", session_id)

    assert second.status_code == 200
    assert second.json()["session_id"] == session_id
    messages = await load_messages(session_id)
    assert len(messages) == 4
    assert [m.content for m in messages if m.role == "user"] == [
        "first question",
        "second question",
    ]


async def test_unknown_session_returns_404(client, make_user):
    headers = await make_user()
    response = await ask(client, headers, "hello", str(uuid.uuid4()))
    assert response.status_code == 404


async def test_user_cannot_use_another_users_session(client, make_user):
    alice = await make_user()
    bob = await make_user()

    session_id = (await ask(client, alice, "alice question")).json()["session_id"]
    response = await ask(client, bob, "bob question", session_id)

    assert response.status_code == 404
    messages = await load_messages(session_id)
    assert len(messages) == 2  # bob's question was not stored in alice's session