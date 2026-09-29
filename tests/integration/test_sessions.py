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
    def answer(self, question, departments, user_id=None,history=None):
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


async def test_owner_can_read_session_history(client, make_user):
    headers = await make_user()
    session_id = (await ask(client, headers, "hello")).json()["session_id"]
    await ask(client, headers, "and again", session_id)

    response = await client.get(f"/v1/sessions/{session_id}", headers=headers)

    assert response.status_code == 200
    messages = response.json()["messages"]
    assert [m["role"] for m in messages] == ["user", "assistant", "user", "assistant"]
    assert messages[0]["content"] == "hello"
    assert messages[1]["sources"][0]["doc_id"] == "NX-TEST-001"


async def test_other_user_cannot_read_session_history(client, make_user):
    alice = await make_user()
    bob = await make_user()
    session_id = (await ask(client, alice, "alice question")).json()["session_id"]

    response = await client.get(f"/v1/sessions/{session_id}", headers=bob)

    assert response.status_code == 404


async def test_session_history_requires_token(client):
    response = await client.get(f"/v1/sessions/{uuid.uuid4()}")
    assert response.status_code in (401, 403)


def _recording_service(calls: list):
    """FakeRagService jo har call ka question aur history save karta hai."""

    class RecordingRagService(FakeRagService):
        def answer(self, question, departments, user_id=None, history=None):
            calls.append({"question": question, "history": history})
            return super().answer(question, departments, user_id, history)

    return RecordingRagService


async def test_history_is_passed_to_rag_service(client, make_user):
    calls: list = []
    app.dependency_overrides[get_rag_service] = lambda: _recording_service(calls)()
    headers = await make_user()

    first = await ask(client, headers, "first question")
    session_id = first.json()["session_id"]
    await ask(client, headers, "second question", session_id)

    # Pehle sawal par history khali
    assert calls[0]["history"] == []
    # Doosre par pehla sawal + jawab, oldest first, current sawal shamil nahi
    assert calls[1]["history"] == [
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "answer to: first question"},
    ]
    assert "second question" not in [m["content"] for m in calls[1]["history"]]


async def test_history_is_limited_to_last_6_messages(client, make_user):
    calls: list = []
    app.dependency_overrides[get_rag_service] = lambda: _recording_service(calls)()
    headers = await make_user()

    session_id = (await ask(client, headers, "q1")).json()["session_id"]
    for q in ["q2", "q3", "q4", "q5"]:
        await ask(client, headers, q, session_id)

    history = calls[4]["history"]  # q5 ka call: pehle 8 messages (q1..q4) maujood the
    assert len(history) == 6
    assert history[0] == {"role": "user", "content": "q2"}  # q1 aur uska jawab bahar
    assert history[-1]["content"] == "answer to: q4"


# -- feedback 

async def load_feedback(trace_id):
    async with async_session_factory() as db:
        result = await db.execute(select(Feedback).where(Feedback.trace_id == trace_id))
        return result.scalars().all()


async def give_feedback(client, headers, trace_id, rating, comment=None):
    body = {"trace_id": trace_id, "rating": rating}
    if comment is not None:
        body["comment"] = comment
    return await client.post("/v1/feedback", json=body, headers=headers)


async def test_feedback_is_saved(client, make_user):
    headers = await make_user()
    trace_id = (await ask(client, headers, "hello")).json()["trace_id"]

    response = await give_feedback(client, headers, trace_id, 1, "helpful")

    assert response.status_code == 200
    rows = await load_feedback(trace_id)
    assert len(rows) == 1
    assert rows[0].rating == 1
    assert rows[0].comment == "helpful"


async def test_feedback_twice_updates_the_same_row(client, make_user):
    headers = await make_user()
    trace_id = (await ask(client, headers, "hello")).json()["trace_id"]

    await give_feedback(client, headers, trace_id, 1, "helpful")
    response = await give_feedback(client, headers, trace_id, -1, "wrong answer")

    assert response.status_code == 200
    rows = await load_feedback(trace_id)
    assert len(rows) == 1  # naya row nahi banna chahiye
    assert rows[0].rating == -1
    assert rows[0].comment == "wrong answer"


async def test_feedback_unknown_trace_id_returns_404(client, make_user):
    headers = await make_user()
    response = await give_feedback(client, headers, "trace-does-not-exist", 1)
    assert response.status_code == 404


async def test_user_cannot_give_feedback_on_another_users_answer(client, make_user):
    alice = await make_user()
    bob = await make_user()
    trace_id = (await ask(client, alice, "alice question")).json()["trace_id"]

    response = await give_feedback(client, bob, trace_id, -1)

    assert response.status_code == 404
    assert await load_feedback(trace_id) == []  # bob ka feedback save nahi hua


async def test_feedback_rejects_invalid_rating(client, make_user):
    headers = await make_user()
    trace_id = (await ask(client, headers, "hello")).json()["trace_id"]

    response = await give_feedback(client, headers, trace_id, 5)

    assert response.status_code == 422
    assert await load_feedback(trace_id) == []


async def test_feedback_requires_token(client):
    response = await client.post(
        "/v1/feedback", json={"trace_id": "x", "rating": 1}
    )
    assert response.status_code in (401, 403)