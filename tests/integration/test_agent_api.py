import time
import uuid

import httpx
import pytest
from sqlalchemy import delete, select

from nexora_rag.agents import graph as g
from nexora_rag.api.deps import get_rag_service
from nexora_rag.api.main import app
from nexora_rag.api.routers import agent as agent_router
from nexora_rag.core.security import create_access_token
from nexora_rag.db.models import ChatSession, Feedback, Message, User
from nexora_rag.db.session import async_session_factory, engine

pytestmark = pytest.mark.anyio


class FakeRagService:
    """Har call record karta hai. Tests attributes badal kar iska behaviour badalte hain."""

    def __init__(self):
        self.calls = []
        self.with_sources = True
        self.slow_word = None  # sawal mein ye lafz ho to 1 second so jata hai

    def answer(self, question, departments, user_id=None, history=None):
        self.calls.append({
            "question": question,
            "departments": departments,
            "user_id": user_id,
            "history": history,
        })
        if self.slow_word and self.slow_word in question:
            time.sleep(1)
        sources = (
            [{"id": 1, "doc_id": "NX-TEST-001", "title": "Test", "page": 1, "snippet": "s"}]
            if self.with_sources
            else []
        )
        return {
            "answer": f"answer to: {question}",
            "sources": sources,
            "trace_id": f"trace-{uuid.uuid4()}",
            "latency_ms": 1,
        }


@pytest.fixture
def service():
    return FakeRagService()


@pytest.fixture
def llm_calls(monkeypatch):
    """Agent ka LLM fake: router hamesha 'kb' kehta hai. Baaki kisi call par test fail."""
    calls = []

    def fake(system_prompt, user_message, **kwargs):
        calls.append(system_prompt)
        if system_prompt == g.ROUTER_PROMPT:
            return "kb"
        raise AssertionError("unexpected LLM call")

    monkeypatch.setattr(g, "ask_llm", fake)
    monkeypatch.setattr(g, "mask_pii", lambda text: text)  # Presidio ke bina tests tez
    return calls


@pytest.fixture
async def client(service):
    app.dependency_overrides[get_rag_service] = lambda: service
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

    async with async_session_factory() as db:
        for user_id in user_ids:
            own_sessions = select(ChatSession.id).where(ChatSession.user_id == user_id)
            await db.execute(delete(Message).where(Message.session_id.in_(own_sessions)))
            await db.execute(delete(ChatSession).where(ChatSession.user_id == user_id))
            await db.execute(delete(Feedback).where(Feedback.user_id == user_id))
            await db.execute(delete(User).where(User.id == user_id))
        await db.commit()
    await engine.dispose()


async def ask_agent(client, headers, question, session_id=None):
    body = {"question": question}
    if session_id:
        body["session_id"] = session_id
    return await client.post("/v1/agent/chat", json=body, headers=headers)


async def ask_chat(client, headers, question):
    return await client.post("/v1/chat", json={"question": question}, headers=headers)


async def load_messages(session_id):
    async with async_session_factory() as db:
        result = await db.execute(
            select(Message).where(Message.session_id == session_id).order_by(Message.id)
        )
        return result.scalars().all()


# ---------- auth ----------

async def test_agent_requires_token(client):
    response = await client.post("/v1/agent/chat", json={"question": "hello"})
    assert response.status_code in (401, 403)


# ---------- answer + session ----------

async def test_agent_answers_and_saves_session(client, make_user, llm_calls):
    headers = await make_user()

    response = await ask_agent(client, headers, "What is the leave policy?")

    assert response.status_code == 200
    body = response.json()
    assert body["route"] == "kb"
    assert body["grounded"] is True
    assert body["tool_calls"] == 1
    assert body["answer"] == "answer to: What is the leave policy?"
    assert body["sources"][0]["doc_id"] == "NX-TEST-001"

    messages = await load_messages(body["session_id"])
    assert [m.role for m in messages] == ["user", "assistant"]
    assert messages[1].content == body["answer"]
    assert messages[1].trace_id == body["trace_id"]
    assert messages[1].sources[0]["doc_id"] == "NX-TEST-001"


async def test_feedback_works_on_agent_trace_id(client, make_user, llm_calls):
    headers = await make_user()
    trace_id = (await ask_agent(client, headers, "What is the leave policy?")).json()["trace_id"]

    response = await client.post(
        "/v1/feedback", json={"trace_id": trace_id, "rating": 1}, headers=headers
    )

    assert response.status_code == 200


async def test_agent_ungrounded_when_no_sources(client, make_user, service, llm_calls):
    service.with_sources = False
    headers = await make_user()

    body = (await ask_agent(client, headers, "What is the CEO's salary?")).json()

    assert body["grounded"] is False
    assert body["sources"] == []


async def test_agent_small_talk_uses_no_tools(client, make_user, service, llm_calls):
    headers = await make_user()

    response = await ask_agent(client, headers, "hello")

    body = response.json()
    assert body["route"] == "direct"
    assert body["tool_calls"] == 0
    assert body["sources"] == []
    assert service.calls == []  # KB tak gaya hi nahi
    assert llm_calls == []      # router ka LLM bhi nahi chala
    assert len(await load_messages(body["session_id"])) == 2


# ---------- RBAC + history ----------

async def test_agent_uses_same_departments_as_chat(client, make_user, service, llm_calls):
    headers = await make_user("engineer")

    await ask_chat(client, headers, "What is the deployment checklist?")
    await ask_agent(client, headers, "What is the rollback trigger?")

    chat_call, agent_call = service.calls
    assert agent_call["departments"] == chat_call["departments"]
    assert agent_call["user_id"] == chat_call["user_id"]
    assert agent_call["departments"]  # khali nahi


async def test_agent_passes_history_to_service(client, make_user, service, llm_calls):
    headers = await make_user()
    first = await ask_agent(client, headers, "What is the leave policy?")
    session_id = first.json()["session_id"]

    await ask_agent(client, headers, "and for maternity?", session_id)

    assert service.calls[0]["history"] == []
    assert service.calls[1]["history"] == [
        {"role": "user", "content": "What is the leave policy?"},
        {"role": "assistant", "content": "answer to: What is the leave policy?"},
    ]


# ---------- session isolation ----------

async def test_agent_unknown_session_returns_404(client, make_user, llm_calls):
    headers = await make_user()
    response = await ask_agent(client, headers, "hello", str(uuid.uuid4()))
    assert response.status_code == 404


async def test_agent_cannot_use_another_users_session(client, make_user, llm_calls):
    alice = await make_user()
    bob = await make_user()
    session_id = (await ask_agent(client, alice, "What is the leave policy?")).json()["session_id"]

    response = await ask_agent(client, bob, "What is the hotel cap?", session_id)

    assert response.status_code == 404
    assert len(await load_messages(session_id)) == 2  # bob ka sawal alice ki session mein nahi gaya


# ---------- timeout ----------

async def test_agent_timeout_returns_504_quickly_and_saves_nothing(
    client, make_user, service, llm_calls, monkeypatch
):
    headers = await make_user()
    session_id = (await ask_agent(client, headers, "What is the leave policy?")).json()["session_id"]

    monkeypatch.setattr(agent_router, "AGENT_TIMEOUT_SECONDS", 0.2)
    service.slow_word = "slow"

    start = time.time()
    response = await ask_agent(client, headers, "slow hotel question", session_id)
    elapsed = time.time() - start

    assert response.status_code == 504
    assert elapsed < 0.9  # service ki 1 second ki neend ka intezar nahi kiya
    assert len(await load_messages(session_id)) == 2  # timeout wala sawal save nahi hua