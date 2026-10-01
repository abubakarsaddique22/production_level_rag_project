import asyncio
import time
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from ...agents.graph import agent
from ...core.rate_limit import limiter
from ...db.models import Message
from ...db.session import get_db
from ...generation.rag_service import RagService
from ..deps import CurrentUser, current_user, get_rag_service
from ..schemas import AgentChatResponse, ChatRequest
from .chat import get_or_create_session, load_history

router = APIRouter(prefix="/v1/agent", tags=["agent"])

AGENT_TIMEOUT_SECONDS = 45


@router.post("/chat", response_model=AgentChatResponse)
@limiter.limit("10/minute")  # agent ek sawal par kai LLM calls karta hai, isliye /chat se kam
async def agent_chat(
    request: Request,
    req: ChatRequest,
    user: CurrentUser = Depends(current_user),
    rag_service: RagService = Depends(get_rag_service),
    db: AsyncSession = Depends(get_db),
) -> AgentChatResponse:
    session = await get_or_create_session(db, req.session_id, user.id)
    history = await load_history(db, session.id)  # naya sawal save hone se pehle

    start = time.time()
    try:
        # RBAC: departments user ke role se aate hain, agent ke tools unhein badalte nahi.
        # agent.invoke blocking hai, isliye thread mein chalta hai.
            state = await asyncio.wait_for(
            asyncio.to_thread(
                agent.invoke,
                {
                    "question": req.question,
                    "departments": user.departments,
                    "user_id": user.id,
                    "history": history,
                    "service": rag_service,
                },
            ),
            timeout=AGENT_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail="Agent timed out"
        )

    answer = state["answer"]
    sources = state.get("sources", [])  # ticket aur direct route mein sources nahi hote
    trace_id = str(uuid.uuid4())
    latency_ms = int((time.time() - start) * 1000)

    db.add(Message(session_id=session.id, role="user", content=req.question))
    db.add(
        Message(
            session_id=session.id,
            role="assistant",
            content=answer,
            sources=sources,
            trace_id=trace_id,
        )
    )
    session.updated_at = datetime.now(timezone.utc)
    await db.commit()

    return AgentChatResponse(
        answer=answer,
        sources=sources,
        trace_id=trace_id,
        latency_ms=latency_ms,
        session_id=session.id,
        route=state["route"],
        grounded=state.get("grounded"),
        tool_calls=state.get("tool_calls", 0),
    )