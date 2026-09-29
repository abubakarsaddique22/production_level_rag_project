from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.concurrency import run_in_threadpool
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ...core.rate_limit import limiter
from ...db.models import ChatSession, Message
from ...db.session import get_db
from ...generation.rag_service import RagService
from ..deps import get_rag_service, current_user, CurrentUser
from ..schemas import ChatRequest, ChatResponse

router = APIRouter(prefix="/v1", tags=["chat"])

HISTORY_LIMIT = 6


async def get_or_create_session(
    db: AsyncSession, session_id: str | None, user_id: str
) -> ChatSession:
    if session_id is None:
        session = ChatSession(user_id=user_id)
        db.add(session)
        await db.flush()  # assigns the session id
        return session

    # Only the owner's sessions are visible. Someone else's id looks the same
    # as an id that does not exist (404), so ids cannot be probed.
    result = await db.execute(
        select(ChatSession).where(
            ChatSession.id == session_id, ChatSession.user_id == user_id
        )
    )
    session = result.scalar_one_or_none()
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Session not found"
        )
    return session


async def load_history(db: AsyncSession, session_id: str) -> list[dict]:
    """Last HISTORY_LIMIT messages of the session, oldest first."""
    result = await db.execute(
        select(Message)
        .where(Message.session_id == session_id)
        .order_by(Message.id.desc())
        .limit(HISTORY_LIMIT)
    )
    messages = result.scalars().all()
    messages.reverse()
    return [{"role": m.role, "content": m.content} for m in messages]


@router.post("/chat", response_model=ChatResponse)
@limiter.limit("20/minute")
async def chat(
    request: Request,
    req: ChatRequest,
    user: CurrentUser = Depends(current_user),
    rag_service: RagService = Depends(get_rag_service),
    db: AsyncSession = Depends(get_db),
) -> ChatResponse:
    session = await get_or_create_session(db, req.session_id, user.id)

    # History is loaded BEFORE saving the new question, so the current
    # question is not in it.
    history = await load_history(db, session.id)

    # RBAC: mandatory department filter, enforced here, never only in the prompt.
    # rag_service is blocking, so it runs in a thread and does not freeze the API.
    result = await run_in_threadpool(
        rag_service.answer,
        question=req.question,
        departments=user.departments,
        user_id=user.id,
        history=history,
    )

    db.add(Message(session_id=session.id, role="user", content=req.question))
    db.add(
        Message(
            session_id=session.id,
            role="assistant",
            content=result["answer"],
            sources=result["sources"],
            trace_id=result["trace_id"],
        )
    )
    session.updated_at = datetime.now(timezone.utc)
    await db.commit()

    return ChatResponse(**result, session_id=session.id)