"""
GET /v1/sessions/{id} -- return a conversation's history (Step S).
Only the owner can read a session.
"""

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ...db.models import Message
from ...db.session import get_db
from ..deps import CurrentUser, current_user
from ..schemas import MessageItem, SessionResponse
from .chat import get_or_create_session

router = APIRouter(prefix="/v1", tags=["sessions"])


@router.get("/sessions/{session_id}", response_model=SessionResponse)
async def get_session(
    session_id: str,
    user: CurrentUser = Depends(current_user),
    db: AsyncSession = Depends(get_db),
) -> SessionResponse:
    # session_id is never None here, so this only looks the session up
    # (404 if it does not exist or belongs to someone else).
    session = await get_or_create_session(db, session_id, user.id)

    result = await db.execute(
        select(Message).where(Message.session_id == session.id).order_by(Message.id)
    )
    messages = [
        MessageItem(
            role=m.role,
            content=m.content,
            sources=m.sources or [],
            trace_id=m.trace_id,
            created_at=m.created_at,
        )
        for m in result.scalars().all()
    ]
    return SessionResponse(session_id=session.id, messages=messages)