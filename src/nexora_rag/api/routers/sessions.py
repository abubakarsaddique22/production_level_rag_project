"""
Sessions endpoints (Step S).

GET /v1/sessions        -- list the current user's chats, newest first.
GET /v1/sessions/{id}   -- return one conversation's history.
Only the owner can read a session.
"""

from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from ...db.models import ChatSession, Message
from ...db.session import get_db
from ..deps import CurrentUser, current_user
from ..schemas import MessageItem, SessionResponse, SessionSummary
from .chat import get_or_create_session

router = APIRouter(prefix="/v1", tags=["sessions"])

LIST_LIMIT = 50
TITLE_MAX_CHARS = 60


def make_title(text: str) -> str:
    """First question of a chat, on one line, shortened for the sidebar."""
    one_line = " ".join(text.split())
    if len(one_line) <= TITLE_MAX_CHARS:
        return one_line
    return one_line[: TITLE_MAX_CHARS - 1].rstrip() + "\u2026"


@router.get("/sessions", response_model=list[SessionSummary])
async def list_sessions(
    user: CurrentUser = Depends(current_user),
    db: AsyncSession = Depends(get_db),
) -> list[SessionSummary]:
    # Id of the first user message of every session (used as the chat title).
    first_question = (
        select(Message.session_id, func.min(Message.id).label("first_id"))
        .where(Message.role == "user")
        .group_by(Message.session_id)
        .subquery()
    )
    # The user_id filter is the access control: nobody sees another user's chats.
    result = await db.execute(
        select(ChatSession.id, ChatSession.updated_at, Message.content)
        .join(first_question, first_question.c.session_id == ChatSession.id)
        .join(Message, Message.id == first_question.c.first_id)
        .where(ChatSession.user_id == user.id)
        .order_by(ChatSession.updated_at.desc())
        .limit(LIST_LIMIT)
    )
    return [
        SessionSummary(session_id=sid, title=make_title(content), updated_at=updated)
        for sid, updated, content in result.all()
    ]


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
