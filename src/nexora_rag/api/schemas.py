from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str
    session_id: str | None = None


class SourceItem(BaseModel):
    id: int
    doc_id: str | None = None
    title: str | None = None
    page: int | None = None
    snippet: str | None = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceItem]
    trace_id: str
    latency_ms: int
    session_id: str


# ============================================================
# Auth schemas (Step Q)
# ============================================================

class LoginRequest(BaseModel):
    email: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


# ============================================================
# Sessions schemas (Step S)
# ============================================================

class MessageItem(BaseModel):
    role: str
    content: str
    sources: list[SourceItem] = []
    trace_id: str | None = None
    created_at: datetime


class SessionResponse(BaseModel):
    session_id: str
    messages: list[MessageItem]


# ============================================================
# Feedback schemas (Step S)
# ============================================================

class FeedbackRequest(BaseModel):
    trace_id: str
    rating: Literal[1, -1]  # 1 = thumbs up, -1 = thumbs down
    comment: str | None = Field(default=None, max_length=1000)


class FeedbackResponse(BaseModel):
    trace_id: str
    rating: int
    comment: str | None