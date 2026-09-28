"""
Pydantic request/response models for the API (Step P).
"""

from pydantic import BaseModel
from pydantic import BaseModel, EmailStr



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