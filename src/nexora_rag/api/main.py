"""
FastAPI app factory (Step P + Step Q: auth, RBAC + Step R: rate limiting).

Run with:
    uvicorn nexora_rag.api.main:app --reload --app-dir src
Then open http://localhost:8000/docs for the interactive Swagger UI.
"""

from fastapi import FastAPI
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from nexora_rag.core.rate_limit import limiter
from nexora_rag.api.middleware import RequestIDMiddleware
from .routers import auth, chat, health, sessions

app = FastAPI(
    title="Nexora Knowledge Assistant",
    description="Internal RAG assistant over Nexora's HR, Engineering, Finance and Product PDFs.",
    version="0.1.0",
)

# --- Step R: rate limiting ---
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# --- Step C/P: request-id + timing middleware ---
app.add_middleware(RequestIDMiddleware)

# --- routers ---
app.include_router(health.router)
app.include_router(auth.router)   # Step Q: /v1/auth/login
app.include_router(chat.router)
app.include_router(sessions.router)