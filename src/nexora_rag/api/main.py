"""
FastAPI app factory (Step P + Step Q: auth, RBAC).

Run with:
    uvicorn nexora_rag.api.main:app --reload --app-dir src
Then open http://localhost:8000/docs for the interactive Swagger UI.
"""

from fastapi import FastAPI

from nexora_rag.api.middleware import RequestIDMiddleware
from .routers import auth, chat, health

app = FastAPI(
    title="Nexora Knowledge Assistant",
    description="Internal RAG assistant over Nexora's HR, Engineering, Finance and Product PDFs.",
    version="0.1.0",
)

# --- Step C/P: request-id + timing middleware ---
app.add_middleware(RequestIDMiddleware)

# --- routers ---
app.include_router(health.router)
app.include_router(auth.router)   # Step Q: /v1/auth/login
app.include_router(chat.router)