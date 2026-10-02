from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from ..core.rate_limit import limiter
from ..observability.metrics import metrics_app
from .middleware import RequestIDMiddleware
from .routers import agent as agent_router
from .routers import auth, chat, feedback, health, sessions

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
app.include_router(feedback.router)  # Step S: /v1/feedback
app.include_router(agent_router.router)
app.mount("/metrics", metrics_app)

# --- Web UI (plain HTML/CSS/JS in frontend/) ---
# Same origin as the API, so no CORS setup is needed.
# Local: <repo>/src/nexora_rag/api/main.py -> parents[3] = <repo>. Docker: /app.
FRONTEND_DIR = Path(__file__).resolve().parents[3] / "frontend"

if FRONTEND_DIR.is_dir():
    app.mount("/ui", StaticFiles(directory=FRONTEND_DIR, html=True), name="ui")

    @app.get("/", include_in_schema=False)
    async def root() -> RedirectResponse:
        return RedirectResponse(url="/ui/")
