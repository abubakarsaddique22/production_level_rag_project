"""LangSmith tracing helpers (Step W)."""
import os
from dotenv import load_dotenv

load_dotenv()  # LANGSMITH_* os.environ mein chahiye, SDK yahin se padhta hai

from langsmith import Client, traceable  # noqa: E402
from langsmith.run_helpers import get_current_run_tree  # noqa: E402

__all__ = ["traceable", "tracing_enabled", "current_trace_id", "send_feedback"]


def tracing_enabled() -> bool:
    return (os.getenv("LANGSMITH_TRACING", "").lower() == "true"
            and bool(os.getenv("LANGSMITH_API_KEY")))


def current_trace_id() -> str | None:
    """Chal rahe trace ka id. Yehi trace_id API ko wapas jayega."""
    run = get_current_run_tree()
    return str(run.trace_id) if run else None


def send_feedback(trace_id: str, score: float, comment: str | None = None) -> None:
    """User ka thumbs up/down LangSmith trace se jodta hai. Fail ho to request na toote."""
    if not tracing_enabled():
        return
    try:
        Client().create_feedback(run_id=trace_id, key="user_rating",
                                 score=score, comment=comment)
    except Exception:
        pass