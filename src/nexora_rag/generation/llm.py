"""
LLM client using LangChain (Step K).

Uses LangChain's ChatGroq for chat completions. Switching model is a
.env change (GROQ_MODEL).
"""

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from pydantic import SecretStr

from ..core.config import settings
from ..core.exceptions import LLMError
from ..core.logging import get_logger

log = get_logger(__name__)


def _as_text(content: str | list[str | dict[Any, Any]]) -> str:
    """Chat models return a plain string, or a list of text parts for some models."""
    if isinstance(content, str):
        return content
    return "".join(p if isinstance(p, str) else str(p.get("text", "")) for p in content)


def ask_llm(
    system_prompt: str,
    user_message: str,
    temperature: float | None = None,
    max_tokens: int = 1024,
) -> str:
    """Sends one message to the LLM (via LangChain/Groq) and returns the
    answer text. Retries once if the first call fails."""

    model = ChatGroq(
        api_key=SecretStr(settings.groq_api_key) if settings.groq_api_key else None,
        model=settings.groq_model,
        max_tokens=max_tokens,
        max_retries=2,
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message),
    ]

    for attempt in (1, 2):
        try:
            response = model.invoke(messages)
            return _as_text(response.content)
        except Exception as exc:
            log.warning("llm_call_failed", extra={"attempt": attempt, "error": str(exc)})
            if attempt == 2:
                raise LLMError(f"LLM call failed: {exc}") from exc

    raise LLMError("LLM call failed")  # not reachable: the second failed attempt raises above


if __name__ == "__main__":
    answer = ask_llm(
        system_prompt="You are a helpful assistant. Answer in one short sentence.",
        user_message="What is the capital of France?",
    )
    print(answer)
