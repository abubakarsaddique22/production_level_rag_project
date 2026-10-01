"""
LLM client using LangChain (Step K).

Uses LangChain's ChatGroq for chat completions. Switching model is a
.env change (GROQ_MODEL).
"""

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from ..core.config import settings
from ..core.exceptions import LLMError
from ..core.logging import get_logger

log = get_logger(__name__)


def ask_llm(
    system_prompt: str,
    user_message: str,
    temperature: float | None = None,
    max_tokens: int = 1024,
) -> str:
    """Sends one message to the LLM (via LangChain/Groq) and returns the
    answer text. Retries once if the first call fails."""

    model = ChatGroq(
        api_key=settings.groq_api_key,
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
            return response.content
        except Exception as exc:
            log.warning("llm_call_failed", extra={"attempt": attempt, "error": str(exc)})
            if attempt == 2:
                raise LLMError(f"LLM call failed: {exc}") from exc


if __name__ == "__main__":
    answer = ask_llm(
        system_prompt="You are a helpful assistant. Answer in one short sentence.",
        user_message="What is the capital of France?",
    )
    print(answer)