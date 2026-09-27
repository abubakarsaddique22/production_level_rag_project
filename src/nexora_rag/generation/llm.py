"""
LLM client using LangChain (Step K).

Uses LangChain's ChatOpenAI pointed at OpenRouter (OpenRouter is
OpenAI-API-compatible, so ChatOpenAI works with it by just changing
base_url + api_key). Switching model/provider later is a .env change.
"""

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

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
    """Sends one message to the LLM (via LangChain) and returns the
    answer text. Retries once if the first call fails."""
    temperature = settings.llm_temperature if temperature is None else temperature

    model = ChatOpenAI(
        base_url=settings.openrouter_base_url,
        api_key=settings.openrouter_api_key,
        model=settings.llm_model,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message),
    ]

    for attempt in (1, 2):
        try:
            response = model.invoke(messages)
            return response.content
        except Exception as exc:  # noqa: BLE001
            log.warning("llm_call_failed", extra={"attempt": attempt, "error": str(exc)})
            if attempt == 2:
                raise LLMError(f"LLM call failed: {exc}") from exc


if __name__ == "__main__":
    answer = ask_llm(
        system_prompt="You are a helpful assistant. Answer in one short sentence.",
        user_message="What is the capital of France?",
    )
    print(answer)