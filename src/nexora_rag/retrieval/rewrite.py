"""
Query rewriting (Step O): turn a follow-up question like "and for G4?" into
a standalone question, using the recent chat history.
"""

from ..core.logging import get_logger
from ..generation.llm import ask_llm

log = get_logger(__name__)

REWRITE_PROMPT = (
    "You rewrite follow-up questions. Given a conversation and the user's last "
    "question, rewrite the last question so it can be understood on its own, "
    "without the conversation. Keep every name, number and grade from the "
    "conversation that the question depends on. If the question is already "
    "standalone, return it unchanged. Return only the rewritten question."
)

MAX_CHARS_PER_MESSAGE = 300
MAX_REWRITE_CHARS = 500


def rewrite_query(question: str, history: list[dict]) -> str:
    """history: [{"role": "user" | "assistant", "content": str}, ...], oldest first.

    Returns a standalone version of the question. If there is no history, or
    the LLM fails, or its answer looks wrong, the original question is returned.
    """
    if not history:
        return question

    conversation = "\n".join(
        f"{m['role']}: {m['content'][:MAX_CHARS_PER_MESSAGE]}" for m in history
    )
    user_message = f"Conversation:\n{conversation}\n\nLast question: {question}"

    try:
        rewritten = ask_llm(REWRITE_PROMPT, user_message, max_tokens=256)
    except Exception as exc:  # noqa: BLE001
        log.warning("query_rewrite_failed", extra={"error": str(exc)})
        return question

    rewritten = rewritten.strip().strip('"')
    if not rewritten or len(rewritten) > MAX_REWRITE_CHARS:
        return question
    return rewritten