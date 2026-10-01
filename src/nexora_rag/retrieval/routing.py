import re

# (pattern, tayyar jawab). Pattern poore message par match hota hai.
_SMALL_TALK = [
    (
        re.compile(
            r"(hi|hello|hey|salam|assalam o alaikum|assalamualaikum|"
            r"good morning|good afternoon|good evening)( there)?"
        ),
        (
            "Hello! Ask me anything about Nexora's HR, Engineering, Finance or "
            "Product documents."
        ),
    ),
    (
        re.compile(r"(thanks|thank you|thx|shukriya|shukria)( a lot| so much)?"),
        "You're welcome! Let me know if you have another question.",
    ),
    (
        re.compile(r"(bye|goodbye|see you|allah hafiz|khuda hafiz)"),
        "Goodbye! Come back any time you have a question.",
    ),
]


def check_small_talk(question: str) -> str | None:
    """Small talk ho to tayyar jawab wapas karta hai, warna None (matlab RAG)."""
    text = re.sub(r"[^\w\s]", "", question.lower())  # punctuation hatao
    text = " ".join(text.split())  # extra spaces hatao

    for pattern, reply in _SMALL_TALK:
        if pattern.fullmatch(text):
            return reply
    return None