import re
import unicodedata

MAX_QUESTION_CHARS = 1000

# Zero-width characters jo attacker chhupane ke liye use karte hain
ZERO_WIDTH = dict.fromkeys(map(ord, "\u200b\u200c\u200d\u2060\ufeff"))

JAILBREAK_PATTERNS = [
    r"(ignore|disregard|forget) (all |any |your )?(previous |prior |above )?(instructions|rules|prompts?)",
    r"you are now (dan|unrestricted|in developer mode)",
    r"\b(dan mode|developer mode|jailbreak)\b",
    r"(reveal|show|print|repeat|tell me) (me )?(your |the )?(system|hidden|initial) (prompt|instructions)",
    r"pretend (you have|there are) no (rules|restrictions)",
    r"act as (an? )?(admin|administrator|root|superuser)",
    r"(bypass|override|disable) (the )?(security|filters?|guardrails?|access control|rbac)",
]
COMPILED = [re.compile(p) for p in JAILBREAK_PATTERNS]

OUT_OF_SCOPE_PATTERNS = [
    r"\b(write|compose|generate|create) (me )?(a |an |some )?(poem|song|story|essay|joke|python|code|script|program)\b",
    r"\btell me (a )?joke\b",
    r"\bweather (today|tomorrow|in|forecast)\b",
    r"\b(bitcoin|cricket score|match score)\b",
    r"\btranslate\b.+\b(into|to)\b",
    r"\bsolve (this |the )?(equation|integral|math)\b",
]
OUT_OF_SCOPE = [re.compile(p) for p in OUT_OF_SCOPE_PATTERNS]


def normalize(text: str) -> str:
    # NFKC: fullwidth/lookalike characters ko normal bana deta hai
    text = unicodedata.normalize("NFKC", text).translate(ZERO_WIDTH)
    return re.sub(r"\s+", " ", text).strip().lower()


def check_input(question: str) -> tuple[bool, str | None]:
    """(True, None) agar theek hai, warna (False, reason)."""
    if not question or not question.strip():
        return False, "empty"
    if len(question) > MAX_QUESTION_CHARS:
        return False, "too_long"
    clean = normalize(question)
    for pattern in COMPILED:
        if pattern.search(clean):
            return False, "jailbreak"
        
    for pattern in OUT_OF_SCOPE:
        if pattern.search(clean):
            return False, "out_of_scope"
    return True, None