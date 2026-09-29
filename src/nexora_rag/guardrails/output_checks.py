from .input_checks import normalize

# Ye phrases normal jawab mein nahi aane chahiye: agar aayen to leak ya jailbreak hai
LEAK_MARKERS = [
    "security rules:",
    "untrusted reference data",
    "you are the nexora knowledge assistant",
    "<document",
    "developer mode enabled",
    "i am dan",
]

OUTPUT_REFUSAL = "Sorry, I can't provide that response. Please rephrase your question."


def check_output(answer: str) -> tuple[bool, str | None]:
    """(True, None) agar jawab theek hai, warna (False, 'leak')."""
    clean = normalize(answer)
    for marker in LEAK_MARKERS:
        if marker in clean:
            return False, "leak"
    return True, None