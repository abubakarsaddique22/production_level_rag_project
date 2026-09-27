"""
Citation verification (Step L).

The LLM is told to cite sources as [1], [2], etc. (see prompts.py). This
module checks that every citation number it actually used really exists
in the retrieved chunks -- if the model ever makes up a citation number
that wasn't given to it, that's a sign of hallucination, and it's
stripped from the answer rather than shown to the user.
"""

import re

CITATION_RE = re.compile(r"\[(\d+)\]")

# Safety net: some models occasionally emit tool-citation-style markers
# like "\u3010123456\u2020L1-L4\u3011" instead of the instructed "[1]" format
# (prompts.py explicitly forbids this, but models don't always comply).
# These aren't recognized as valid citations either way, so they're just
# stripped as noise rather than left visible in the answer.
STRAY_CITATION_RE = re.compile(r"\u3010[^\u3011]*\u3011")


def extract_citations(answer: str) -> list[int]:
    """Returns every citation number found in the answer, e.g. "...[1]
    and [2]..." -> [1, 2]. Duplicates removed, order preserved."""
    seen = []
    for match in CITATION_RE.finditer(answer):
        n = int(match.group(1))
        if n not in seen:
            seen.append(n)
    return seen


def validate_citations(answer: str, chunks: list[dict]) -> dict:
    """Checks citations used in `answer` against the chunks that were
    actually retrieved (numbered 1..len(chunks), same order as
    prompts.build_context_block).

    Returns:
        used         -- citation numbers found in the answer
        valid        -- citation numbers that exist in chunks
        invalid      -- citation numbers that do NOT exist (hallucinated)
        clean_answer -- the answer with invalid citations removed
    """
    used = extract_citations(answer)
    valid = [n for n in used if 1 <= n <= len(chunks)]
    invalid = [n for n in used if n not in valid]

    clean_answer = answer
    for n in invalid:
        clean_answer = clean_answer.replace(f"[{n}]", "")
    clean_answer = STRAY_CITATION_RE.sub("", clean_answer)

    return {
        "used": used,
        "valid": valid,
        "invalid": invalid,
        "clean_answer": clean_answer.strip(),
    }


def build_sources(chunks: list[dict], valid_citation_numbers: list[int]) -> list[dict]:
    """Builds the "sources" list for the API response -- only the
    sources actually cited (and valid), in citation order."""
    sources = []
    for n in valid_citation_numbers:
        chunk = chunks[n - 1]  # citation [1] -> chunks[0]
        sources.append({
            "id": n,
            "doc_id": chunk.get("doc_id"),
            "title": chunk.get("title"),
            "page": chunk.get("page"),
            "snippet": chunk.get("content", "")[:200],
        })
    return sources


if __name__ == "__main__":
    fake_chunks = [
        {"doc_id": "NX-HR-001", "title": "Leave Policy", "page": 3, "content": "Employees get 20 days annual leave."},
        {"doc_id": "NX-HR-001", "title": "Leave Policy", "page": 4, "content": "Paid maternity leave is 90 calendar days."},
    ]
    fake_answer = "Paid maternity leave is 90 calendar days. [2] Also see [5] for more."

    result = validate_citations(fake_answer, fake_chunks)
    print("Used:", result["used"])
    print("Valid:", result["valid"])
    print("Invalid (hallucinated):", result["invalid"])
    print("Clean answer:", result["clean_answer"])
    print("Sources:", build_sources(fake_chunks, result["valid"]))