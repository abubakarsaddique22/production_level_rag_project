"""
Prompt building for grounded RAG answers (Step K/L).

Turns retrieved chunks into a numbered source list, and gives the LLM a
system prompt that says: answer only from the context, cite sources like
[1], [2], and say "I don't know" if the context doesn't have the answer.
This is what stops the model from making things up (hallucinating).
"""

SYSTEM_PROMPT = """You are the Nexora Knowledge Assistant...

Rules:
- Answer ONLY using the information in the context below. Do not use outside knowledge.
- Cite sources using ONLY this exact format: a number in square brackets right after the claim, like [1] or [2], matching the source number it came from. Example: "Employees get 20 days of annual leave [1]."
- Do NOT use any other citation style (no footnote markers, no special symbols, no line references). Only plain [1], [2], [3] etc.
- If the context does not contain enough information to answer, say "I don't have enough information to answer that" -- do not guess or make anything up.
- Keep answers clear and concise.
"""


def build_context_block(chunks: list[dict]) -> str:
    """Turns retrieved chunks into a numbered source list, e.g.:

        [1] Leave Policy, page 3
        Employees are entitled to 20 days of annual leave...

        [2] Expense Policy, page 2
        ...

    The LLM cites these numbers in its answer, and citations.py (next)
    checks that every number it used actually exists here.
    """
    blocks = []
    for i, chunk in enumerate(chunks, start=1):
        title = chunk.get("title") or chunk.get("doc_id", "Unknown document")
        page = chunk.get("page", "?")
        content = chunk.get("content", "")
        blocks.append(f"[{i}] {title}, page {page}\n{content}")
    return "\n\n".join(blocks)


def build_user_message(question: str, chunks: list[dict]) -> str:
    """Combines the question with the numbered context into one message."""
    context = build_context_block(chunks)
    return (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Answer using only the context above, with citations like [1], [2]."
    )


if __name__ == "__main__":
    fake_chunks = [
        {"title": "Leave Policy", "page": 3, "content": "Employees get 20 days annual leave."},
        {"title": "Expense Policy", "page": 2, "content": "Claims must be submitted within 14 days."},
    ]
    print(SYSTEM_PROMPT)
    print("---")
    print(build_user_message("How many annual leave days do I get?", fake_chunks))