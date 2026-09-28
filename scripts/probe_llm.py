"""
Probe the LLM for rate-limit behaviour (Step R).

Sends a prompt of the same size as a real RAG prompt (~3000 tokens) several
times back to back and prints how long each call takes. If the first calls
are fast and a later call jumps to ~20 seconds, Groq is rate limiting.

Run from the project root:
    python scripts/probe_llm.py
"""

import sys
from pathlib import Path

# Make "src/" importable regardless of the current working directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import time

from nexora_rag.generation.llm import ask_llm

N_CALLS = 5
FILLER = "Nexora policy text about leave, expenses and deployment procedures. " * 160
SYSTEM = "You are a helpful assistant. Answer in one short sentence."


def main() -> None:
    for i in range(1, N_CALLS + 1):
        # call number goes first so every prompt is different (no prompt caching)
        user_message = f"[{i}] What is this text about?\n\n{FILLER}"
        start = time.perf_counter()
        answer = ask_llm(SYSTEM, user_message)
        elapsed = time.perf_counter() - start
        print(f"call {i}: {elapsed:5.2f}s  ({len(user_message)} chars in, {len(answer)} chars out)")


if __name__ == "__main__":
    main()