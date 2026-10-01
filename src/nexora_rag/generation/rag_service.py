"""
Top-level RAG service (Step J-L): ties retrieval, generation and
citation-checking into ONE function.

    RagService.answer(question)
        -> retrieve + rerank (Step J/M/N)
        -> build prompt (Step K)
        -> call LLM (Step K)
        -> verify citations (Step L)
        -> return {answer, sources, trace_id, latency_ms}

This is the ONE function the API layer (Step P) will call -- it never
needs to know about chunks, prompts, or citation-checking itself.

Step W: every call is traced in LangSmith (rag_answer -> rewrite ->
retrieve_rerank -> ChatGroq). trace_id is the LangSmith trace id, so user
feedback can be linked back to the exact trace.
"""

import re
import time
import uuid

from ..core.cache import get_cached_answer, make_key, set_cached_answer
from ..guardrails.input_checks import check_input
from ..guardrails.output_checks import OUTPUT_REFUSAL, check_output
from ..guardrails.pii import mask_pii
from ..observability.tracing import current_trace_id, traceable
from ..retrieval.reranker import RerankingRetriever
from ..retrieval.rewrite import rewrite_query
from ..retrieval.routing import check_small_talk
from .citations import build_sources, validate_citations
from .llm import ask_llm
from .prompts import SYSTEM_PROMPT, build_user_message
from ..observability.metrics import CACHE_LOOKUPS

# Some LLM answers cite with fullwidth brackets (e.g. 【1】) instead of [1].
_FULLWIDTH_CITATION = re.compile("\u3010\\s*(\\d+)[^\u3011]*\u3011")


def normalize_citations(text: str) -> str:
    return _FULLWIDTH_CITATION.sub(r"[\1]", text)


REFUSALS = {
    "empty": "Please type a question.",
    "too_long": "Your question is too long. Please shorten it.",
    "jailbreak": "I can't help with that request. I can only answer questions about the company documents.",
    "out_of_scope": "I can only answer questions about Nexora's company documents.",
}


def _trace_id() -> str:
    """LangSmith trace id if tracing is on, otherwise a random uuid."""
    return current_trace_id() or str(uuid.uuid4())


# --- Step W: separate spans for the pipeline stages ---
@traceable(name="rewrite", run_type="chain")
def _rewrite(question: str, history: list[dict]) -> str:
    return rewrite_query(question, history)


@traceable(name="retrieve_rerank", run_type="retriever")
def _retrieve(retriever: RerankingRetriever, query: str, top_k: int, departments: list[str]):
    return retriever.search(query, top_k=top_k, departments=departments)


class RagService:
    def __init__(self, retriever: RerankingRetriever | None = None, top_k: int = 3):
        self.retriever = retriever or RerankingRetriever()
        self.top_k = top_k

    @traceable(name="rag_answer", run_type="chain")
    def answer(
        self,
        question: str,
        departments: list[str],
        user_id: str | None = None,
        history: list[dict] | None = None,
    ) -> dict:
        """Answers one question, grounded in the retrieved documents.

        Returns:
            answer     -- the LLM's answer, with any invalid/hallucinated
                          citations stripped out
            sources    -- [{id, doc_id, title, page, snippet}, ...] for
                          only the sources actually cited
            trace_id   -- LangSmith trace id (for logs/feedback)
            latency_ms -- total time for retrieval + generation
        """
        start = time.time()

        # Guardrail: bura ya ghalat input retriever/LLM/cache tak nahi jata
        ok, reason = check_input(question)
        if not ok:
            return {
                "answer": REFUSALS[reason],
                "sources": [],
                "trace_id": _trace_id(),
                "latency_ms": int((time.time() - start) * 1000),
            }

        # Small talk: RAG, cache aur LLM ke bina seedha jawab
        reply = check_small_talk(question)
        if reply is not None:
            return {
                "answer": reply,
                "sources": [],
                "trace_id": _trace_id(),
                "latency_ms": int((time.time() - start) * 1000),
            }

        # Follow-up sawal ko standalone banao (history na ho to skip)
        standalone = _rewrite(question, history) if history else question

        # Answer cache: key includes the user's departments (RBAC-safe)
        cache_key = make_key(standalone, departments)
        cached = get_cached_answer(cache_key)
        CACHE_LOOKUPS.labels(result="hit" if cached is not None else "miss").inc() 
        if cached is not None:
            return {
                "answer": cached["answer"],
                "sources": cached["sources"],
                "trace_id": _trace_id(),
                "latency_ms": int((time.time() - start) * 1000),
            }

        chunks = _retrieve(self.retriever, standalone, self.top_k, departments)

        if not chunks:
            return {
                "answer": "I don't have enough information to answer that.",
                "sources": [],
                "trace_id": _trace_id(),
                "latency_ms": int((time.time() - start) * 1000),
            }

        user_message = build_user_message(standalone, chunks)
        raw_answer = ask_llm(SYSTEM_PROMPT, user_message)
        raw_answer = normalize_citations(raw_answer)

        # Guardrail: leak ya jailbreak wala jawab user tak nahi jata, cache bhi nahi hota
        ok, _ = check_output(raw_answer)
        if not ok:
            return {
                "answer": OUTPUT_REFUSAL,
                "sources": [],
                "trace_id": _trace_id(),
                "latency_ms": int((time.time() - start) * 1000),
            }

        result = validate_citations(raw_answer, chunks)
        clean_answer = mask_pii(result["clean_answer"])
        sources = build_sources(chunks, result["valid"])
        for s in sources:
            s["snippet"] = mask_pii(s["snippet"])
        if sources:  # refusals and uncited answers are not cached
            set_cached_answer(cache_key, clean_answer, sources)

        return {
            "answer": clean_answer,
            "sources": sources,
            "trace_id": _trace_id(),
            "latency_ms": int((time.time() - start) * 1000),
        }


if __name__ == "__main__":
    service = RagService()
    response = service.answer(
        "How many days of paid maternity leave are there?",
        departments=["HR", "Product"],
    )

    print("Answer:", response["answer"])
    print("\nSources:")
    for s in response["sources"]:
        print(f"  [{s['id']}] {s['title']}, page {s['page']}")
    print(f"\nLatency: {response['latency_ms']}ms")
    print(f"Trace ID: {response['trace_id']}")