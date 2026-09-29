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
"""

import time
import uuid
import re 

from ..retrieval.reranker import RerankingRetriever
from .citations import build_sources, validate_citations
from .llm import ask_llm
from .prompts import SYSTEM_PROMPT, build_user_message
from ..core.cache import get_cached_answer, make_key, set_cached_answer
from ..retrieval.rewrite import rewrite_query
from ..retrieval.routing import check_small_talk

# Some LLM answers cite with fullwidth brackets (e.g. 【1】) instead of [1].
_FULLWIDTH_CITATION = re.compile("\u3010\\s*(\\d+)[^\u3011]*\u3011")


def normalize_citations(text: str) -> str:
    return _FULLWIDTH_CITATION.sub(r"[\1]", text)

class RagService:
    def __init__(self, retriever: RerankingRetriever | None = None, top_k: int = 3):
        self.retriever = retriever or RerankingRetriever()
        self.top_k = top_k

    def answer(self, 
               question: str,
               departments: list[str],
               user_id: str | None = None,
                history: list[dict] | None = None) -> dict:
        """Answers one question, grounded in the retrieved documents.

        Returns:
            answer     -- the LLM's answer, with any invalid/hallucinated
                          citations stripped out
            sources    -- [{id, doc_id, title, page, snippet}, ...] for
                          only the sources actually cited
            trace_id   -- unique id for this request (for logs/feedback later)
            latency_ms -- total time for retrieval + generation
        """
        start = time.time()

        # Small talk: RAG, cache aur LLM ke bina seedha jawab
        reply = check_small_talk(question)
        if reply is not None:
            return {
                "answer": reply,
                "sources": [],
                "trace_id": str(uuid.uuid4()),
                "latency_ms": int((time.time() - start) * 1000),
            }
        
        # Follow-up sawal ko standalone banao (history na ho to skip)
        standalone = rewrite_query(question, history) if history else question

        # Answer cache: key includes the user's departments (RBAC-safe)
        cache_key = make_key(question, departments)
        cached = get_cached_answer(cache_key)
        if cached is not None:
            return {
                "answer": cached["answer"],
                "sources": cached["sources"],
                "trace_id": str(uuid.uuid4()),
                "latency_ms": int((time.time() - start) * 1000),
            }

     
        
        chunks = self.retriever.search(standalone, top_k=self.top_k,departments=departments)

        if not chunks:
            return {
                "answer": "I don't have enough information to answer that.",
                "sources": [],
                "trace_id": str(uuid.uuid4()),
                "latency_ms": int((time.time() - start) * 1000),
            }

        user_message = build_user_message(standalone, chunks)
        raw_answer = ask_llm(SYSTEM_PROMPT, user_message)

        result = validate_citations(raw_answer, chunks)
        sources = build_sources(chunks, result["valid"])
        if sources:  # refusals and uncited answers are not cached
                    set_cached_answer(cache_key, result["clean_answer"], sources)

        return {
            "answer": result["clean_answer"],
            "sources": sources,
            "trace_id": str(uuid.uuid4()),
            "latency_ms": int((time.time() - start) * 1000),
        }


if __name__ == "__main__":
    service = RagService()
    response = service.answer("How many days of paid maternity leave are there?",departments=["HR", "Product"],)

    print("Answer:", response["answer"])
    print("\nSources:")
    for s in response["sources"]:
        print(f"  [{s['id']}] {s['title']}, page {s['page']}")
    print(f"\nLatency: {response['latency_ms']}ms")
    print(f"Trace ID: {response['trace_id']}")