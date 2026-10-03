# ADR 0005: LangSmith for tracing and user feedback (instead of Langfuse)

- **Status:** Accepted
- **Date:** 2026-10

## Context

Step W of the blueprint asks for tracing so a bad answer can be inspected stage by stage. The blueprint lists Langfuse and LangSmith.
The project already uses LangChain and LangGraph components (agent, Groq client), which LangSmith traces with very little code.

## Decision

Use **LangSmith**.

- `observability/tracing.py` wraps tracing. `current_trace_id()` returns the running trace id, and the API returns it as `trace_id` with every answer.
- `POST /v1/feedback` stores the thumbs up/down in Postgres **and** sends it to the LangSmith trace (`send_feedback`), so a bad rating links to the exact trace.
- A LangSmith failure is caught and logged. It must never break a user request.
- CI sets `LANGSMITH_TRACING=false`, so tests never send traces.

Prometheus and Grafana cover the numbers side (`rag_requests_total`, `rag_latency_seconds`, `rag_cache_lookups_total`).
LangSmith covers the per-request detail.

## Consequences

- (+) Per-request traces with user feedback attached.
- (-) LangSmith is a hosted service, so question text leaves the server. Fine for a demo corpus, a review item for real confidential data (Langfuse can be self-hosted).
- (-) Two tools (LangSmith and Prometheus) to look at. Accepted: they answer different questions.
