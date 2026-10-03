# ADR 0001: Qdrant as the vector store

- **Status:** Accepted
- **Date:** 2026-09

## Context

The system needs dense vector search where every query is **filtered by the user's allowed departments**
(see [ADR 0004](0004-rbac-inside-retrieval.md)). The production host is a single EC2 `t3.medium` (4 GB RAM) that already runs
torch, a reranker, Presidio, Postgres, Redis, Prometheus and Grafana, so memory is tight.

## Decision

Use **Qdrant**, through **Qdrant Cloud** for the deployed app. A local Qdrant container is still defined
in `docker-compose.yml` and as the `local-qdrant` profile in `docker-compose.prod.yml`.

Why it fits this code base:

- Payload filtering is a first-class feature. `department`, `doc_id` and `version` are indexed as keyword payload fields
  (`VectorStore.ensure_collection`), and the department filter runs **inside** the search with `MatchAny`.
- Point ids are a UUID5 derived from `chunk_id`, so re-indexing is idempotent: the same chunk overwrites itself and never duplicates.
- Running it as a managed service keeps the vector index off the small EC2 instance.

## Consequences

- (+) Filter and search are one call, so RBAC cannot be skipped by a post-filter bug.
- (+) The index can be rebuilt any time from `chunks.json` plus the embedder.
- (-) One more external dependency and network hop. If Qdrant Cloud is down, retrieval is down.
- (-) BM25 is **not** in Qdrant: it runs in-process from `chunks.json` (`SparseIndex`), which is why `chunks.json` is committed to git.

## Alternatives considered

> Edit this section to match what you actually compared. These are the usual candidates, not a record of experiments.

- **pgvector** (one database for everything): fewer moving parts, but filtered ANN search and tuning are more manual.
- **FAISS**: fast and simple, but no built-in metadata filtering or persistence, so RBAC would be hand-written.
- **Chroma**: easy locally, less common for a production-style deployment.
