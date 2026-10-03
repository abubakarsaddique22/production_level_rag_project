# ADR 0002: Hybrid search (dense + BM25 + RRF) and a cross-encoder reranker

- **Status:** Accepted
- **Date:** 2026-09

## Context

Dense-only retrieval on the 30-question golden set missed three questions (q004, q005, q019).
The documents contain exact terms (policy names, numbers, file types) where keyword match helps,
and phrases buried inside chunks whose main topic looks different (q019: "submission days" inside a per-diem chunk).

## Decision

1. **Hybrid retrieval:** dense (Qdrant) and BM25 each return 30 candidates.
2. **Fusion with Reciprocal Rank Fusion**, `score = sum(1 / (60 + rank))`. Cosine similarity and BM25 scores are on different scales,
   so averaging raw scores is wrong. RRF uses only rank positions and needs no tuning (k = 60 is the standard constant).
3. **Cross-encoder rerank** (`cross-encoder/ms-marco-MiniLM-L-6-v2`) over a **pool of 15**, then the **top 3** go to the LLM.

## Evidence (30-question golden set, retrieval only, no LLM)

| Retriever | hit@3 | MRR | recall@3 | Failed questions |
|---|---|---|---|---|
| Dense | 0.900 | 0.794 | 0.850 | q004, q005, q019 |
| Hybrid (BM25 + RRF) | 0.933 | 0.850 | 0.867 | q004, q019 |
| Hybrid + rerank | 1.000 | 0.889 | 0.950 | none |

Hybrid fixed q005. The reranker fixed q004 and q019.

## Tuning after measuring latency

A stage breakdown showed reranking took 78% of the time (7.38 s average) with a pool of 30.
Cutting the pool to 15 and the LLM context from 5 chunks to 3 gave rerank 2.79 s with the same retrieval metrics.
Source: [`docs/latency_results.md`](../latency_results.md).

## Consequences

- (+) Every golden question finds its relevant chunk in the top 3.
- (-) The reranker is the most expensive stage on CPU (about 2.8 s).
- (-) 30 questions is a small sample. A 1.000 hit@3 here does not mean 1.000 in general.
- (-) Pool size 15 was validated on this set only. A larger corpus may need a larger pool.
