# Resume bullets (only numbers measured in this repo)

Every number below comes from a file in the repository. Source in brackets.

## Pick 4 to 5 for your resume

- Built a production-style RAG assistant over 10 company PDFs with hybrid retrieval (dense + BM25 + Reciprocal Rank Fusion) and cross-encoder reranking; raised hit@3 from 0.90 to 1.00 and MRR from 0.794 to 0.889 on a hand-written 30-question golden set. [`docs/eval/report.md`]
- Profiled the pipeline stage by stage, found reranking took 78% of latency (7.38 s), and cut it to 2.79 s by shrinking the candidate pool from 30 to 15 with no change in retrieval metrics. [`docs/latency_results.md`]
- Evaluated answers with DeepEval and a self-hosted judge: faithfulness 0.989, answer relevancy 0.933; wrote a regression-check script that fails when a metric drops beyond tolerance. [`docs/eval/report.md`, `scripts/check_regression.py`]
- Enforced role-based access control inside the vector search and the cache key (5 roles, 4 departments), verified by 28 integration tests and a 29-case data-driven red-team suite (2 known gaps tracked as xfail). [`tests/`]
- Shipped CI/CD to AWS: GitHub Actions (lint, mypy, 230 tests at 84.58% coverage, Docker build), image to ECR, zero-SSH deploy through SSM to EC2 with automatic health-check rollback; Prometheus/Grafana monitoring and LangSmith tracing. [`.github/workflows`, `docs/deployment.md`]
- Compared a LangGraph agent against plain RAG on 20 questions (16/20 each) and published the negative result, including the failure mode the groundedness check cannot catch. [`docs/agent_compare_results.md`]

## Do not write

- "Reduced latency from 21 s to 4.5 s". The 21 s baseline was measured while the Groq free tier was throttling, and the two runs used different settings. Use the rerank 7.38 s to 2.79 s bullet instead.
- Any uptime, requests per day, user count or production latency. None of these is measured yet.
- "100% accuracy". hit@3 = 1.00 is on 30 answerable questions only. Say "on a 30-question golden set".
- "Zero data leaks in production". Say "verified by 28 integration tests".

## Numbers to measure once more (then add to the resume)

- p50 / p95 latency of the **deployed** AWS instance (`scripts/measure_latency.py` against the live URL).
- Chunk-size ablation result (ADR 0003).
- Latency after adding PII masking (not re-measured yet).

## Interview angle: "Tell me about a project you are proud of and what you would improve"

**Story (about 90 seconds):**
"I built Nexora, a RAG assistant where answers are cited and limited to the departments a user's role can see.
I measured everything: on a 30-question golden set, dense search got hit@3 0.90, and hybrid search plus a cross-encoder reranker got 1.00.
When it was slow I did not guess. A stage breakdown showed the reranker was 78% of the time, so I cut its candidate pool from 30 to 15 and went from 7.4 to 2.8 seconds with no quality loss.
I put access control inside the vector search, so the model never sees forbidden chunks, and I deployed it on AWS with a GitHub Actions pipeline that rolls back automatically if the health check fails.
I am proud that I also published the negative results: my agent was not better than plain RAG."

**What I would improve (have these ready):**
1. Contextual relevancy is 0.121. I suspect large chunks and I would run a chunk-size ablation to prove or disprove it.
2. The golden set is small and has no unanswerable or multi-document questions, so refusal accuracy is unmeasured and 4 multi-document questions fail.
3. Single EC2 instance and long-lived IAM keys. I would move CI to GitHub OIDC and add a load test.

**Likely follow-ups:** why RRF instead of weighted score averaging; why the RBAC filter is not a post-filter; how the cache avoids leaking across roles; what the agent comparison taught you; how rollback works; what happens if Qdrant Cloud or Groq is down.
