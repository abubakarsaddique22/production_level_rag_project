# Nexora Knowledge Assistant

A production-style **Retrieval-Augmented Generation (RAG)** system for an internal company knowledge base.
Employees ask questions in natural language and get answers **with citations**, restricted to the
documents their role is allowed to see. The project follows a 26-step (A to Z) production RAG blueprint and
covers the full path from ingestion to evaluation: hybrid search, reranking, auth and RBAC, caching,
conversation memory, guardrails, a LangGraph agent and a measured evaluation.

![CI](https://github.com/abubakarsaddique22/production_level_rag_project/actions/workflows/ci.yml/badge.svg)

> **Status:** Steps A to V are done. Steps W to Z (observability, experiment tracking and the rest of the blueprint) are planned.
> See [What is not done](#what-is-not-done).

---

## Table of contents

1. [Highlights](#highlights)
2. [Architecture](#architecture)
3. [Tech stack](#tech-stack)
4. [Steps covered](#steps-covered)
5. [Project structure](#project-structure)
6. [Quickstart](#quickstart)
7. [API](#api)
8. [Retrieval pipeline (Steps A to P)](#retrieval-pipeline-steps-a-to-p)
9. [Auth and RBAC (Step Q)](#auth-and-rbac-step-q)
10. [Caching and rate limiting (Step R)](#caching-and-rate-limiting-step-r)
11. [Conversation memory, feedback and routing (Step S and rest of Step O)](#conversation-memory-feedback-and-routing-step-s-and-rest-of-step-o)
12. [Database migrations (Alembic)](#database-migrations-alembic)
13. [Guardrails and security (Step T)](#guardrails-and-security-step-t)
14. [LangGraph agent (Step U)](#langgraph-agent-step-u)
15. [Evaluation (Step V)](#evaluation-step-v)
16. [Performance](#performance)
17. [Testing and CI](#testing-and-ci)
18. [Design decisions and lessons learned](#design-decisions-and-lessons-learned)
19. [What is not done](#what-is-not-done)

---

## Highlights

- **Hybrid retrieval:** dense vectors plus BM25, fused with Reciprocal Rank Fusion (RRF), then reranked by a cross-encoder.
  Retrieval hit@3 goes from 0.90 (dense only) to **1.00** (hybrid + rerank) on the golden set.
- **Cited answers:** every answer carries `[n]` citations that are validated against the retrieved chunks.
  Answers without valid citations return no sources and are never cached.
- **Role-based access control:** JWT login, each role maps to a set of departments, and the department filter
  is applied inside the vector search, not after it.
- **Fast and cheap on repeat questions:** exact-match answer cache in Redis and per-user rate limiting.
  Median latency over the 30 golden questions went from about 21 s to about 4.5 s, and a warm cache answers instantly.
- **Conversation memory:** sessions and messages in Postgres, follow-up questions are rewritten into standalone questions.
- **Guardrails:** input checks, prompt hardening against injected documents, PII masking with Presidio,
  output leak checks, and a data-driven red-team test suite that runs in CI.
- **Agent:** a LangGraph agent with a router, a safe calculator, a ticket lookup tool, a groundedness check and a bounded retry loop,
  evaluated honestly against plain RAG.
- **Measured quality:** a 30-question golden set, retrieval metrics, DeepEval generation metrics with a self-hosted judge,
  a generated report with charts and a regression check script.

---

## Architecture

```
                         +-----------------------------+
   User ---- JWT ----->  |  FastAPI                    |
                         |  /v1/chat   /v1/agent/chat  |
                         |  /v1/sessions  /v1/feedback |
                         +--------------+--------------+
                                        |
        rate limit (Redis) -> auth (JWT) -> role -> allowed departments
                                        |
                                        v
                            +-----------------------+
                            |      RagService       |
                            +-----------------------+
   1. input guard (length, jailbreak, out of scope, empty)
   2. small-talk routing (ready reply, no retrieval, no LLM)
   3. query rewrite using the last 6 messages (follow-up -> standalone question)
   4. answer cache lookup (Redis, key = standalone question + departments)
   5. retrieval: dense (Qdrant) + BM25 -> RRF -> cross-encoder rerank (pool 15) -> top 3
   6. prompt: chunks wrapped as untrusted <document> data + security rules
   7. LLM (Groq) -> normalize citations -> validate citations
   8. output check (prompt leak) -> PII masking -> cache (only if sources exist)
                                        |
                                        v
              answer + sources + trace_id  ->  saved as messages in Postgres
```

The agent endpoint (`/v1/agent/chat`) uses the same `RagService` as one of its tools (see [Step U](#langgraph-agent-step-u)).

---

## Tech stack

| Area | Technology |
|---|---|
| Language | Python 3.12 |
| API | FastAPI, Uvicorn, Pydantic |
| Vector database | Qdrant (Qdrant Cloud for the app, a local container is also defined in `docker-compose.yml`) |
| Sparse retrieval | BM25 with Reciprocal Rank Fusion |
| Reranking | Cross-encoder reranker (CPU) |
| LLM | Groq free tier, `openai/gpt-oss-120b` (used for all final numbers) |
| Cache and rate limit | Redis 7, slowapi |
| Relational DB | PostgreSQL 16, async SQLAlchemy, Alembic migrations |
| Auth | JWT with role-based access control |
| Guardrails | Regex-based input and output checks, Presidio + spaCy `en_core_web_sm` for PII |
| Agent | LangGraph |
| Evaluation | Own retrieval metrics, DeepEval, Ollama as self-hosted judge |
| Testing and CI | pytest, GitHub Actions |
| Dev environment | Windows, VS Code, PowerShell, `.venv`, `pip` + `requirements.txt` |

---

## Steps covered

| Step | Topic | Status |
|---|---|---|
| A to P | Foundation, ingestion, embeddings, Qdrant indexing, hybrid search, reranking, generation with citations, FastAPI service (`/health`, `/v1/chat`) | Done |
| O (rest) | Query rewrite and small-talk routing (calculation routing deliberately skipped) | Done |
| Q | JWT authentication and RBAC | Done |
| R | Caching and rate limiting | Done |
| S | Conversation memory and feedback (plus Alembic migrations) | Done |
| T | Guardrails and security, red-team tests, CI | Done |
| U | LangGraph agent with tools, groundedness check, API endpoint | Done |
| V | Evaluation: golden set results, report, regression check | Done |
| W to Z | Observability, experiment tracking and the remaining blueprint steps | Planned |

---

## Project structure

Main parts of the repository:

```
nexora-rag/
├── src/nexora_rag/
│   ├── config.py                  # settings (env prefix RAG_)
│   ├── api/
│   │   ├── main.py                # app, routers
│   │   ├── deps.py                # auth dependencies, role -> departments
│   │   ├── schemas.py             # request / response models
│   │   └── routers/               # chat, sessions, feedback, agent
│   ├── core/
│   │   ├── cache.py               # answer cache (make_key, get / set)
│   │   └── rate_limit.py          # slowapi + Redis
│   ├── db/models.py               # User, ChatSession, Message, Feedback
│   ├── retrieval/
│   │   ├── rewrite.py             # rewrite_query(question, history)
│   │   └── routing.py             # check_small_talk(question)
│   ├── generation/
│   │   ├── llm.py                 # Groq client (retries)
│   │   ├── prompts.py             # hardened system prompt, document tags
│   │   └── rag_service.py         # the full answer() pipeline
│   ├── guardrails/
│   │   ├── input_checks.py        # check_input
│   │   ├── pii.py                 # mask_pii
│   │   └── output_checks.py       # check_output
│   ├── agents/
│   │   ├── state.py               # agent state
│   │   ├── tools.py               # calculator, ticket lookup, kb_search
│   │   └── graph.py               # LangGraph graph
│   └── evaluation/
│       ├── dataset.py             # golden set validation (pydantic)
│       ├── judge.py               # Ollama wrapped as a DeepEval judge
│       ├── metrics.py             # retrieval metrics + DeepEval contextual metrics
│       └── ragas_runner.py        # generation metrics (faithfulness, answer relevancy)
├── scripts/
│   ├── create_tables.py, seed_users.py
│   ├── measure_latency.py, latency_breakdown.py, probe_llm.py
│   ├── compare_agent.py
│   ├── run_eval.py                # full eval, logs to docs/eval/results.csv
│   ├── make_report.py             # builds docs/eval/report.md and charts
│   └── check_regression.py        # compares two eval runs
├── alembic/                       # migrations (0001 baseline, 0002 feedback constraint)
├── data/eval/                     # golden set, agent comparison questions and answers
├── docs/                          # latency results, agent results, eval report and charts
├── tests/
│   ├── unit/                      # pure logic, fakes, no external services
│   ├── integration/               # real Postgres + Redis, fake RagService
│   └── redteam/                   # attacks.json + test_red_team.py
├── docker-compose.yml             # api, qdrant, redis, postgres
├── requirements.txt
└── .github/workflows/ci.yml
```

---

## Quickstart

> Commands are for PowerShell on Windows with a `.venv`. Check the exact module and script names against your checkout.

**1. Install**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

`requirements.txt` has a CPU-only PyTorch index on its first line and installs the spaCy model `en_core_web_sm` from a wheel URL,
so no extra steps are needed on Linux CI.

**2. Configure**

Create a `.env` file (it is git-ignored). All settings use the `RAG_` prefix and are defined in `src/nexora_rag/config.py`.
The only setting without a default is the Qdrant URL.

```
RAG_QDRANT_URL=<your Qdrant Cloud endpoint>
RAG_GROQ_MODEL=openai/gpt-oss-120b
# Also set the Qdrant API key, Groq API key and a real JWT secret (see config.py for the exact names).
# The default JWT secret is a placeholder and must be changed.
```

**3. Start Redis and Postgres**

```powershell
docker compose up -d redis postgres
```

**4. Create tables and seed test users**

```powershell
python scripts/create_tables.py
python scripts/seed_users.py
```

This creates the tables and seeds 5 test users, one per role. Alembic migrations are also available (see below).

**5. Ingest and index the documents**

Run the ingestion and indexing scripts for Steps B to F (names depend on your checkout).

**6. Run the API**

```powershell
uvicorn src.nexora_rag.api.main:app --reload
```

Open `http://localhost:8000/docs` for Swagger, log in as a seeded user and call `/v1/chat`.

---

## API

| Method and path | Purpose | Notes |
|---|---|---|
| `GET /health` | Liveness check | public |
| `POST /v1/chat` | RAG answer with citations | auth required, 20 requests per minute per user, optional `session_id` |
| `POST /v1/agent/chat` | Agent answer (router, tools, groundedness check) | auth required, 10 requests per minute, hard timeout, returns `route`, `grounded`, `tool_calls` |
| `GET /v1/sessions/{id}` | Read a conversation | owner only |
| `POST /v1/feedback` | Thumbs up or down for an answer | keyed by `trace_id`, rating `1` or `-1`, owner only, a second submit updates the same row |

Every answer response contains the `answer`, the `sources` (document, page, snippet) and a `trace_id`.
Refusals (blocked input, unsafe output) are normal `200` answers with empty `sources`.

---

## Retrieval pipeline (Steps A to P)

1. **Ingestion:** documents are parsed and split into page-level chunks with metadata
   (document id, title, page, department).
2. **Embeddings and indexing:** chunks are embedded and stored in Qdrant together with their metadata,
   so department filters can be applied at search time.
3. **Hybrid search:** a dense vector search and a BM25 search are fused with RRF.
4. **Reranking:** a cross-encoder reranks a candidate pool of **15** chunks and the top **3** go to the LLM.
   The pool was cut from 30 to 15 and the prompt from 5 chunks to 3 after measuring that quality did not change (see [Performance](#performance)).
5. **Generation with citations:** the LLM must cite chunks as `[n]`. Fullwidth brackets (`【1】`) that some models produce
   are normalized to `[1]` before validation. Only valid citations become `sources`.
6. **Service:** FastAPI exposes this as `/v1/chat`.

---

## Auth and RBAC (Step Q)

- Login returns a JWT. Each user has a role, and each role maps to a set of departments.
- The departments are passed into the retrieval step and applied as a filter inside the search, so a user can never retrieve
  chunks from a department their role cannot see.
- The answer cache key includes the departments, so one role can never get a cached answer built from another role's documents.
- Verified manually in Swagger and by `tests/integration/test_rbac.py` (28 tests).

---

## Caching and rate limiting (Step R)

- **Rate limiting:** per-user limits with slowapi and Redis (`/v1/chat` 20 per minute, `/v1/agent/chat` 10 per minute).
- **Exact-match answer cache:** key = rewritten standalone question + departments, with a configurable TTL.
  Only answers that have valid sources are cached. Small talk, refusals and leaky answers are never cached.
- **Query embedding cache** was evaluated and rejected: embedding takes about 0.08 s, so it is not worth it.
- **Finding the real bottleneck:** a stage-by-stage latency breakdown showed that reranking was 78 percent of the time
  (7.38 s average) before any change. See [Performance](#performance).
- A test of the cache (`tests/integration/test_answer_cache.py`) caught two real bugs while it was being written.

---

## Conversation memory, feedback and routing (Step S and rest of Step O)

- **Models:** `ChatSession`, `Message` (integer auto-increment id, so message order is exact even when timestamps tie) and `Feedback`.
- **Chat with `session_id`:** messages are saved after each turn and the last 6 messages are loaded as history.
- **Query rewrite:** `rewrite_query(question, history)` turns a follow-up ("and for managers?") into a standalone question.
  It falls back to the original question if the rewrite fails. The standalone question is used for the cache key, the retrieval and the prompt.
- **Small-talk routing:** `check_small_talk` uses rules (full-match on the lowercased message without punctuation)
  for English and Roman Urdu greetings, thanks and goodbyes. It returns a ready reply without touching retrieval or the LLM.
- **Calculation routing** was deliberately skipped in `RagService`. Calculations are handled by the agent instead.
- **Feedback:** `POST /v1/feedback` stores a rating per `trace_id` with a unique constraint on `(user_id, trace_id)`.
- Covered by `tests/integration/test_sessions.py` (isolation between users, history limits, feedback rules).

---

## Database migrations (Alembic)

- Async Alembic template. `alembic/env.py` reads the database URL from the app settings and the metadata from the SQLAlchemy models.
- `0001_baseline` (hand-written) covers `users`, `chat_sessions`, `messages` and `feedback` and was stamped on the existing database.
- `0002` adds the unique constraint `uq_feedback_user_trace` on `feedback (user_id, trace_id)`.
- Verified on a fresh empty database: `upgrade head` runs both migrations and `alembic check` reports no new operations.

---

## Guardrails and security (Step T)

The core guardrails are plain Python and regex, so they are deterministic, fast and testable in CI without any external service.
Presidio is used for PII.

| Layer | What it does |
|---|---|
| Input checks (`check_input`) | Rejects empty input, too long input, known jailbreak patterns and out-of-scope requests. Each reason has its own refusal message. |
| Prompt hardening | One system prompt with security rules. Retrieved chunks are wrapped in `<document id title page>` tags as untrusted data, and the model is told never to follow instructions inside them. `neutralize()` strips fake or fullwidth document tags from chunk text and titles. |
| Citation validation | Citations must point to retrieved chunks. Answers without valid citations get no sources. |
| Output checks (`check_output`) | Detects system prompt leaks with leak markers and replaces the answer with a refusal. Leaky answers are never cached. |
| PII masking (`mask_pii`) | Presidio with spaCy for email, phone, credit card and IBAN, plus custom `PK_CNIC` and `PK_PHONE` recognizers. Applied to the answer and to source snippets before caching. Person names are deliberately not masked, because names of colleagues and customers are part of normal business answers. |

### Red-team tests

`tests/redteam/attacks.json` holds attacks as data and `tests/redteam/test_red_team.py` runs them deterministically
(no Groq, Qdrant or Redis needed), so they run in CI on every push.

- 15 input attacks, 6 document-injection attacks, 5 output attacks, 3 RBAC attacks
- 2 **known gaps** are kept as `xfail`: paraphrased jailbreaks and Roman Urdu jailbreaks. They document what the regex layer does not catch.

### OWASP LLM Top 10 mapping

| Risk | Mitigation in this project |
|---|---|
| LLM01 Prompt injection | Input checks, untrusted document tags, `neutralize()`, red-team tests |
| LLM02 Sensitive information disclosure | RBAC filter inside the search, PII masking, department-aware cache key |
| LLM05 Improper output handling | Output checks, citation validation, answers never rendered as code |
| LLM06 Excessive agency | Agent has a fixed tool list, a calculator based on `ast` (no `eval`), a tool call cap and a retry cap |
| LLM07 System prompt leakage | Leak markers in `check_output` |
| LLM10 Unbounded consumption | Per-user rate limits, input length limit, LLM retry limits, agent timeout |

---

## LangGraph agent (Step U)

**Graph:** `guard` -> `router` -> one of `kb` / `calc` / `ticket` -> `synth` -> `check` -> (retry or finish).

- `guard` runs the same input checks and small-talk routing as the normal pipeline, and the synthesized answer also goes through
  `check_output` and `mask_pii`, so the agent path is not less safe than `/v1/chat`.
- **Tools:** a calculator (`ast`-based, never `eval`), a mock `lookup_ticket`, and `kb_search`, which wraps the same `RagService`
  with the user's departments and history.
- **Limits:** at most 2 retries and at most 4 tool calls. Only the calculation route retries.
- **Endpoint:** `POST /v1/agent/chat` with the same auth, departments and session saving as `/v1/chat`,
  a 10 per minute limit and a timeout. The response adds `route`, `grounded` and `tool_calls`.
- Tested with a fake LLM and fake service (retry loop, tool cap, departments reaching the service, guard and PII wiring)
  and through the API (`tests/integration/test_agent_api.py`, including the timeout case).

### Agent vs plain RAG (20 questions, final code)

Questions are in `data/eval/agent_compare.json` and results in `docs/agent_compare_results.md`.

| | Plain RAG | Agent |
|---|---|---|
| Overall | 16 / 20 | 16 / 20 |
| Calculation questions | 14 / 18 | 14 / 18 |
| Lookup questions | 2 / 2 | 2 / 2 |
| Single-document calculation questions | 14 / 14 | 14 / 14 |
| Multi-document questions | 0 / 4 | 0 / 4 |

**Honest conclusion:** on this data the agent is **not better** than plain RAG. Both pass every single-document calculation.
Both fail the four multi-document questions (`c013`, `c016`, `c017`, `c018`).

- A sub-question fallback fixed `c013` in an isolated run but not reliably in the full run, so it is not counted as a win.
- On a held-out multi-document question the agent gave a confident wrong number (it applied a percentage to gross salary instead of basic salary).
  The groundedness check cannot catch this kind of error, because the number was computed correctly from a wrongly retrieved fact.
- The agent is slower than plain RAG on the failing multi-document questions (20 to 45 seconds).
- **Next step would be real multi-hop retrieval**, for example splitting the question up front or iterative retrieval.

---

## Evaluation (Step V)

All numbers come from a hand-written golden set of 30 questions (`data/eval/golden_dataset.json`).
Final run: `final_k3` (top_k = 3, answer model `openai/gpt-oss-120b`, DeepEval judge self-hosted on Ollama).
The full report with charts is in [`docs/eval/report.md`](docs/eval/report.md).

### Retrieval (retrieval-only, no LLM)

| Retriever | hit@3 | MRR | recall@3 |
|---|---|---|---|
| Dense | 0.900 | 0.794 | 0.850 |
| Hybrid (BM25 + RRF) | 0.933 | 0.850 | 0.867 |
| Hybrid + Rerank | **1.000** | **0.889** | **0.950** |

- Dense fails on q004, q005 and q019. Hybrid search fixes q005, and the cross-encoder reranker fixes q004 and q019.
- With the reranker, every golden question finds its relevant chunk in the top 3.

### Generation and DeepEval

| Metric | Score |
|---|---|
| Faithfulness | 0.989 |
| Answer relevancy | 0.933 |
| Contextual precision | 0.928 |
| Contextual recall | 0.967 |
| Contextual relevancy | 0.121 |

An earlier run with the older pipeline (top_k = 5) gave faithfulness 1.0 and answer relevancy about 0.95.
The difference is within judge noise, so a separate top_k = 5 baseline run was not repeated.

### Limitations of the evaluation

- **Contextual relevancy is low (0.121).** Chunks are large page-level text, so they contain much content unrelated to the question.
  This explanation has not been verified yet.
- **Judge noise.** DeepEval scores moved by about 0.02 to 0.03 between two runs, so small differences are not real improvements.
- **Small golden set.** 30 questions, all answerable. There are no unanswerable, adversarial or multi-document questions,
  so refusal accuracy is not measured. Multi-document behaviour is only covered by the agent comparison above.

### Reproduce

```powershell
python scripts/run_eval.py          # full eval, slow (DeepEval judge on Ollama), logs to docs/eval/results.csv
python scripts/make_report.py       # rebuild docs/eval/report.md and charts from the saved results
```

Optional regression check after changing the pipeline (chunking, reranker, prompt, model):

```powershell
python scripts/check_regression.py final_k3 <new_label>
```

It compares two runs from `docs/eval/results.csv` and exits with code 1 if a metric drops more than its tolerance
(0.02 for retrieval metrics, 0.05 for DeepEval metrics; contextual relevancy only warns because it is very noisy).
It is not part of the main CI because a full eval run takes a long time.

---

## Performance

### Latency over the 30 golden questions

| Run | Model | p50 | p95 | mean |
|---|---|---|---|---|
| Baseline (unpaced, before any optimization) | gpt-oss-120b | 21.13 s | 24.18 s | 20.03 s |
| After changes, cold cache (paced) | gpt-oss-120b | 4.50 s | 5.75 s | 4.62 s |
| After changes, cold cache (paced) | qwen3 (dev model) | 3.67 s | 4.78 s | 3.82 s |
| Warm cache (all answers cached) | qwen3 (dev model) | 0.00 s | 0.00 s | 0.00 s |

The baseline was measured while the Groq free tier was throttling, so part of the 21 s is rate-limit waiting and not pure pipeline time.
The pipeline itself needs about 4 to 5 s per question when it is not throttled.

### What actually moved the numbers

| Change | Effect |
|---|---|
| Measured stage by stage instead of guessing | Reranking was 78 percent of the time (7.38 s average) |
| Reranker candidate pool 30 -> 15 | Rerank time 7.38 s -> 2.79 s, retrieval metrics unchanged (hit 1.0, MRR 0.889) |
| LLM context 5 -> 3 chunks | Prompt 9 to 11K characters -> 5.4 to 7K, fewer Groq rate-limit hits, hit@3 stays 1.0 |
| Citation normalization (`【1】` -> `[1]`) | About half of the answers had been uncached because their fullwidth citations were not recognized. Fixed, and covered by a unit test |
| Exact-match answer cache | Repeat questions answered in about 0 s |

PII masking with Presidio was added after these measurements, and latency has not been re-measured since.

---

## Testing and CI

- **Unit tests** (`tests/unit`): input guard, small-talk routing, rewrite, prompts security, PII, output checks, citation normalization,
  `RagService` wiring with fakes, agent tools and the agent graph with a fake LLM.
- **Integration tests** (`tests/integration`): RBAC (28 tests), rate limiting, answer cache, sessions and feedback, agent API.
  These use real Postgres and Redis with a fake `RagService`, so they are deterministic.
- **Red-team tests** (`tests/redteam`): data-driven attacks, no external services.
- **CI** (`.github/workflows/ci.yml`): Python 3.12 on Ubuntu, `pip install -r requirements.txt`, then `pytest tests/unit tests/redteam`
  with a dummy `RAG_QDRANT_URL`. More than 100 tests pass, and the 2 known red-team gaps are reported as `xfail`.
  Integration tests and the full evaluation need Docker services or a slow judge and are run locally.

```powershell
python -m pytest tests/unit tests/redteam -q
python -m pytest tests/integration -q       # needs: docker compose up -d redis postgres
```

---

## Design decisions and lessons learned

- **Measure before optimizing.** The assumption "the LLM is slow" was wrong. A stage breakdown showed the reranker was the bottleneck,
  and a probe showed the rest was Groq free-tier rate limiting.
- **Keep guardrails deterministic.** Regex and plain Python can be tested in CI. Known gaps are written down as `xfail` tests instead of hidden.
- **Security belongs inside the data path.** The RBAC filter is part of the vector search and of the cache key, not a post-filter.
- **Fake the LLM in tests, use real services where it is cheap.** Integration tests run against real Postgres and Redis.
- **Be honest about negative results.** The agent did not beat plain RAG on this data, and the report says so, including the failure that the groundedness check cannot catch.
- **Do not over-engineer.** pip and `requirements.txt` instead of a packaging setup, small functions instead of class hierarchies,
  and skipped features (calculation routing, query embedding cache) were skipped on purpose after looking at the data.
- **Model for development vs final numbers.** A different Groq model was used during development to avoid free-tier 429 errors.
  All final latency and evaluation numbers state which model produced them.

---

## What is not done

- **Steps W to Z** of the blueprint, including observability (tracing with Langfuse and LangSmith-style traces) and experiment tracking (MLflow).
- **Frontend**, including CORS and a streaming endpoint.
- **Running conversation summary** (only the last 6 messages are used as history).
- **Multi-hop retrieval** for questions that need facts from two documents.
- **A larger golden set** with unanswerable, adversarial, table and multi-document questions, and a refusal-accuracy metric.
- **Paraphrased and Roman Urdu jailbreak detection** (tracked as `xfail` red-team tests).
- **Latency re-measurement** after adding Presidio.
- **Verifying the cause of low contextual relevancy** (large page-level chunks are the suspected reason).