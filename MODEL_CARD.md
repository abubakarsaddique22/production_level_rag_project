# Model card: Nexora Knowledge Assistant

A RAG system, not a single model. This card lists every model in the pipeline, what the system is for, how it was measured and where it fails.

## Models used

| Role | Model | Where it runs | Notes |
|---|---|---|---|
| Embeddings | `BAAI/bge-small-en-v1.5` | In the API container (CPU) | Same model at ingestion and query time |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | In the API container (CPU) | Reranks a pool of 15, keeps top 3 |
| Answer LLM | `openai/gpt-oss-120b` via Groq | Groq API (free tier) | Used for all final numbers and set in the production env. The default in `core/config.py` is `gpt-oss-20b`, so the env value is what counts. |
| Query rewrite | Same Groq model | Groq API | Falls back to the original question if rewrite fails |
| PII detection | Presidio + spaCy `en_core_web_sm` | In the API container | Plus custom `PK_CNIC` and `PK_PHONE` recognizers |
| Evaluation judge | Self-hosted model on Ollama | Developer machine | Used only for DeepEval, never in production |

## Intended use

- Internal question answering over a fixed set of company PDFs (HR, Engineering, Finance, Product).
- Users are employees with a role. Each answer must have citations and respect the role's departments.

## Out of scope

- Legal, medical or financial advice.
- Questions that need combining facts from several documents (measured failure, see below).
- Documents outside the 10 indexed PDFs. The system should refuse or return no sources, but refusal accuracy is **not measured**.
- Languages other than English for retrieval (small-talk routing also handles Roman Urdu greetings).

## Data

- **Knowledge base:** 10 PDFs in `data/raw` (API documentation, benefits, coding standards, deployment SOP, employee handbook, expense policy, leave policy, procurement policy, product documentation, product FAQ). Chunk files are committed.
- **Golden set:** 30 hand-written questions with relevant chunk ids (`data/eval/golden_dataset.json`). All are answerable.
- **Agent comparison set:** 20 questions (18 calculation, 2 lookup).
- **User data:** chat sessions, messages and feedback in Postgres. PII in answers and snippets is masked before caching.

## Evaluation

Final run `final_k3`, 30 questions, top_k = 3 ([full report](docs/eval/report.md)).

| Metric | Value |
|---|---|
| Retrieval hit@3 / MRR / recall@3 | 1.000 / 0.889 / 0.950 |
| Faithfulness | 0.989 |
| Answer relevancy | 0.933 |
| Contextual precision / recall | 0.928 / 0.967 |
| Contextual relevancy | 0.121 |
| Agent vs plain RAG (20 questions) | 16/20 vs 16/20 |
| Cross-role leaks in RBAC tests | 0 (28 integration tests, run locally) |

## Limitations and known failures

- Contextual relevancy is low. Cause suspected (large chunks), not verified.
- 4 of 4 multi-document calculation questions fail, for plain RAG and for the agent.
- The agent once produced a confident wrong number from a correctly computed but wrongly retrieved fact. The groundedness check cannot catch that.
- Judge noise of about 0.02 to 0.03 between DeepEval runs. Differences smaller than that are not real.
- Regex guardrails miss paraphrased and Roman Urdu jailbreaks (2 `xfail` red-team tests).
- Person names are intentionally **not** masked, because colleague and customer names are part of normal answers.
- The LLM runs on the Groq free tier, which has rate limits (8,000 tokens per minute was the observed limit during tests).

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Prompt injection through documents | Chunks wrapped as untrusted `<document>` data, `neutralize()` strips fake tags |
| Cross-role data leak | RBAC inside search, department-aware cache key, safety net in the retriever |
| System prompt leak | Output leak markers, leaky answers replaced and never cached |
| Made-up answers | Citation validation: no valid citation means no sources and no cache |
| Runaway cost or abuse | Per-user rate limits, input length limit, agent tool and retry caps, timeout |

## Contact and maintenance

Maintained by the repository owner (GitHub: abubakarsaddique22). Re-run `scripts/run_eval.py` and `scripts/check_regression.py` after any change to chunking, reranker, prompt or model.
