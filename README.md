# Nexora Knowledge Assistant

Production-style RAG system over internal PDFs (HR, Engineering, Finance, Product),
built step-by-step (Steps A–Z). This README will fill up with the architecture
diagram, results table and quick-start as we build.

## Quick start

```bash
# 1. clone / open in VS Code
# 2. create venv + install deps
make setup          # or: uv pip install -r requirements.txt

# 3. copy env template and fill in secrets
cp .env.example .env

# 4. start Qdrant / Postgres / Redis
make up

# 5. ingest PDFs (once ingestion is built — Step I)
make ingest

# 6. run the API (once built — Step P)
make run
```

## Project status

Currently at: **Step A — Problem definition & success criteria**.

See `docs/PRD.md` for targets and non-goals, and the project blueprint for the
full A–Z roadmap.

## Structure

See `src/nexora_rag/` for the package layout — each subfolder maps to a
build step (ingestion, retrieval, generation, agents, guardrails, evaluation,
observability, api).




## Evaluation

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

### Limitations

- **Contextual relevancy is low (0.121).** Chunks are large page-level text, so they contain much content unrelated to the question. This explanation has not been verified yet.
- **Judge noise.** DeepEval scores moved by about 0.02 to 0.03 between two runs, so small differences are not real improvements. For the same reason I did not rerun a separate top_k = 5 baseline.
- **Small golden set.** 30 questions, all answerable. There are no unanswerable, adversarial or multi-document questions, so refusal accuracy is not measured.
- The agent (Step U) is evaluated separately, see the agent section.

### Reproduce

```bash
python scripts/run_eval.py          # full eval, slow (DeepEval judge on Ollama), logs to docs/eval/results.csv
python scripts/make_report.py       # rebuild docs/eval/report.md and charts from the saved results
```

Optional regression check after changing the pipeline (chunking, reranker, prompt, model):

```bash
python scripts/check_regression.py final_k3 <new_label>
```

It compares two runs from `docs/eval/results.csv` and exits with code 1 if a metric drops more than its tolerance
(0.02 for retrieval metrics, 0.05 for DeepEval metrics; contextual relevancy only warns because it is very noisy).
It is not part of the main CI because a full eval run takes a long time.