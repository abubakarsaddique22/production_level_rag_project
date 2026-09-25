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
