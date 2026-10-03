# ADR 0003: Chunking strategy (about 600 tokens, 80 overlap, split inside pages)

- **Status:** Accepted, with an open question (see Consequences)
- **Date:** 2026-09

## Context

Source documents are 10 PDFs (policies, handbook, API docs, SOPs, FAQ) with prose, tables and some image text (OCR).
Citations must point to a **document and page**, so a chunk must never span two pages.

## Decision

- Pages are cleaned first (`clean_pages.json`), then each page is split with LangChain's `RecursiveCharacterTextSplitter`.
- Settings (`core/config.py`): `chunk_tokens = 600`, `chunk_overlap = 80`.
  The splitter counts characters and uses **4 characters per token**, so the real size is about **2,400 characters** with **320 characters** of overlap.
- Separators in order: paragraph, line, sentence end (`. ? !`), space. The splitter tries to cut at the largest natural boundary first.
- Every chunk keeps `doc_id`, `title`, `department`, `version`, `page`, `chunk_index`, `chunk_type` (prose / table / image_ocr / mixed) and a per-chunk SHA-256 hash.
- `chunk_id = <doc_id>-p<page>-c<index>-<hash>`, so ids are stable and the Qdrant upsert is idempotent.

## Consequences

- (+) Page-bound chunks give exact page citations.
- (+) Chunk-level metadata supports the RBAC filter and traceability.
- (-) **Open question:** DeepEval contextual relevancy is only **0.121**. My suspected cause is that 2,400-character chunks carry a lot of text unrelated to the question.
  **No chunk-size experiment was run**, so 600 tokens is a reasonable default and not a measured optimum.
- (-) The "4 characters per token" rule is an approximation. It is not a real tokenizer count.

## Next step

Re-run the retrieval metrics and DeepEval with 300, 600 and 1000 tokens and record the result here.
Retrieval metrics are deterministic and cheap, so this is a one-evening experiment.
