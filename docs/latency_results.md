<!-- | Run | Date | N | p50 (s) | p95 (s) | Mean (s) |
|---|---|---|---|---|---|
| baseline | 2026-09-28 | 30 | 21.13 | 24.18 | 20.03 |
| after_cold_use_gpt | 2026-09-28 | 30 | 4.50 | 5.75 | 4.62 |
| after_warm_use_gpt | 2026-09-28 | 30 | 12.34 | 16.07 | 8.84 |
| after_cold_fixed_use_qwen | 2026-09-28 | 30 | 3.67 | 4.78 | 3.82 |
| after_warm_fixed_use_qwen | 2026-09-28 | 30 | 0.00 | 0.00 | 0.00 | -->

# Latency results (Step R)

30 golden-set questions, admin role (all departments), end-to-end
`RagService.answer()` (retrieval, rerank, LLM), measured with
`scripts/measure_latency.py`.

| Run | Model | Setup | p50 (s) | p95 (s) | Mean (s) |
|---|---|---|---|---|---|
| baseline | gpt-oss-120b | rerank pool 30, 5 chunks, no pause | 21.13 | 24.18 | 20.03 |
| after_cold | gpt-oss-120b | rerank pool 15, 3 chunks, 12 s pause | 4.50 | 5.75 | 4.62 |
| after_cold_fixed | qwen3.8-27b | as above + citation fix | 3.67 | 4.78 | 3.82 |
| after_warm | qwen3.8-27b | same questions again, all cache hits | <0.01 | <0.01 | <0.01 |

## What changed
1. Rerank candidate pool 30 -> 15 (rerank 7.4 s -> 2.8 s; hit@5 and MRR unchanged).
2. LLM prompt 5 -> 3 chunks (hit@3 1.0, MRR 0.889, recall 0.983 -> 0.95).
3. Redis answer cache, key = question + allowed departments (RBAC-safe).
4. Citation fix: fullwidth 【1】 is converted to [1]; before it about half the answers had no sources.

## Notes
- The baseline was measured while the Groq free-tier limit (8,000 tokens/min)
  was throttling requests, so most of its ~20 s was waiting, not pipeline work.
- Later runs pause 12 s between questions to stay under that limit.
- The model differs between rows 2 and 3, so that comparison mixes two effects.
- The warm run only measures repeated questions (cache hits).
- Answer quality with 3 chunks and qwen is not measured yet (Step V).