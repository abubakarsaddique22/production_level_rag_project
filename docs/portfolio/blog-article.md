# I measured my RAG system before I tried to improve it. Here is what I found.

*Draft blog article, about 6 minutes. Publish on Medium, dev.to or your own site.*

Most RAG tutorials end when the chatbot answers a question. I wanted to build the version a company could actually run:
answers with citations, access control per role, tests, a deployment, and numbers I could defend in an interview.
This is what I learned building Nexora, a knowledge assistant over 10 company PDFs.

## 1. Start with a golden set, not a model

Before tuning anything I wrote 30 questions by hand, each with the exact chunk that answers it.
That gave me retrieval metrics (hit@k, MRR, recall@k) that need **no LLM**, so they are deterministic and free to re-run.
The set is small and every question is answerable, which I state as a limitation. But it turned arguments into numbers.

## 2. Hybrid search, then a reranker

| Retriever | hit@3 | MRR |
|---|---|---|
| Dense only | 0.900 | 0.794 |
| Dense + BM25 with RRF | 0.933 | 0.850 |
| + cross-encoder rerank | 1.000 | 0.889 |

Dense search failed on three questions. BM25 fixed one, where an exact term mattered.
The cross-encoder fixed the other two, including a question whose answer ("submission days") was buried in a chunk about per diem.
A cross-encoder reads the query and chunk together, so it catches that. It costs more, so it only sees 15 candidates.

I fused the two rankings with Reciprocal Rank Fusion because cosine scores and BM25 scores are on different scales.
RRF only uses rank positions, so there is nothing to tune.

## 3. "The LLM is slow" was wrong

The first end-to-end p50 was about 21 seconds. I assumed the LLM. A stage-by-stage breakdown said otherwise:
reranking was **78% of the time (7.38 s)**. Shrinking the candidate pool from 30 to 15 and the LLM context from 5 chunks to 3 gave a rerank of 2.79 s with **unchanged retrieval metrics**.

One honest caveat: that 21 s baseline was measured while the Groq free tier was throttling me, so part of it was rate-limit waiting.
The pipeline itself needs about 4 to 5 seconds per question. I wrote that in the report rather than claiming a 5x speedup.

Two small things also mattered. An exact-match Redis cache makes repeat questions instant, and fixing a citation bug
(some models output full-width brackets, 【1】, instead of [1]) rescued about half of the answers that were not being cached.

## 4. Security that lives in the data path

Telling the model "do not reveal Finance documents" is not access control. The department filter runs **inside** the vector search,
so forbidden chunks are never retrieved. The cache key includes the departments, so a cached Finance answer cannot be served to an employee.
Retrieved chunks are wrapped as untrusted data in the prompt, PII is masked, and a red-team test suite runs in CI.
Two attacks (paraphrased and Roman Urdu jailbreaks) still get through the regex layer. They are kept as `xfail` tests so the gap is visible.

## 5. The agent did not win

I built a LangGraph agent with a router, a safe calculator, a ticket lookup and a groundedness check. Then I compared it with plain RAG on 20 questions.
Both scored 16/20. Both failed the same four multi-document questions.
Worse, on a held-out question the agent returned a confident wrong number: the arithmetic was right, but it applied a percentage to the wrong salary figure.
A groundedness check cannot catch that. The fix is multi-hop retrieval, not a smarter agent. I published this result.

## 6. Shipping to AWS

Pushing a git tag builds the image, pushes it to ECR and deploys on one EC2 instance through AWS Systems Manager, so SSH stays closed.
Secrets travel from GitHub secrets to SSM Parameter Store to the server, never into the image. If the new version fails its health check, the script rolls back automatically.

## What I would do next

My DeepEval contextual relevancy is only 0.121. I suspect my 2,400-character chunks, but I have not tested it.
The next step is a chunk-size experiment at 300, 600 and 1000 tokens, then a bigger golden set with unanswerable questions.

**Code, ADRs, model card and the full evaluation report:** `<GITHUB-LINK>`
