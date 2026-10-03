# LinkedIn post (draft)

Post the demo video or GIF as the main media. Put the GitHub link in the first comment (LinkedIn reduces reach of posts with links in the body).

---

I just finished my final year project: a RAG assistant that answers company questions with citations, and only from the documents your role is allowed to see.

The part I'm proudest of is not the chatbot. It's that I measured everything.

On a 30-question golden set:
- Dense search only: hit@3 0.90
- Hybrid search (BM25 + RRF): 0.93
- Hybrid + cross-encoder reranker: 1.00 (MRR 0.794 to 0.889)

When it felt slow I didn't guess. A stage-by-stage breakdown showed the reranker was 78% of the time. Cutting its candidate pool from 30 to 15 took it from 7.4 s to 2.8 s with no quality loss.

Other things I built:
- Access control inside the vector search, so the model never sees chunks a user can't read
- PII masking, prompt-injection defences and a red-team test suite that runs in CI
- CI/CD to AWS: GitHub Actions, ECR, deploy through SSM (no open SSH port), automatic rollback
- Tracing with LangSmith and monitoring with Prometheus + Grafana

What didn't work: my LangGraph agent scored 16/20, exactly the same as plain RAG. I published that result instead of hiding it.

What's next: my contextual relevancy score is only 0.12. I think my chunks are too big, so my next step is a chunk-size experiment.

Live demo, code, architecture decisions and the full evaluation report are in the first comment.

What would you improve first?

#RAG #LLM #AWS #Python #FastAPI #MachineLearning #FinalYearProject

---

Edit before posting: add your demo link, and keep only claims that are still true when you post.
