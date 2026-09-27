from src.nexora_rag.retrieval.reranker import RerankingRetriever
from src.nexora_rag.generation.prompts import SYSTEM_PROMPT, build_user_message
from src.nexora_rag.generation.llm import ask_llm
from src.nexora_rag.generation.citations import validate_citations, build_sources

retriever = RerankingRetriever()
question = 'How many days of paid maternity leave are there?'
chunks = retriever.search(question, top_k=5)

user_message = build_user_message(question, chunks)
answer = ask_llm(SYSTEM_PROMPT, user_message)

result = validate_citations(answer, chunks)
print('Answer:', result['clean_answer'])
print('Sources:', build_sources(chunks, result['valid']))
