from src.nexora_rag.evaluation.dataset import load_golden_set
from src.nexora_rag.evaluation.judge import get_judge
from src.nexora_rag.generation.rag_service import RagService
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase

golden_set = load_golden_set()
item = next(i for i in golden_set if i.id == "q015")

service = RagService()
response = service.answer(item.query)
chunks = service.retriever.search(item.query, top_k=5)

test_case = LLMTestCase(
    input=item.query,
    actual_output=response["answer"],
    expected_output=item.ground_truth_answer,
    retrieval_context=[c["content"] for c in chunks],
)

metric = AnswerRelevancyMetric(threshold=0.85, model=get_judge(), include_reason=True)
metric.measure(test_case)

print("Query:", item.query)
print("Actual answer:", response["answer"])
print("Ground truth:", item.ground_truth_answer)
print("Score:", metric.score)
print("Reason:", metric.reason)