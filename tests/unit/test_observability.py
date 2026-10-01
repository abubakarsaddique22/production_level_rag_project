import uuid

import pytest
from prometheus_client import REGISTRY
from starlette.testclient import TestClient

from nexora_rag.generation import rag_service
from nexora_rag.generation.rag_service import RagService
from nexora_rag.observability import metrics, tracing


def sample(name: str, labels: dict | None = None) -> float:
    return REGISTRY.get_sample_value(name, labels or {}) or 0.0


# ---------- tracing_enabled ----------

def test_tracing_is_off_without_env(monkeypatch):
    monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    assert tracing.tracing_enabled() is False


def test_tracing_needs_both_flag_and_key(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    assert tracing.tracing_enabled() is False

    monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2_test")
    assert tracing.tracing_enabled() is True

    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    assert tracing.tracing_enabled() is False


# ---------- trace id ----------

def test_no_current_trace_outside_a_traced_call():
    assert tracing.current_trace_id() is None


def test_trace_id_falls_back_to_a_uuid_when_untraced():
    value = rag_service._trace_id()
    assert str(uuid.UUID(value)) == value


def test_traceable_function_still_works_with_tracing_off():
    @tracing.traceable(name="demo")
    def add(a, b):
        return a + b

    assert add(1, 2) == 3


# ---------- send_feedback ----------

class RecordingClient:
    calls: list = []

    def create_feedback(self, **kwargs):
        RecordingClient.calls.append(kwargs)


class BrokenClient:
    def create_feedback(self, **kwargs):
        raise RuntimeError("LangSmith is down")


@pytest.fixture
def tracing_on(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2_test")
    RecordingClient.calls = []


def test_feedback_is_not_sent_when_tracing_is_off(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "false")

    def explode():
        raise AssertionError("Client must not be created")

    monkeypatch.setattr(tracing, "Client", explode)
    tracing.send_feedback("trace-1", 1.0, "ok")  # must simply return


def test_feedback_is_sent_with_the_right_values(tracing_on, monkeypatch):
    monkeypatch.setattr(tracing, "Client", RecordingClient)

    tracing.send_feedback("trace-1", 0.0, "wrong answer")

    assert RecordingClient.calls == [
        {"run_id": "trace-1", "key": "user_rating", "score": 0.0, "comment": "wrong answer"}
    ]


def test_feedback_errors_never_reach_the_caller(tracing_on, monkeypatch):
    monkeypatch.setattr(tracing, "Client", BrokenClient)
    tracing.send_feedback("trace-1", 1.0)  # must not raise


# ---------- Prometheus metrics ----------

def test_request_counter_and_latency_histogram():
    answered = sample("rag_requests_total", {"outcome": "answered"})
    count = sample("rag_latency_seconds_count")

    metrics.REQUESTS.labels(outcome="answered").inc()
    metrics.LATENCY.observe(1.5)

    assert sample("rag_requests_total", {"outcome": "answered"}) == answered + 1
    assert sample("rag_latency_seconds_count") == count + 1


def test_metrics_endpoint_serves_prometheus_text():
    metrics.CACHE_LOOKUPS.labels(result="hit").inc()
    body = TestClient(metrics.metrics_app).get("/").text
    assert "rag_requests_total" in body
    assert "rag_cache_lookups_total" in body


class EmptyRetriever:
    def search(self, *args, **kwargs):
        return []


class BoomRetriever:
    def search(self, *args, **kwargs):
        raise AssertionError("cache hit must not reach the retriever")


def test_cache_miss_is_counted(monkeypatch):
    monkeypatch.setattr(rag_service, "get_cached_answer", lambda key: None)
    before = sample("rag_cache_lookups_total", {"result": "miss"})

    RagService(retriever=EmptyRetriever()).answer("How many leave days?", departments=["HR"])

    assert sample("rag_cache_lookups_total", {"result": "miss"}) == before + 1


def test_cache_hit_is_counted(monkeypatch):
    cached = {"answer": "90 days [1]", "sources": [{"id": 1}]}
    monkeypatch.setattr(rag_service, "get_cached_answer", lambda key: cached)
    before = sample("rag_cache_lookups_total", {"result": "hit"})

    result = RagService(retriever=BoomRetriever()).answer("How many leave days?", departments=["HR"])

    assert result["answer"] == "90 days [1]"
    assert sample("rag_cache_lookups_total", {"result": "hit"}) == before + 1
