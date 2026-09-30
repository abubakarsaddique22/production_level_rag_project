"""Agent kb node: poora sawal na mile to calc route sawal ko tukron mein todta hai."""
import src.nexora_rag.agents.graph as agent_graph

NOT_FOUND = "I don't have enough information to answer that."
QUESTION = "A G3 employee resigns with 12 unused leave days after 3 years. Total settlement?"


class FakeService:
    """Tayyar jawab deta hai; jo sawal replies mein nahi wo 'not found' (koi source nahi)."""

    def __init__(self, replies):
        self.replies = replies
        self.calls = []

    def answer(self, question, departments=None, user_id=None, history=None):
        self.calls.append(question)
        answer, sources = self.replies.get(question, (NOT_FOUND, []))
        return {"answer": answer, "sources": sources, "trace_id": "t"}


def make_state(service, route="calc"):
    return {"question": QUESTION, "departments": ["HR"], "route": route, "service": service}


def fake_llm(monkeypatch, reply):
    monkeypatch.setattr(agent_graph, "ask_llm", lambda prompt, text: reply)


def llm_must_not_run(monkeypatch):
    def boom(prompt, text):
        raise AssertionError("ask_llm should not be called")

    monkeypatch.setattr(agent_graph, "ask_llm", boom)


# ---------- helpers ----------

def test_split_question_cleans_numbering_and_caps_at_three(monkeypatch):
    fake_llm(monkeypatch, "1. Leave rule?\n- Gratuity rule?\n\n3) Third?\nFourth?")
    assert agent_graph.split_question(QUESTION) == ["Leave rule?", "Gratuity rule?", "Third?"]


def test_split_question_returns_empty_when_llm_fails(monkeypatch):
    def boom(prompt, text):
        raise RuntimeError("429")

    monkeypatch.setattr(agent_graph, "ask_llm", boom)
    assert agent_graph.split_question(QUESTION) == []


def test_safe_converts_fullwidth_citations():
    assert agent_graph.safe("Leave is paid at basic/30【1】 and more【2】.") == "Leave is paid at basic/30[1] and more[2]."


def test_shift_citations():
    assert agent_graph.shift_citations("A [1] and B [2]", 2) == "A [3] and B [4]"
    assert agent_graph.shift_citations("no citations", 5) == "no citations"


# ---------- kb_node ----------

def test_kb_node_splits_when_full_question_finds_nothing(monkeypatch):
    service = FakeService({
        "Leave rule?": ("Leave is paid at basic/30 [1].", [{"doc_id": "NX-HR-001", "page": 4}]),
        "Gratuity rule?": ("One month basic per year [1].", [{"doc_id": "NX-HR-003", "page": 2}]),
    })
    fake_llm(monkeypatch, "Leave rule?\nGratuity rule?")

    out = agent_graph.kb_node(make_state(service))

    assert out["kb_found"] is True
    assert [s["doc_id"] for s in out["sources"]] == ["NX-HR-001", "NX-HR-003"]
    # second answer's [1] becomes [2], so citations still match the merged sources
    assert out["kb_answer"] == "Leave is paid at basic/30 [1]. One month basic per year [2]."
    assert out["tool_calls"] == 1  # ek logical KB tool call
    assert service.calls == [QUESTION, "Leave rule?", "Gratuity rule?"]


def test_kb_node_does_not_split_when_full_question_is_found(monkeypatch):
    service = FakeService({QUESTION: ("Full answer [1].", [{"doc_id": "NX-HR-001", "page": 4}])})
    llm_must_not_run(monkeypatch)

    out = agent_graph.kb_node(make_state(service))

    assert out["kb_found"] is True
    assert out["kb_answer"] == "Full answer [1]."
    assert service.calls == [QUESTION]


def test_kb_node_does_not_split_for_kb_route(monkeypatch):
    service = FakeService({})
    llm_must_not_run(monkeypatch)

    out = agent_graph.kb_node(make_state(service, route="kb"))

    assert out["kb_found"] is False
    assert service.calls == [QUESTION]


def test_kb_node_stays_not_found_when_split_gives_one_question(monkeypatch):
    service = FakeService({})
    fake_llm(monkeypatch, "Only one question?")

    out = agent_graph.kb_node(make_state(service))

    assert out["kb_found"] is False
    assert service.calls == [QUESTION]  # no sub-search for a single part


def test_kb_node_stays_not_found_when_no_part_is_found(monkeypatch):
    service = FakeService({})
    fake_llm(monkeypatch, "Part one?\nPart two?")

    out = agent_graph.kb_node(make_state(service))

    assert out["kb_found"] is False
    assert out["sources"] == []
    assert service.calls == [QUESTION, "Part one?", "Part two?"]


def test_kb_node_survives_a_crash_in_the_split_search(monkeypatch):
    class CrashingService(FakeService):
        def answer(self, question, departments=None, user_id=None, history=None):
            if question != QUESTION:
                raise RuntimeError("Groq 429")
            return super().answer(question, departments, user_id, history)

    fake_llm(monkeypatch, "Part one?\nPart two?")

    out = agent_graph.kb_node(make_state(CrashingService({})))

    assert out["kb_found"] is False  # normal refusal, no exception