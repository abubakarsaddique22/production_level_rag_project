"""Plain RAG vs Agent comparison.

Usage (from project root):
    python scripts/compare_agent.py [pause_seconds] [question_ids ...]

Examples:
    python scripts/compare_agent.py 12          # full run, saves report + answers
    python scripts/compare_agent.py 12 c002     # only c002, prints full answers
"""
import sys
from pathlib import Path

# Make "src/" importable regardless of the current working directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import json
import re
import time
from datetime import date

from nexora_rag.agents.graph import agent
from nexora_rag.api.deps import ROLE_DEPARTMENTS
from nexora_rag.generation.rag_service import RagService

QUESTIONS_PATH = Path("data/eval/agent_compare.json")
RESULTS_PATH = Path("docs/agent_compare_results.md")
ANSWERS_PATH = Path("data/eval/agent_compare_answers.json")


def answer_text(result) -> str:
    """RagService.answer() may return a str, a dict or an object - handle all three."""
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        return result.get("answer", "")
    return getattr(result, "answer", "")


def is_correct(answer: str, expected: str) -> bool:
    """True if the expected number appears in the answer (commas ignored)."""
    numbers = re.findall(r"\d+(?:\.\d+)?", answer.replace(",", ""))
    return any(float(n) == float(expected) for n in numbers)


def status(answer: str, expected: str) -> str:
    """OK / FAIL / ERROR. ERROR = the call crashed (e.g. Groq 429), not a wrong answer."""
    if answer.startswith("ERROR:"):
        return "ERROR"
    return "OK" if is_correct(answer, expected) else "FAIL"


def run_plain(service, question, departments):
    start = time.perf_counter()
    try:
        text = answer_text(service.answer(question, departments=departments))
    except Exception as exc:
        text = f"ERROR: {exc}"
    return text, time.perf_counter() - start


def run_agent(service, question, departments):
    start = time.perf_counter()
    try:
        result = agent.invoke(
            {
                "question": question,
                "user_id": None,
                "departments": departments,
                "history": [],
                "service": service,
                "retries": 0,
                "tool_calls": 0,
            }
        )
        text = result.get("answer", "")
        route = result.get("route", "?")
    except Exception as exc:
        text, route = f"ERROR: {exc}", "?"
    return text, route, time.perf_counter() - start


def main() -> None:
    pause = float(sys.argv[1]) if len(sys.argv) > 1 else 12.0
    ids = sys.argv[2:]  # optional: only run these question ids, e.g. c002
    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    if ids:
        questions = [q for q in questions if q["id"] in ids]
    departments = ROLE_DEPARTMENTS["admin"]  # all departments

    service = RagService()
    service.answer("warm up", departments=departments)  # not counted: loads models

    rows = []
    answers = []
    for i, q in enumerate(questions, start=1):
        time.sleep(pause)
        plain_text, plain_time = run_plain(service, q["query"], departments)
        time.sleep(pause)
        agent_text, route, agent_time = run_agent(service, q["query"], departments)

        plain_ok = status(plain_text, q["expected"])
        agent_ok = status(agent_text, q["expected"])
        rows.append((q, plain_ok, agent_ok, plain_time, agent_time, route))
        answers.append(
            {"id": q["id"], "expected": q["expected"], "plain": plain_text, "agent": agent_text}
        )
        print(
            f"[{i}/{len(questions)}] {q['id']} ({q['type']}) "
            f"plain={plain_ok} ({plain_time:.1f}s)  "
            f"agent={agent_ok} ({agent_time:.1f}s, route={route})"
        )
        if ids:  # single-question debug run: show the full answers
            print(f"\nEXPECTED: {q['expected']}")
            print(f"\n--- PLAIN ---\n{plain_text}")
            print(f"\n--- AGENT ---\n{agent_text}\n")

    # ---- summary per type ----
    lines = [
        f"## Agent vs plain RAG ({date.today()})\n",
        "| id | type | expected | plain | agent | route | plain (s) | agent (s) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for q, p_st, a_st, p_t, a_t, route in rows:
        lines.append(
            f"| {q['id']} | {q['type']} | {q['expected']} | "
            f"{p_st} | {a_st} | {route} | "
            f"{p_t:.1f} | {a_t:.1f} |"
        )

    lines += ["", "| type | n | plain correct | agent correct | errors (plain/agent) |", "|---|---|---|---|---|"]
    for kind in ("calc", "lookup"):
        subset = [r for r in rows if r[0]["type"] == kind]
        if not subset:
            continue
        n = len(subset)
        p = sum(r[1] == "OK" for r in subset)
        a = sum(r[2] == "OK" for r in subset)
        p_err = sum(r[1] == "ERROR" for r in subset)
        a_err = sum(r[2] == "ERROR" for r in subset)
        lines.append(f"| {kind} | {n} | {p}/{n} | {a}/{n} | {p_err}/{a_err} |")

    report = "\n".join(lines) + "\n"
    print("\n" + report)

    if ids:  # partial run: do not overwrite the full-run files
        return

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(report, encoding="utf-8")
    ANSWERS_PATH.write_text(json.dumps(answers, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved to {RESULTS_PATH} and {ANSWERS_PATH}")


if __name__ == "__main__":
    main()