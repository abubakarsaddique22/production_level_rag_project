"""
Typed tools: calculator, KB search wrapper, mock ticket lookup.

STATUS: placeholder — implemented in Step U.
"""

import ast
import operator

from pydantic import BaseModel, Field

# ---------- Calculator ----------

_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.USub: operator.neg,
}


def _eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    raise ValueError("unsupported expression")


class CalcInput(BaseModel):
    expression: str = Field(min_length=1, max_length=100)


class CalcOutput(BaseModel):
    expression: str
    result: float


def calculator(expression: str) -> CalcOutput:
    """Safe arithmetic: + - * / and brackets only. eval() kabhi nahi."""
    data = CalcInput(expression=expression)
    cleaned = data.expression.replace(",", "")  # "30,000 * 5" bhi chale
    tree = ast.parse(cleaned, mode="eval")
    return CalcOutput(expression=cleaned, result=float(_eval(tree.body)))


# ---------- Mock ticket lookup ----------

_TICKETS = {
    "TCK-101": {"status": "open", "title": "VPN access request"},
    "TCK-102": {"status": "closed", "title": "Laptop replacement"},
    "TCK-103": {"status": "in_progress", "title": "Expense reimbursement query"},
}


class TicketOutput(BaseModel):
    ticket_id: str
    found: bool
    status: str | None = None
    title: str | None = None


def lookup_ticket(ticket_id: str) -> TicketOutput:
    ticket = _TICKETS.get(ticket_id.strip().upper())
    if ticket is None:
        return TicketOutput(ticket_id=ticket_id, found=False)
    return TicketOutput(ticket_id=ticket_id, found=True, **ticket)


# ---------- KB search (RagService ko wrap karta hai) ----------

class KBOutput(BaseModel):
    question: str
    found: bool
    answer: str
    sources: list[dict]
    trace_id: str


def kb_search(service, question: str, departments: list[str],
              user_id: str | None = None, history: list | None = None) -> KBOutput:
    """Knowledge base search. departments hamesha user ke role se aate hain, LLM se nahi."""
    result = service.answer(question, departments=departments, user_id=user_id, history=history)
    return KBOutput(
        question=question,
        found=bool(result["sources"]),
        answer=result["answer"],
        sources=result["sources"],
        trace_id=result["trace_id"],
    )
