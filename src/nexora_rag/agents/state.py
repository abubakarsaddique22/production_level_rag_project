from typing import TypedDict


class AgentState(TypedDict, total=False):
    question: str
    user_id: str | None
    departments: list[str]   # user ke role se aate hain
    history: list
    route: str               # "kb" | "calc" | "ticket"
    kb_answer: str
    kb_found: bool
    sources: list
    calc_result: str
    calc_value: float | None
    ticket_result: str
    ticket_found: bool
    answer: str
    grounded: bool
    retries: int             # ab tak kitne check fail hue (max 2 retries)
    tool_calls: int          # max 4