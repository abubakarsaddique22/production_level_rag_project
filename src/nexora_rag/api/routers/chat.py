"""
POST /v1/chat -- ask a question, get a grounded, cited answer (Step P).
"""

from fastapi import APIRouter, Depends

from ...generation.rag_service import RagService
from ..deps import get_rag_service, current_user, CurrentUser
from ..schemas import ChatRequest, ChatResponse

router = APIRouter(prefix="/v1", tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
def chat(
    req: ChatRequest,
    user: CurrentUser = Depends(current_user),
    rag_service: RagService = Depends(get_rag_service),
) -> ChatResponse:
    # RBAC: mandatory department filter, enforced here — never only in the prompt
    result = rag_service.answer(
        question=req.question,
        departments=user.departments,
        user_id=user.id,
    )
    return ChatResponse(**result)