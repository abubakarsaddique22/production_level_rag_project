from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.concurrency import run_in_threadpool
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ...db.models import ChatSession, Feedback, Message
from ...db.session import get_db
from ...observability.tracing import send_feedback
from ..deps import CurrentUser, current_user
from ..schemas import FeedbackRequest, FeedbackResponse

router = APIRouter(prefix="/v1", tags=["feedback"])


@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(
    req: FeedbackRequest,
    user: CurrentUser = Depends(current_user),
    db: AsyncSession = Depends(get_db),
) -> FeedbackResponse:
    # Sirf apne session ka assistant message. Doosre ka trace_id aur
    # jo exist hi nahi karta, dono ek jaisa 404 dete hain.
    result = await db.execute(
        select(Message.id)
        .join(ChatSession, ChatSession.id == Message.session_id)
        .where(
            Message.trace_id == req.trace_id,
            Message.role == "assistant",
            ChatSession.user_id == user.id,
        )
    )
    if result.first() is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Answer not found"
        )

    # Pehle se feedback hai to update, warna naya
    result = await db.execute(
        select(Feedback).where(
            Feedback.user_id == user.id, Feedback.trace_id == req.trace_id
        )
    )
    feedback = result.scalar_one_or_none()
    if feedback is None:
        feedback = Feedback(
            user_id=user.id,
            trace_id=req.trace_id,
            rating=req.rating,
            comment=req.comment,
        )
        db.add(feedback)
    else:
        feedback.rating = req.rating
        feedback.comment = req.comment

    await db.commit()
    await run_in_threadpool(
       send_feedback, req.trace_id, 1.0 if req.rating == 1 else 0.0, req.comment
    )
    return FeedbackResponse(
        trace_id=req.trace_id, rating=req.rating, comment=req.comment
    )