"""
Dependency injection for the API (Step P, extended in Step Q).

ONE RagService instance is created and reused across every request --
building it fresh per request would reload the embedding model, the
reranker model, and reconnect to Qdrant every single time.
"""

from dataclasses import dataclass
from functools import lru_cache

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError

from ..core.security import decode_access_token
from ..generation.rag_service import RagService


@lru_cache
def get_rag_service() -> RagService:
    return RagService()


# ============================================================
# Auth (Step Q)
# ============================================================

ROLE_DEPARTMENTS: dict[str, list[str]] = {
    "employee": ["HR", "Product"],
    "engineer": ["Engineering", "HR", "Product"],
    "finance": ["HR", "Finance", "Product"],
    "hr": ["HR", "Product"],
    "admin": ["Engineering", "HR", "Finance", "Product"],
}

bearer_scheme = HTTPBearer()


@dataclass
class CurrentUser:
    id: str
    role: str
    departments: list[str]


def current_user(
    creds: HTTPAuthorizationCredentials = Depends(bearer_scheme),
) -> CurrentUser:
    try:
        payload = decode_access_token(creds.credentials)
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        )

    user_id = payload.get("sub")
    role = payload.get("role")

    if user_id is None or role not in ROLE_DEPARTMENTS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload",
        )

    return CurrentUser(id=user_id, role=role, departments=ROLE_DEPARTMENTS[role])