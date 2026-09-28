"""
Per-user rate limiting backed by Redis (Step R).
"""

from fastapi import Request
from jose import JWTError
from slowapi import Limiter
from slowapi.util import get_remote_address

from .config import settings
from .security import decode_access_token


def user_key(request: Request) -> str:
    """Rate-limit key: the user id from the JWT, else the client IP."""
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        try:
            return decode_access_token(auth[7:])["sub"]
        except (JWTError, KeyError):
            pass
    return get_remote_address(request)


limiter = Limiter(key_func=user_key, storage_uri=settings.redis_url)