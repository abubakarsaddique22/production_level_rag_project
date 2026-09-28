"""
Exact-match answer cache backed by Redis (Step R).

The cache key includes the user's allowed departments, so an answer
produced for one role is never served to a role with different access.
"""

import hashlib
import json

import redis

from .config import settings
from .logging import get_logger

log = get_logger(__name__)

_client = redis.Redis.from_url(
    settings.redis_url,
    decode_responses=True,
    socket_connect_timeout=0.5,
    socket_timeout=0.5,
)


def make_key(question: str, departments: list[str]) -> str:
    normalized = " ".join(question.lower().split())
    raw = json.dumps(
        [normalized, sorted(departments), settings.groq_model], ensure_ascii=False
    )
    return "answer:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()


def get_cached_answer(key: str) -> dict | None:
    try:
        value = _client.get(key)
    except redis.RedisError as exc:
        log.warning("answer_cache_get_failed", extra={"error": str(exc)})
        return None
    return json.loads(value) if value else None


def set_cached_answer(key: str, answer: str, sources: list[dict]) -> None:
    try:
        _client.set(
            key,
            json.dumps({"answer": answer, "sources": sources}, ensure_ascii=False),
            ex=settings.answer_cache_ttl_seconds,
        )
    except redis.RedisError as exc:
        log.warning("answer_cache_set_failed", extra={"error": str(exc)})