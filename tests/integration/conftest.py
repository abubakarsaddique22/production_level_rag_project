"""Integration tests need real Postgres and Redis. Skip (instead of failing with a
confusing connection error) when they are not running:  docker compose up -d redis postgres
"""

import socket
from urllib.parse import urlparse

import pytest

from nexora_rag.core.config import settings


def _reachable(url: str) -> bool:
    parsed = urlparse(url)
    try:
        with socket.create_connection((parsed.hostname, parsed.port), timeout=1):
            return True
    except (OSError, TypeError):
        return False


@pytest.fixture(scope="session", autouse=True)
def _require_services():
    missing = [
        name
        for name, url in (("Postgres", settings.database_url), ("Redis", settings.redis_url))
        if not _reachable(url)
    ]
    if missing:
        pytest.skip(f"{' and '.join(missing)} not reachable (docker compose up -d redis postgres)")