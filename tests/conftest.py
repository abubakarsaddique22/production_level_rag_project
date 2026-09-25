"""
Shared pytest fixtures for unit/integration/e2e tests.

STATUS: placeholder — grows as each step adds testable components.
"""

import pytest


@pytest.fixture
def anyio_backend():
    return "asyncio"

# TODO: add fixtures for test Qdrant collection, test DB session, mocked LLM client
