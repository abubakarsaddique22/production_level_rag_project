import sys
from pathlib import Path

import pytest

# Make "src/" importable in tests (no pyproject.toml, so no editable install).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


@pytest.fixture
def anyio_backend():
    return "asyncio"