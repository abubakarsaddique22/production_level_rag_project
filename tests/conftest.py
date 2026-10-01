import os
import sys
from pathlib import Path

import pytest

# Step W added LangSmith tracing. If a developer's .env has LANGSMITH_TRACING=true,
# every unit test that calls RagService.answer() would send fake traces to the real
# LangSmith project. Switch tracing off BEFORE any app module is imported
# (tracing.py calls load_dotenv(), which never overrides variables that already exist).
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"

# qdrant_url is the only required setting; unit tests never connect to it.
os.environ.setdefault("RAG_QDRANT_URL", "http://localhost:6333")

# Make "src/" importable in tests (no pyproject.toml, so no editable install).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


@pytest.fixture
def anyio_backend():
    return "asyncio"