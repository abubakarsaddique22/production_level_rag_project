"""
TEMPORARY helper to create database tables directly from the models,
without a migration tool.

This is a stand-in until Step S sets up proper Alembic migrations.
Do not use this approach in production -- schema changes won't be
tracked or reversible.

Run with:
    python scripts/create_tables.py
"""

import sys
from pathlib import Path

# Make "src/" importable regardless of the current working directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import asyncio

from nexora_rag.db.models import Base
from nexora_rag.db.session import engine


async def create_tables() -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print("Tables created.")


if __name__ == "__main__":
    asyncio.run(create_tables())