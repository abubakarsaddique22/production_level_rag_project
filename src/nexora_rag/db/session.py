"""
Async SQLAlchemy engine/session factory.

STATUS: placeholder — implemented in Step S.
"""

# TODO(Step S): implement this module
"""
Database session management (Step Q).

Provides an async SQLAlchemy engine and a FastAPI dependency
that yields a session per request.
"""

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from ..core.config import settings

engine = create_async_engine(settings.database_url, echo=False)

async_session_factory = async_sessionmaker(
    engine, expire_on_commit=False, class_=AsyncSession
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with async_session_factory() as session:
        yield session