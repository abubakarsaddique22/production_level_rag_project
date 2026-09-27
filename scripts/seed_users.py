"""
Seed the database with one test user per role (Step Q).

Run with:
    python scripts/seed_users.py

Creates: employee, engineer, finance, hr and admin test accounts,
all with the same password, so you can manually test /v1/auth/login
and RBAC on /v1/chat.
"""

import sys
from pathlib import Path

# Make "src/" importable regardless of the current working directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import asyncio

from sqlalchemy import select

from nexora_rag.core.security import hash_password
from nexora_rag.db.models import User
from nexora_rag.db.session import async_session_factory

TEST_PASSWORD = "Test@1234"

TEST_USERS = [
    {"email": "employee@nexora.test", "role": "employee", "full_name": "Test Employee"},
    {"email": "engineer@nexora.test", "role": "engineer", "full_name": "Test Engineer"},
    {"email": "finance@nexora.test", "role": "finance", "full_name": "Test Finance"},
    {"email": "hr@nexora.test", "role": "hr", "full_name": "Test HR"},
    {"email": "admin@nexora.test", "role": "admin", "full_name": "Test Admin"},
]


async def seed() -> None:
    async with async_session_factory() as session:
        for u in TEST_USERS:
            existing = await session.execute(
                select(User).where(User.email == u["email"])
            )
            if existing.scalar_one_or_none() is not None:
                print(f"Skipping {u['email']} (already exists)")
                continue

            user = User(
                email=u["email"],
                hashed_password=hash_password(TEST_PASSWORD),
                full_name=u["full_name"],
                role=u["role"],
            )
            session.add(user)
            print(f"Created {u['email']} (role={u['role']})")

        await session.commit()


if __name__ == "__main__":
    asyncio.run(seed())