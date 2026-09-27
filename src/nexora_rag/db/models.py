# """
# SQLAlchemy models: User, Session, Message, Feedback.

# STATUS: placeholder — implemented in Step S.
# """

# # TODO(Step S): implement this module


# from datetime import datetime, timezone

# from sqlalchemy import String, DateTime
# from sqlalchemy.orm import Mapped, mapped_column

# from nexora_rag.db.session import Base  # your existing declarative base


# class User(Base):
#     __tablename__ = "users"

#     id: Mapped[int] = mapped_column(primary_key=True)
#     email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
#     hashed_password: Mapped[str] = mapped_column(String(255))
#     roles: Mapped[str] = mapped_column(String(255))  # comma-separated: "employee,finance"
#     is_active: Mapped[bool] = mapped_column(default=True)
#     created_at: Mapped[datetime] = mapped_column(
#         DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
#     )

#     def role_list(self) -> list[str]:
#         return [r.strip() for r in self.roles.split(",") if r.strip()]


"""
Database models for the Nexora RAG project.

Step Q introduces the User table for authentication and
role-based access control.
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(
        String, primary_key=True, default=lambda: str(uuid.uuid4())
    )
    email: Mapped[str] = mapped_column(String, unique=True, index=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String, nullable=False)
    full_name: Mapped[str] = mapped_column(String, nullable=True)

    # employee | engineer | finance | hr | admin
    role: Mapped[str] = mapped_column(String, nullable=False, default="employee")

    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )