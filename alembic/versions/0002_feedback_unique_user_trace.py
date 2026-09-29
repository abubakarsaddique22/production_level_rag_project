"""feedback unique user trace

Revision ID: 0002
Revises: 0001
Create Date: 2026-09-29
"""
from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_unique_constraint(
        "uq_feedback_user_trace", "feedback", ["user_id", "trace_id"]
    )


def downgrade() -> None:
    op.drop_constraint("uq_feedback_user_trace", "feedback", type_="unique")