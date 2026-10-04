"""FeedbackStorePort 的 PostgreSQL 实现。"""

from __future__ import annotations

import json
from datetime import datetime

from sqlalchemy import Boolean, Column, Index, MetaData, String, Table, Text

from knowresearch.adapters.postgres_base import create_postgres_engine
from knowresearch.core.ports import FeedbackStorePort
from knowresearch.core.schemas import UserFeedback


class PostgresFeedbackStore(FeedbackStorePort):
    TABLE_NAME = "feedback"

    def __init__(self, database_url: str):
        self.engine = create_postgres_engine(database_url)
        self.metadata = MetaData()
        self.table = Table(
            self.TABLE_NAME,
            self.metadata,
            Column("id", String, primary_key=True),
            Column("user_id", String, nullable=False),
            Column("trace_id", String, nullable=False, default=""),
            Column("target_type", String, nullable=False, default="answer"),
            Column("target_id", String, nullable=False, default=""),
            Column("rating", String, nullable=False),
            Column("correction", Text, nullable=False, default=""),
            Column("created_at", String, nullable=False),
            Column("processed", Boolean, nullable=False, default=False),
            Column("data", Text, nullable=False),
        )
        Index(
            f"idx_{self.TABLE_NAME}_user_target",
            self.table.c.user_id,
            self.table.c.target_type,
        )
        self.metadata.create_all(self.engine)

    def save(self, feedback: UserFeedback) -> None:
        values = {
            "id": feedback.id,
            "user_id": feedback.user_id,
            "trace_id": feedback.trace_id,
            "target_type": feedback.target_type,
            "target_id": feedback.target_id,
            "rating": feedback.rating,
            "correction": feedback.correction,
            "created_at": feedback.created_at.isoformat(),
            "processed": feedback.processed,
            "data": json.dumps(feedback.model_dump(mode="json"), ensure_ascii=False),
        }
        if self.engine.dialect.name == "postgresql":
            from sqlalchemy.dialects.postgresql import insert as dialect_insert
        else:
            from sqlalchemy.dialects.sqlite import insert as dialect_insert
        stmt = dialect_insert(self.table).values(**values)
        stmt = stmt.on_conflict_do_update(
            index_elements=[self.table.c.id],
            set_=values,
        )
        with self.engine.begin() as conn:
            conn.execute(stmt)

    def list(
        self,
        user_id: str | None = None,
        target_type: str | None = None,
        processed: bool | None = None,
        limit: int = 100,
    ) -> list[UserFeedback]:
        stmt = self.table.select()
        if user_id is not None:
            stmt = stmt.where(self.table.c.user_id == user_id)
        if target_type is not None:
            stmt = stmt.where(self.table.c.target_type == target_type)
        if processed is not None:
            stmt = stmt.where(self.table.c.processed == processed)
        stmt = stmt.order_by(self.table.c.created_at.desc()).limit(limit)
        with self.engine.connect() as conn:
            rows = conn.execute(stmt).mappings().all()
        return [UserFeedback(**json.loads(row["data"])) for row in rows]
