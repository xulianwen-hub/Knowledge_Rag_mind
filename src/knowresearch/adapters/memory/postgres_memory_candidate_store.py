"""MemoryCandidateStorePort 的 PostgreSQL 实现。"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from sqlalchemy import Column, Float, Index, MetaData, String, Table, Text, select

from knowresearch.adapters.postgres_base import JSONType, create_postgres_engine
from knowresearch.core.ports import MemoryCandidateStorePort
from knowresearch.core.schemas import MemoryCandidate


class PostgresMemoryCandidateStore(MemoryCandidateStorePort):
    """基于 PostgreSQL 的候选记忆审核表。"""

    TABLE_NAME = "memory_candidates"

    def __init__(self, database_url: str):
        self.engine = create_postgres_engine(database_url)
        self.metadata = MetaData()
        self.table = Table(
            self.TABLE_NAME,
            self.metadata,
            Column("id", String, primary_key=True),
            Column("user_id", String, nullable=False),
            Column("kind", String, nullable=False, default="profile"),
            Column("content", Text, nullable=False),
            Column("candidate_metadata", JSONType, nullable=False),
            Column("source", String, nullable=False, default=""),
            Column("evidence", Text, nullable=False, default=""),
            Column("confidence", Float, nullable=False, default=0.0),
            Column("status", String, nullable=False, default="pending"),
            Column("proposed_action", String, nullable=False, default="add"),
            Column("target_memory_id", String, nullable=True),
            Column("created_at", String, nullable=False),
            Column("reviewed_at", String, nullable=True),
            Column("reviewed_by", String, nullable=False, default=""),
            Column("review_note", Text, nullable=False, default=""),
        )
        Index(
            f"idx_{self.TABLE_NAME}_user_status",
            self.table.c.user_id,
            self.table.c.status,
        )
        Index(f"idx_{self.TABLE_NAME}_source", self.table.c.source)
        self.metadata.create_all(self.engine)

    def save(self, candidate: MemoryCandidate) -> None:
        values = self._candidate_to_row(candidate)
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

    def get(self, candidate_id: str) -> MemoryCandidate | None:
        with self.engine.connect() as conn:
            row = conn.execute(
                select(self.table).where(self.table.c.id == candidate_id)
            ).mappings().first()
        return self._row_to_candidate(row) if row else None

    def list(
        self,
        user_id: str | None = None,
        status: str | None = None,
        limit: int = 100,
    ) -> list[MemoryCandidate]:
        stmt = select(self.table)
        if user_id is not None:
            stmt = stmt.where(self.table.c.user_id == user_id)
        if status is not None:
            stmt = stmt.where(self.table.c.status == status)
        stmt = stmt.order_by(self.table.c.created_at.desc()).limit(limit)
        with self.engine.connect() as conn:
            rows = conn.execute(stmt).mappings().all()
        return [self._row_to_candidate(row) for row in rows]

    def update_review(
        self,
        candidate_id: str,
        status: str,
        reviewed_by: str = "",
        review_note: str = "",
        target_memory_id: str | None = None,
    ) -> None:
        values: dict[str, Any] = {
            "status": status,
            "reviewed_at": datetime.now().isoformat(),
            "reviewed_by": reviewed_by,
            "review_note": review_note,
        }
        if target_memory_id is not None:
            values["target_memory_id"] = target_memory_id
        with self.engine.begin() as conn:
            conn.execute(
                self.table.update()
                .where(self.table.c.id == candidate_id)
                .values(**values)
            )

    def exists_for_source(self, user_id: str, source: str) -> bool:
        if not source:
            return False
        with self.engine.connect() as conn:
            row = conn.execute(
                select(self.table.c.id).where(
                    self.table.c.user_id == user_id,
                    self.table.c.source == source,
                )
            ).first()
        return row is not None

    def delete(self, candidate_id: str) -> None:
        with self.engine.begin() as conn:
            conn.execute(
                self.table.delete().where(self.table.c.id == candidate_id)
            )

    def _candidate_to_row(self, candidate: MemoryCandidate) -> dict[str, Any]:
        return {
            "id": candidate.id,
            "user_id": candidate.user_id,
            "kind": candidate.kind,
            "content": candidate.content,
            "candidate_metadata": candidate.metadata,
            "source": candidate.source,
            "evidence": candidate.evidence,
            "confidence": candidate.confidence,
            "status": candidate.status,
            "proposed_action": candidate.proposed_action,
            "target_memory_id": candidate.target_memory_id,
            "created_at": candidate.created_at.isoformat(),
            "reviewed_at": (
                candidate.reviewed_at.isoformat() if candidate.reviewed_at else None
            ),
            "reviewed_by": candidate.reviewed_by,
            "review_note": candidate.review_note,
        }

    @staticmethod
    def _row_to_candidate(row: Any) -> MemoryCandidate:
        metadata = row["candidate_metadata"]
        if isinstance(metadata, str):
            metadata = json.loads(metadata)
        return MemoryCandidate(
            id=row["id"],
            user_id=row["user_id"],
            kind=row["kind"],
            content=row["content"],
            metadata=metadata,
            source=row["source"],
            evidence=row["evidence"],
            confidence=row["confidence"],
            status=row["status"],
            proposed_action=row["proposed_action"],
            target_memory_id=row["target_memory_id"],
            created_at=row["created_at"],
            reviewed_at=row["reviewed_at"],
            reviewed_by=row["reviewed_by"],
            review_note=row["review_note"],
        )
