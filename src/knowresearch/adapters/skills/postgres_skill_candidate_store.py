"""SkillCandidateStorePort 的 PostgreSQL 实现。"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from sqlalchemy import Column, Index, MetaData, String, Table

from knowresearch.adapters.postgres_base import JSONType, create_postgres_engine
from knowresearch.core.ports import SkillCandidateStorePort
from knowresearch.core.schemas import SkillCandidate


class PostgresSkillCandidateStore(SkillCandidateStorePort):
    """基于 PostgreSQL 的候选技能存储。"""

    TABLE_NAME = "skill_candidates"

    def __init__(self, database_url: str):
        self.engine = create_postgres_engine(database_url)
        self.metadata = MetaData()
        self.table = Table(
            self.TABLE_NAME,
            self.metadata,
            Column("id", String, primary_key=True),
            Column("user_id", String, nullable=False),
            Column("name", String, nullable=False, default=""),
            Column("status", String, nullable=False, default="pending"),
            Column("source", String, nullable=False, default=""),
            Column("created_at", String, nullable=False),
            Column("updated_at", String, nullable=False),
            Column("data", JSONType, nullable=False),
        )
        Index(
            f"idx_{self.TABLE_NAME}_user_status",
            self.table.c.user_id,
            self.table.c.status,
        )
        Index(f"idx_{self.TABLE_NAME}_source", self.table.c.source)
        self.metadata.create_all(self.engine)

    def save(self, candidate: SkillCandidate) -> None:
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

    def get(self, candidate_id: str) -> SkillCandidate | None:
        stmt = self.table.select().where(self.table.c.id == candidate_id)
        with self.engine.connect() as conn:
            row = conn.execute(stmt).mappings().first()
        return self._row_to_candidate(row) if row else None

    def list(
        self,
        user_id: str | None = None,
        status: str | None = None,
        limit: int = 100,
    ) -> list[SkillCandidate]:
        stmt = self.table.select()
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
        target_skill_id: str | None = None,
    ) -> None:
        candidate = self.get(candidate_id)
        if candidate is None:
            return
        candidate.status = status
        candidate.reviewed_by = reviewed_by
        candidate.review_note = review_note
        candidate.reviewed_at = datetime.now()
        if target_skill_id is not None:
            candidate.target_skill_id = target_skill_id
        self.save(candidate)

    def exists_for_source(self, user_id: str, source: str) -> bool:
        if not source:
            return False
        stmt = (
            self.table.select()
            .where(self.table.c.user_id == user_id)
            .where(self.table.c.source == source)
        )
        with self.engine.connect() as conn:
            row = conn.execute(stmt).first()
        return row is not None

    def delete(self, candidate_id: str) -> None:
        with self.engine.begin() as conn:
            conn.execute(
                self.table.delete().where(self.table.c.id == candidate_id)
            )

    def _candidate_to_row(self, candidate: SkillCandidate) -> dict[str, Any]:
        return {
            "id": candidate.id,
            "user_id": candidate.user_id,
            "name": candidate.name,
            "status": candidate.status,
            "source": candidate.source_trace_ids[0] if candidate.source_trace_ids else "",
            "created_at": candidate.created_at.isoformat(),
            "updated_at": datetime.now().isoformat(),
            "data": json.dumps(candidate.model_dump(mode="json"), ensure_ascii=False),
        }

    @staticmethod
    def _row_to_candidate(row: Any) -> SkillCandidate:
        data = row["data"]
        if isinstance(data, str):
            data = json.loads(data)
        return SkillCandidate(**data)
