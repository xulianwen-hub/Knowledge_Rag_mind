"""SkillStorePort 的 PostgreSQL 实现。"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from sqlalchemy import Column, Index, Integer, MetaData, String, Table

from knowresearch.adapters.postgres_base import (
    JSONType,
    create_postgres_engine,
    record_to_row,
    row_to_record,
)
from knowresearch.core.ports import SkillStorePort
from knowresearch.core.schemas import Record


class PostgresSkillStore(SkillStorePort):
    """基于 PostgreSQL 的技能存储。"""

    TABLE_NAME = "skills"
    DEFAULT_USER_ID = "default"

    def __init__(self, database_url: str):
        self.engine = create_postgres_engine(database_url)
        self.metadata = MetaData()
        self.table = Table(
            self.TABLE_NAME,
            self.metadata,
            Column("id", String, primary_key=True),
            Column("user_id", String, nullable=False, default=self.DEFAULT_USER_ID),
            Column("kind", String, nullable=False, default="skill"),
            Column("name", String, nullable=False, default=""),
            Column("source", String, nullable=False, default=""),
            Column("status", String, nullable=False, default="pending"),
            Column("version", Integer, nullable=False, default=1),
            Column("usage_count", Integer, nullable=False, default=0),
            Column("success_count", Integer, nullable=False, default=0),
            Column("created_at", String, nullable=False),
            Column("updated_at", String, nullable=False),
            Column("data", JSONType, nullable=False),
        )
        Index(f"idx_{self.TABLE_NAME}_user_status", self.table.c.user_id, self.table.c.status)
        self.metadata.create_all(self.engine)

    def save(self, skill: Record) -> None:
        user_id = self._resolve_user_id(skill, None)
        self._upsert(skill, user_id)

    def get(self, skill_id: str, user_id: str | None = None) -> Record | None:
        stmt = self.table.select().where(self.table.c.id == skill_id)
        if user_id is not None:
            stmt = stmt.where(self.table.c.user_id == user_id)
        with self.engine.connect() as conn:
            row = conn.execute(stmt).mappings().first()
        return row_to_record(row) if row else None

    def search(
        self,
        query: str,
        top_k: int = 5,
        user_id: str | None = None,
    ) -> list[Record]:
        records = self.list_all(user_id=user_id)
        if not query:
            return records[:top_k]
        needle = query.lower()
        matched = []
        for record in records:
            metadata_text = json.dumps(record.metadata, ensure_ascii=False).lower()
            if needle in record.content.lower() or needle in metadata_text:
                matched.append(record)
        return matched[:top_k]

    def list_all(
        self,
        user_id: str | None = None,
        status: str | None = None,
        limit: int = 100,
    ) -> list[Record]:
        stmt = self.table.select()
        if user_id is not None:
            stmt = stmt.where(self.table.c.user_id == user_id)
        if status is not None:
            stmt = stmt.where(self.table.c.status == status)
        stmt = stmt.order_by(self.table.c.updated_at.desc()).limit(limit)
        with self.engine.connect() as conn:
            rows = conn.execute(stmt).mappings().all()
        return [row_to_record(row) for row in rows]

    def update(
        self,
        skill_id: str,
        user_id: str | None = None,
        **kwargs: Any,
    ) -> None:
        record = self.get(skill_id, user_id=user_id)
        if record is None:
            return
        record.metadata.update(kwargs)
        record.updated_at = datetime.now()
        self._upsert(record, self._resolve_user_id(record, user_id))

    def delete(self, skill_id: str, user_id: str | None = None) -> None:
        stmt = self.table.delete().where(self.table.c.id == skill_id)
        if user_id is not None:
            stmt = stmt.where(self.table.c.user_id == user_id)
        with self.engine.begin() as conn:
            conn.execute(stmt)

    def update_status(
        self,
        skill_id: str,
        status: str,
        user_id: str | None = None,
    ) -> None:
        self.update(skill_id, user_id=user_id, status=status)

    def _upsert(self, skill: Record, user_id: str) -> None:
        skill.metadata.setdefault("user_id", user_id)
        skill.metadata.setdefault("status", "pending")
        skill.metadata.setdefault("version", 1)
        skill.metadata.setdefault("usage_count", 0)
        skill.metadata.setdefault("success_count", 0)

        values = record_to_row(skill)
        values.update(
            {
                "user_id": user_id,
                "name": skill.metadata.get("name") or skill.content[:100],
                "status": skill.metadata.get("status", "pending"),
                "version": int(skill.metadata.get("version", 1)),
                "usage_count": int(skill.metadata.get("usage_count", 0)),
                "success_count": int(skill.metadata.get("success_count", 0)),
            }
        )
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

    def _resolve_user_id(self, skill: Record, user_id: str | None) -> str:
        return user_id or skill.metadata.get("user_id") or self.DEFAULT_USER_ID
