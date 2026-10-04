"""LongTermMemoryPort 的 PostgreSQL 实现。

当前只实现 kind="profile" 的用户画像。表设计在通用 Record 字段之外
增加 user_id 独立列，用于多用户隔离和按用户读取。
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from sqlalchemy import Column, Index, MetaData, String, Table

from knowresearch.adapters.postgres_base import (
    JSONType,
    create_postgres_engine,
    record_to_row,
    row_to_record,
)
from knowresearch.core.ports import LongTermMemoryPort
from knowresearch.core.schemas import Record


class PostgresLongTermMemory(LongTermMemoryPort):
    """基于 PostgreSQL 的长期记忆，当前只服务 profile。"""

    TABLE_NAME = "long_term_memory"
    DEFAULT_USER_ID = "default"

    def __init__(self, database_url: str):
        self.engine = create_postgres_engine(database_url)
        self.metadata = MetaData()
        self.table = Table(
            self.TABLE_NAME,
            self.metadata,
            Column("id", String, primary_key=True),
            Column("user_id", String, nullable=False, default=self.DEFAULT_USER_ID),
            Column("kind", String, nullable=False),
            Column("source", String, nullable=False, default=""),
            Column("created_at", String, nullable=False),
            Column("updated_at", String, nullable=False),
            Column("data", JSONType, nullable=False),
        )
        Index(f"idx_{self.TABLE_NAME}_user_kind", self.table.c.user_id, self.table.c.kind)
        self.metadata.create_all(self.engine)

    def add(self, memory: Record, user_id: str | None = None) -> None:
        uid = self._resolve_user_id(memory, user_id)
        self._upsert(memory, uid)

    def search(
        self,
        query: str,
        top_k: int = 5,
        user_id: str | None = None,
    ) -> list[Record]:
        records = self.list(user_id=user_id)
        if not query:
            return records[:top_k]

        needle = query.lower()
        matched = []
        for record in records:
            metadata_text = json.dumps(record.metadata, ensure_ascii=False).lower()
            if needle in record.content.lower() or needle in metadata_text:
                matched.append(record)
        return matched[:top_k]

    def list(
        self,
        user_id: str | None = None,
        kind: str | None = None,
    ) -> list[Record]:
        stmt = self.table.select()
        if user_id is not None:
            stmt = stmt.where(self.table.c.user_id == user_id)
        if kind is not None:
            stmt = stmt.where(self.table.c.kind == kind)
        stmt = stmt.order_by(self.table.c.created_at.desc())

        with self.engine.connect() as conn:
            rows = conn.execute(stmt).mappings().all()
        return [row_to_record(row) for row in rows]

    def get(self, memory_id: str, user_id: str | None = None) -> Record | None:
        stmt = self.table.select().where(self.table.c.id == memory_id)
        if user_id is not None:
            stmt = stmt.where(self.table.c.user_id == user_id)
        with self.engine.connect() as conn:
            row = conn.execute(stmt).mappings().first()
        return row_to_record(row) if row else None

    def update(
        self,
        memory_id: str,
        user_id: str | None = None,
        **kwargs: Any,
    ) -> None:
        record = self.get(memory_id, user_id=user_id)
        if record is None:
            return
        record.metadata.update(kwargs)
        record.updated_at = datetime.now()
        self._upsert(record, self._resolve_user_id(record, user_id))

    def delete(self, memory_id: str, user_id: str | None = None) -> None:
        stmt = self.table.delete().where(self.table.c.id == memory_id)
        if user_id is not None:
            stmt = stmt.where(self.table.c.user_id == user_id)
        with self.engine.begin() as conn:
            conn.execute(stmt)

    def upsert_profile(
        self,
        user_id: str,
        content: str,
        dimension: str,
        source: str = "",
        confidence: float = 0.0,
        explicit: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> Record:
        """按 (user_id, dimension) 写入或更新一条用户画像。"""
        existing = self.get_profile(user_id, dimension)
        if existing is not None:
            if existing.metadata.get("explicit") is True and not explicit:
                return existing
            existing.content = content
            existing.source = source or existing.source
            existing.metadata.update(metadata or {})
            existing.metadata.update(
                {
                    "user_id": user_id,
                    "dimension": dimension,
                    "confidence": confidence,
                    "explicit": explicit,
                }
            )
            existing.updated_at = datetime.now()
            self._upsert(existing, user_id)
            return existing

        memory = Record(
            kind="profile",
            content=content,
            source=source,
            metadata={
                **(metadata or {}),
                "user_id": user_id,
                "dimension": dimension,
                "confidence": confidence,
                "explicit": explicit,
            },
        )
        self._upsert(memory, user_id)
        return memory

    def get_profile(self, user_id: str, dimension: str) -> Record | None:
        for record in self.list(user_id=user_id, kind="profile"):
            if record.metadata.get("dimension") == dimension:
                return record
        return None

    def _upsert(self, memory: Record, user_id: str) -> None:
        memory.metadata["user_id"] = user_id
        values = record_to_row(memory)
        values["user_id"] = user_id
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

    def _resolve_user_id(self, memory: Record, user_id: str | None) -> str:
        return user_id or memory.metadata.get("user_id") or self.DEFAULT_USER_ID
