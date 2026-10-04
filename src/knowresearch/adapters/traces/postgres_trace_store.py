"""TraceStorePort 的 PostgreSQL 实现。"""

from __future__ import annotations

from typing import Any

from sqlalchemy import MetaData

from knowresearch.adapters.postgres_base import (
    build_record_table,
    create_postgres_engine,
    delete_record,
    get_record,
    list_records,
    upsert_record,
)
from knowresearch.core.ports import TraceStorePort
from knowresearch.core.schemas import Record


class PostgresTraceStore(TraceStorePort):
    """基于 PostgreSQL 的任务轨迹存储。"""

    TABLE_NAME = "traces"

    def __init__(self, database_url: str):
        self.engine = create_postgres_engine(database_url)
        self.metadata = MetaData()
        self.table = build_record_table(self.TABLE_NAME, self.metadata)
        self.metadata.create_all(self.engine)

    def save(self, trace: Record) -> None:
        upsert_record(self.engine, self.table, trace)

    def get(self, trace_id: str) -> Record | None:
        return get_record(self.engine, self.table, trace_id)

    def list(self, filters: dict[str, Any] | None = None) -> list[Record]:
        if not filters:
            return list_records(self.engine, self.table)

        column_filters: dict[str, Any] = {}
        metadata_filters: dict[str, Any] = {}
        valid_columns = {"id", "kind", "source", "created_at", "updated_at"}
        for key, value in filters.items():
            if key in valid_columns:
                column_filters[key] = value
            else:
                metadata_filters[key] = value

        records = list_records(
            self.engine,
            self.table,
            kind=column_filters.get("kind"),
            source=column_filters.get("source"),
        )
        if "id" in column_filters:
            records = [r for r in records if r.id == column_filters["id"]]
        if metadata_filters:
            records = [
                r
                for r in records
                if all(r.metadata.get(k) == v for k, v in metadata_filters.items())
            ]
        return records

    def delete(self, trace_id: str) -> None:
        delete_record(self.engine, self.table, trace_id)
