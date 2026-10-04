"""TraceStorePort 适配器：SQLite 实现。

任务轨迹用 Record 承载，kind = "trace"。
M1 阶段 list 支持按 kind/source 过滤（独立列），
按 metadata 过滤在 Python 层做。
"""

from typing import Any

from knowresearch.adapters.sqlite_base import (
    delete_record,
    ensure_table,
    get_record,
    insert_record,
    list_records,
)
from knowresearch.core.ports import TraceStorePort
from knowresearch.core.schemas import Record


class SQLiteTraceStore(TraceStorePort):
    """基于 SQLite 的任务轨迹存储。"""

    TABLE = "traces"

    def __init__(self, db_path: str):
        self._db_path = db_path
        ensure_table(db_path, self.TABLE)

    def save(self, trace: Record) -> None:
        insert_record(self._db_path, self.TABLE, trace)

    def get(self, trace_id: str) -> Record | None:
        return get_record(self._db_path, self.TABLE, trace_id)

    def list(self, filters: dict[str, Any] | None = None) -> list[Record]:
        """过滤。filters 的 key 优先匹配独立列（id/kind/source），
        其余 key 在 Python 层匹配 metadata。"""
        if not filters:
            return list_records(self._db_path, self.TABLE)

        column_filters: dict[str, Any] = {}
        meta_filters: dict[str, Any] = {}
        valid_columns = {"id", "kind", "source", "created_at", "updated_at"}
        for key, value in filters.items():
            if key in valid_columns:
                column_filters[key] = value
            else:
                meta_filters[key] = value

        records = list_records(self._db_path, self.TABLE, column_filters or None)

        if meta_filters:
            records = [
                r
                for r in records
                if all(r.metadata.get(k) == v for k, v in meta_filters.items())
            ]
        return records

    def delete(self, trace_id: str) -> None:
        delete_record(self._db_path, self.TABLE, trace_id)