"""LongTermMemoryPort 适配器：SQLite 平表实现。

用 Record 承载记忆，kind 区分记忆类型（profile / glossary / entity）。
M1 阶段 search 用关键词匹配（content LIKE %query%），
M4 接入 embedding 后换成语义检索。
"""

from typing import Any

from knowresearch.adapters.sqlite_base import (
    delete_record,
    ensure_table,
    get_record,
    insert_record,
    list_records,
    update_record_data,
)
from knowresearch.core.ports import LongTermMemoryPort
from knowresearch.core.schemas import Record


class SQLLongTermMemory(LongTermMemoryPort):
    """基于 SQLite 平表的长期记忆。"""

    TABLE = "long_term_memory"

    def __init__(self, db_path: str):
        self._db_path = db_path
        ensure_table(db_path, self.TABLE)

    def add(self, memory: Record, user_id: str | None = None) -> None:
        if user_id is not None:
            memory.metadata["user_id"] = user_id
        insert_record(self._db_path, self.TABLE, memory)

    def search(
        self,
        query: str,
        top_k: int = 5,
        user_id: str | None = None,
    ) -> list[Record]:
        """M1 阶段用关键词匹配，M4 换 embedding 语义检索。"""
        all_records = self.list(user_id=user_id)
        if not query:
            return all_records[:top_k]
        matched = [r for r in all_records if query in r.content]
        return matched[:top_k]

    def list(
        self,
        user_id: str | None = None,
        kind: str | None = None,
    ) -> list[Record]:
        records = list_records(self._db_path, self.TABLE)
        if kind is not None:
            records = [r for r in records if r.kind == kind]
        if user_id is not None:
            records = [
                r for r in records if r.metadata.get("user_id") == user_id
            ]
        return records

    def update(
        self,
        memory_id: str,
        user_id: str | None = None,
        **kwargs: Any,
    ) -> None:
        record = get_record(self._db_path, self.TABLE, memory_id)
        if record is None:
            return
        if user_id is not None and record.metadata.get("user_id") != user_id:
            return
        record.metadata.update(kwargs)
        update_record_data(self._db_path, self.TABLE, record)

    def delete(self, memory_id: str, user_id: str | None = None) -> None:
        record = get_record(self._db_path, self.TABLE, memory_id)
        if record is None:
            return
        if user_id is not None and record.metadata.get("user_id") != user_id:
            return
        delete_record(self._db_path, self.TABLE, memory_id)
