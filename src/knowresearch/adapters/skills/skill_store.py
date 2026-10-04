"""SkillStorePort 适配器：SQLite 实现。

技能用 Record 承载，kind = "skill"。
M1 阶段 search 用关键词匹配，M5 接入时换成语义匹配。
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
from knowresearch.core.ports import SkillStorePort
from knowresearch.core.schemas import Record


class SQLiteSkillStore(SkillStorePort):
    """基于 SQLite 的技能存储。"""

    TABLE = "skills"

    def __init__(self, db_path: str):
        self._db_path = db_path
        ensure_table(db_path, self.TABLE)

    def save(self, skill: Record) -> None:
        insert_record(self._db_path, self.TABLE, skill)

    def get(self, skill_id: str) -> Record | None:
        return get_record(self._db_path, self.TABLE, skill_id)

    def search(self, query: str, top_k: int = 5) -> list[Record]:
        """M1 阶段用关键词匹配，M5 换语义匹配。"""
        all_skills = list_records(self._db_path, self.TABLE)
        if not query:
            return all_skills[:top_k]
        matched = [s for s in all_skills if query in s.content]
        return matched[:top_k]

    def list_all(
        self,
        user_id: str | None = None,
        status: str | None = None,
        limit: int = 100,
    ) -> list[Record]:
        records = list_records(self._db_path, self.TABLE)
        if user_id is not None:
            records = [
                record
                for record in records
                if record.metadata.get("user_id") == user_id
            ]
        if status is not None:
            records = [
                record
                for record in records
                if record.metadata.get("status") == status
            ]
        return records[:limit]

    def update(self, skill_id: str, **kwargs: Any) -> None:
        record = get_record(self._db_path, self.TABLE, skill_id)
        if record is None:
            return
        record.metadata.update(kwargs)
        update_record_data(self._db_path, self.TABLE, record)

    def delete(self, skill_id: str) -> None:
        delete_record(self._db_path, self.TABLE, skill_id)

    def update_status(
        self,
        skill_id: str,
        status: str,
        user_id: str | None = None,
    ) -> None:
        record = get_record(self._db_path, self.TABLE, skill_id)
        if record is None:
            return
        if user_id is not None and record.metadata.get("user_id") != user_id:
            return
        record.metadata["status"] = status
        update_record_data(self._db_path, self.TABLE, record)
