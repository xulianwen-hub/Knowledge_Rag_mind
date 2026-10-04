"""WorkingMemoryPort 适配器：SQLite 实现。

任务状态存在 working_memory 表，task_id 为主键，state 存 JSON。
支持 save / load / update / delete。
"""

import json
from typing import Any

from knowresearch.adapters.migrations import ensure_latest
from knowresearch.adapters.sqlite_base import get_connection, transaction
from knowresearch.core.ports import WorkingMemoryPort


class SQLiteWorkingMemory(WorkingMemoryPort):
    """基于 SQLite 的工作记忆。"""

    def __init__(self, db_path: str):
        self._db_path = db_path
        ensure_latest(self._db_path)

    def save_state(self, task_id: str, state: dict[str, Any]) -> None:
        state_json = json.dumps(state, ensure_ascii=False)
        from datetime import datetime

        updated_at = datetime.now().isoformat()
        with transaction(self._db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO working_memory "
                "(task_id, state, updated_at) VALUES (?, ?, ?)",
                (task_id, state_json, updated_at),
            )

    def load_state(self, task_id: str) -> dict[str, Any] | None:
        conn = get_connection(self._db_path)
        row = conn.execute(
            "SELECT state FROM working_memory WHERE task_id = ?",
            (task_id,),
        ).fetchone()
        if row is None:
            return None
        return json.loads(row["state"])

    def update_state(self, task_id: str, **kwargs: Any) -> None:
        state = self.load_state(task_id) or {}
        state.update(kwargs)
        self.save_state(task_id, state)

    def delete_task(self, task_id: str) -> None:
        with transaction(self._db_path) as conn:
            conn.execute(
                "DELETE FROM working_memory WHERE task_id = ?",
                (task_id,),
            )