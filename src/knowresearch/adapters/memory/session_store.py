"""SessionStorePort 适配器：SQLite + TTL 自动过期 + 历史摘要。

会话消息存在 session_messages 表，带 expires_at 时间戳和 summary 字段。
每次 load/delete 时清理过期会话（惰性清理，不依赖后台线程）。

summary 字段存储溢出滑动窗口的早期对话摘要，与 messages 同生命周期。
"""

import json
from datetime import datetime, timedelta

from knowresearch.adapters.migrations import ensure_latest
from knowresearch.adapters.sqlite_base import get_connection, transaction
from knowresearch.core.ports import SessionStorePort


class SQLiteSessionStore(SessionStorePort):
    """基于 SQLite 的短期记忆，带 TTL 和历史摘要。"""

    def __init__(self, db_path: str):
        self._db_path = db_path
        ensure_latest(self._db_path)

    def _cleanup_expired(self) -> None:
        """惰性清理过期会话。"""
        now = datetime.now().isoformat()
        with transaction(self._db_path) as conn:
            conn.execute("DELETE FROM session_messages WHERE expires_at < ?", (now,))

    def save_messages(
        self,
        session_id: str,
        messages: list[dict[str, str]],
        ttl_seconds: int = 3600,
        summary: str = "",
    ) -> None:
        expires_at = (datetime.now() + timedelta(seconds=ttl_seconds)).isoformat()
        messages_json = json.dumps(messages, ensure_ascii=False)
        with transaction(self._db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO session_messages "
                "(session_id, messages, expires_at, summary) VALUES (?, ?, ?, ?)",
                (session_id, messages_json, expires_at, summary),
            )

    def load_messages(self, session_id: str) -> list[dict[str, str]]:
        self._cleanup_expired()
        conn = get_connection(self._db_path)
        row = conn.execute(
            "SELECT messages FROM session_messages WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        if row is None:
            return []
        return json.loads(row["messages"])

    def load_summary(self, session_id: str) -> str:
        self._cleanup_expired()
        conn = get_connection(self._db_path)
        row = conn.execute(
            "SELECT summary FROM session_messages WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        if row is None:
            return ""
        return row["summary"] or ""

    def delete_session(self, session_id: str) -> None:
        with transaction(self._db_path) as conn:
            conn.execute(
                "DELETE FROM session_messages WHERE session_id = ?",
                (session_id,),
            )