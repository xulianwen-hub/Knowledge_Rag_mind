"""迁移 001：初始化所有表。

把原分散在各适配器 __init__ 里的 CREATE TABLE 集中到此处，
作为版本化 schema 的起点。

创建的表：
- long_term_memory / skills / traces：通用 Record 表（id/kind/source/created_at/updated_at/data + 索引）
- session_messages：会话消息（带 TTL 过期时间）
- working_memory：工作记忆（任务状态）
"""

import sqlite3


VERSION = "001"


def _create_record_table(conn: sqlite3.Connection, table_name: str) -> None:
    """创建通用 Record 表（结构与 sqlite_base.ensure_table 原逻辑一致）。"""
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id TEXT PRIMARY KEY,
            kind TEXT NOT NULL,
            source TEXT DEFAULT '',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            data TEXT NOT NULL
        )
        """
    )
    conn.execute(
        f"CREATE INDEX IF NOT EXISTS idx_{table_name}_kind ON {table_name}(kind)"
    )
    conn.execute(
        f"CREATE INDEX IF NOT EXISTS idx_{table_name}_source ON {table_name}(source)"
    )


def upgrade(conn: sqlite3.Connection) -> None:
    for table in ("long_term_memory", "skills", "traces"):
        _create_record_table(conn, table)

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS session_messages (
            session_id TEXT PRIMARY KEY,
            messages TEXT NOT NULL,
            expires_at TEXT NOT NULL
        )
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS working_memory (
            task_id TEXT PRIMARY KEY,
            state TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )