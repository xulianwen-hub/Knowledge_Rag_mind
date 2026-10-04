"""迁移 003：给 session_messages 表加 summary 字段。

用于存储多轮对话的历史摘要。当对话超过滑动窗口上限时，
把溢出的早期对话压缩成摘要存在 summary 字段，
构建 prompt 时和最近窗口的对话一起注入 LLM。
"""

import sqlite3


VERSION = "003"


def upgrade(conn: sqlite3.Connection) -> None:
    try:
        conn.execute(
            "ALTER TABLE session_messages ADD COLUMN summary TEXT DEFAULT ''"
        )
    except sqlite3.OperationalError:
        pass