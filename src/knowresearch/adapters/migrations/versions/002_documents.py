"""迁移 002：创建 documents 表。

用于存储文档元数据（Document）和章节结构（SectionBlock），
复用通用 Record 表结构，通过 kind 字段区分 document / section。

M3 检索时需要根据 chunk 的 document_id 反查文档标题、
根据 section_id 反查章节标题，因此需要持久化。
"""

import sqlite3


VERSION = "002"


def upgrade(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS documents (
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
        "CREATE INDEX IF NOT EXISTS idx_documents_kind ON documents(kind)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_documents_source ON documents(source)"
    )