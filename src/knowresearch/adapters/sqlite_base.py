"""SQLite 通用基础：连接管理 + Record 序列化/反序列化。

SessionStore / WorkingMemory / LongTermMemory / SkillStore / TraceStore
五个端口都基于 SQLite + Record，共享这一层。

设计要点：
- Record 序列化为 JSON 字符串存 BLOB 列，保持 schema 简洁
- 元数据（id/kind/source/created_at/updated_at）拆成独立列，方便索引和查询
- 所有表共享同一套建表逻辑，通过 table_name 参数区分
"""

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

from knowresearch.core.schemas import Record


_local = threading.local()


def get_connection(db_path: str) -> sqlite3.Connection:
    """获取当前线程的 SQLite 连接（线程局部，避免跨线程共享连接）。

    SQLite 默认不支持多线程共享连接，每个线程用自己的连接。
    """
    key = f"conn_{db_path}"
    conn = getattr(_local, key, None)
    if conn is None:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        setattr(_local, key, conn)
    return conn


def close_connection(db_path: str) -> None:
    """关闭并移除当前线程缓存的 SQLite 连接。"""
    key = f"conn_{db_path}"
    conn = getattr(_local, key, None)
    if conn is not None:
        conn.close()
        delattr(_local, key)


@contextmanager
def transaction(db_path: str) -> Iterator[sqlite3.Connection]:
    """事务上下文管理器：成功提交，失败回滚。"""
    conn = get_connection(db_path)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def ensure_table(db_path: str, table_name: str) -> None:
    """确保 Record 表存在。

    建表逻辑已统一收归迁移机制管理（versions/001_initial.py），
    此处仅触发 ensure_latest，保证所有表（含传入的 table_name）已创建。
    table_name 参数保留以维持现有调用签名。
    """
    from knowresearch.adapters.migrations import ensure_latest

    ensure_latest(db_path)


def record_to_row(record: Record) -> tuple[str, str, str, str, str, str]:
    """Record → 数据库行。data 字段存完整 JSON（mode='json' 把 datetime 转成 ISO 字符串）。"""
    data = json.dumps(record.model_dump(mode="json"), ensure_ascii=False)
    return (
        record.id,
        record.kind,
        record.source,
        record.created_at.isoformat(),
        record.updated_at.isoformat(),
        data,
    )


def row_to_record(row: sqlite3.Row) -> Record:
    """数据库行 → Record。从 data 字段的 JSON 反序列化。"""
    data = json.loads(row["data"])
    return Record(**data)


def insert_record(db_path: str, table_name: str, record: Record) -> None:
    """插入一条 Record。"""
    ensure_table(db_path, table_name)
    with transaction(db_path) as conn:
        conn.execute(
            f"INSERT OR REPLACE INTO {table_name} "
            f"(id, kind, source, created_at, updated_at, data) "
            f"VALUES (?, ?, ?, ?, ?, ?)",
            record_to_row(record),
        )


def get_record(db_path: str, table_name: str, record_id: str) -> Record | None:
    """按 ID 查询 Record，不存在返回 None。"""
    ensure_table(db_path, table_name)
    conn = get_connection(db_path)
    row = conn.execute(
        f"SELECT * FROM {table_name} WHERE id = ?", (record_id,)
    ).fetchone()
    return row_to_record(row) if row else None


def delete_record(db_path: str, table_name: str, record_id: str) -> None:
    """按 ID 删除 Record。"""
    ensure_table(db_path, table_name)
    with transaction(db_path) as conn:
        conn.execute(f"DELETE FROM {table_name} WHERE id = ?", (record_id,))


def list_records(
    db_path: str,
    table_name: str,
    filters: dict[str, Any] | None = None,
) -> list[Record]:
    """列出所有 Record，可按列过滤（id/kind/source/created_at/updated_at）。

    filters 中的 key 必须是表的独立列，不能是 data 里的字段。
    如需按 metadata 过滤，在 Python 层做。
    """
    ensure_table(db_path, table_name)
    conn = get_connection(db_path)
    if filters:
        clauses = []
        values: list[Any] = []
        for key, value in filters.items():
            clauses.append(f"{key} = ?")
            values.append(value)
        where = " WHERE " + " AND ".join(clauses)
        rows = conn.execute(
            f"SELECT * FROM {table_name}{where} ORDER BY created_at DESC",
            values,
        ).fetchall()
    else:
        rows = conn.execute(
            f"SELECT * FROM {table_name} ORDER BY created_at DESC"
        ).fetchall()
    return [row_to_record(row) for row in rows]


def update_record_data(db_path: str, table_name: str, record: Record) -> None:
    """更新一条 Record（全量覆盖 data 字段，并刷新 updated_at）。"""
    record.updated_at = datetime.now()
    insert_record(db_path, table_name, record)