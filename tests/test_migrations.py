"""验证数据迁移机制：版本化 schema、幂等、向后兼容。

测试范围：
1. 全新库执行 ensure_latest 后所有表存在，schema_migrations 记录 version=001
2. 幂等：重复执行 ensure_latest 不报错、不重复执行迁移
3. 已有旧表（手工建表）的库执行 ensure_latest 能平滑接管
4. 新增迁移脚本后版本能正确升级
"""

import importlib
import os
import sys
import tempfile

import pytest

from knowresearch.adapters.migrations import ensure_latest, get_current_version
from knowresearch.adapters.sqlite_base import close_connection, get_connection

DB_PATH = os.path.join(tempfile.gettempdir(), "knowresearch_test_migrations.db")

EXPECTED_TABLES = {
    "long_term_memory",
    "skills",
    "traces",
    "session_messages",
    "working_memory",
    "documents",
    "schema_migrations",
}


def _cleanup():
    close_connection(DB_PATH)
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)


@pytest.fixture(autouse=True)
def setup_teardown():
    _cleanup()
    yield
    _cleanup()


def _table_names() -> set[str]:
    conn = get_connection(DB_PATH)
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()
    return {row["name"] for row in rows}


def test_ensure_latest_creates_all_tables():
    ensure_latest(DB_PATH)
    tables = _table_names()
    assert EXPECTED_TABLES.issubset(tables), f"缺少表: {EXPECTED_TABLES - tables}"


def test_ensure_latest_records_latest_version():
    ensure_latest(DB_PATH)
    assert get_current_version(DB_PATH) == "003"


def test_ensure_latest_is_idempotent():
    ensure_latest(DB_PATH)
    ensure_latest(DB_PATH)
    ensure_latest(DB_PATH)

    conn = get_connection(DB_PATH)
    count = conn.execute(
        "SELECT COUNT(*) FROM schema_migrations"
    ).fetchone()[0]
    assert count == 3, "迁移不应重复执行（001 + 002 + 003）"
    assert get_current_version(DB_PATH) == "003"


def test_ensure_latest_works_with_pre_existing_tables():
    """模拟旧数据库：表已存在但没有 schema_migrations 表，
    ensure_latest 应能平滑接管（CREATE TABLE IF NOT EXISTS 不报错）。"""
    conn = get_connection(DB_PATH)
    conn.execute(
        """
        CREATE TABLE long_term_memory (
            id TEXT PRIMARY KEY,
            kind TEXT NOT NULL,
            source TEXT DEFAULT '',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            data TEXT NOT NULL
        )
        """
    )
    conn.commit()

    ensure_latest(DB_PATH)

    tables = _table_names()
    assert EXPECTED_TABLES.issubset(tables)
    assert get_current_version(DB_PATH) == "003"


def test_new_migration_upgrades_version(tmp_path):
    """模拟新增 004 迁移脚本，验证版本升级。"""
    import knowresearch.adapters.migrations.versions as versions_pkg
    versions_dir = os.path.dirname(versions_pkg.__file__)

    fake_004 = os.path.join(versions_dir, "004_test_add_table.py")
    with open(fake_004, "w", encoding="utf-8") as f:
        f.write(
            'VERSION = "004"\n\n'
            "import sqlite3\n\n"
            "def upgrade(conn: sqlite3.Connection) -> None:\n"
            '    conn.execute("CREATE TABLE IF NOT EXISTS test_migration_004 (id TEXT PRIMARY KEY)")\n'
        )

    try:
        importlib.invalidate_caches()

        ensure_latest(DB_PATH)

        assert get_current_version(DB_PATH) == "004"
        tables = _table_names()
        assert "test_migration_004" in tables
    finally:
        if os.path.exists(fake_004):
            os.remove(fake_004)
        importlib.invalidate_caches()