"""数据库迁移引擎。

版本化管理 SQLite schema，后续加字段/加表只需新增迁移脚本，
不破坏已有数据。

设计要点：
- schema_migrations 表记录已执行的版本号
- 迁移脚本放在 versions/ 目录，每个脚本暴露 VERSION 和 upgrade(conn)
- ensure_latest 幂等：已执行的迁移不会重复执行
- 通过版本号对比决定是否需要执行，避免每次全量扫描
"""

import importlib
import pkgutil
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Callable

from knowresearch.adapters.sqlite_base import get_connection, transaction


MIGRATIONS_TABLE = "schema_migrations"


def _ensure_migrations_table(db_path: str) -> None:
    conn = get_connection(db_path)
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {MIGRATIONS_TABLE} (
            version TEXT PRIMARY KEY,
            applied_at TEXT NOT NULL
        )
        """
    )
    conn.commit()


def _get_applied_versions(db_path: str) -> set[str]:
    conn = get_connection(db_path)
    rows = conn.execute(f"SELECT version FROM {MIGRATIONS_TABLE}").fetchall()
    return {row["version"] for row in rows}


def _get_migration_scripts() -> list[tuple[str, Callable[[sqlite3.Connection], None]]]:
    """自动发现 versions/ 目录下的迁移脚本。

    约定：文件名以数字开头，模块暴露 VERSION(str) 和 upgrade(conn)。
    返回按版本号升序排序的 [(version, upgrade_func), ...]。
    """
    import knowresearch.adapters.migrations.versions as versions_pkg

    versions_path = Path(versions_pkg.__file__).parent
    scripts: list[tuple[str, Callable[[sqlite3.Connection], None]]] = []

    for module_info in pkgutil.iter_modules([str(versions_path)]):
        if not module_info.name[0].isdigit():
            continue
        module = importlib.import_module(
            f"knowresearch.adapters.migrations.versions.{module_info.name}"
        )
        version = getattr(module, "VERSION", None)
        upgrade = getattr(module, "upgrade", None)
        if version is None or upgrade is None:
            continue
        scripts.append((version, upgrade))

    scripts.sort(key=lambda x: x[0])
    return scripts


def ensure_latest(db_path: str) -> None:
    """执行所有未执行的迁移，保证数据库 schema 为最新版本。

    幂等：已执行的迁移不会重复执行。
    线程安全：复用 sqlite_base 的线程局部连接。
    每次只查询 schema_migrations 最新版本（单行查询），无进程级缓存，
    避免数据库文件被外部替换后状态不一致。
    """
    _ensure_migrations_table(db_path)

    scripts = _get_migration_scripts()
    if not scripts:
        return

    latest_registered = scripts[-1][0]

    conn = get_connection(db_path)
    row = conn.execute(
        f"SELECT version FROM {MIGRATIONS_TABLE} ORDER BY version DESC LIMIT 1"
    ).fetchone()
    db_version = row["version"] if row else None

    if db_version == latest_registered:
        return

    applied = _get_applied_versions(db_path)
    for version, upgrade in scripts:
        if version in applied:
            continue
        with transaction(db_path) as conn:
            upgrade(conn)
            conn.execute(
                f"INSERT INTO {MIGRATIONS_TABLE} (version, applied_at) VALUES (?, ?)",
                (version, datetime.now().isoformat()),
            )


def get_current_version(db_path: str) -> str | None:
    """返回当前数据库的最新迁移版本号，无任何迁移时返回 None。"""
    _ensure_migrations_table(db_path)
    conn = get_connection(db_path)
    row = conn.execute(
        f"SELECT version FROM {MIGRATIONS_TABLE} ORDER BY version DESC LIMIT 1"
    ).fetchone()
    return row["version"] if row else None