"""bootstrap 路径和基础组装逻辑测试。"""

from pathlib import Path

from knowresearch.bootstrap import (
    PROJECT_ROOT,
    resolve_project_path,
    sqlite_db_path,
)


def test_resolve_project_path_relative():
    resolved = resolve_project_path("data/index")
    assert resolved == (PROJECT_ROOT / "data" / "index").resolve()


def test_resolve_project_path_absolute():
    absolute = str(PROJECT_ROOT / "data")
    assert resolve_project_path(absolute) == Path(absolute)


def test_sqlite_db_path_from_url():
    path = sqlite_db_path("sqlite:///./data/knowresearch.db")
    assert path == (PROJECT_ROOT / "data" / "knowresearch.db").resolve()
