"""数据库迁移模块。

对外暴露：
- ensure_latest(db_path)：执行所有未执行的迁移
- get_current_version(db_path)：查询当前版本号
"""

from knowresearch.adapters.migrations.runner import (
    ensure_latest,
    get_current_version,
)

__all__ = ["ensure_latest", "get_current_version"]