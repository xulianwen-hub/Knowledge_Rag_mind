"""配置管理：统一从 .env / 配置文件读取可替换项，避免代码硬编码。"""

from knowresearch.config.settings import Settings, settings

__all__ = ["Settings", "settings"]