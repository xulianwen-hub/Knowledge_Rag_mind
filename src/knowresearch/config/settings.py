"""
统一配置模块。

所有可替换项从 .env / 环境变量读取，代码中零硬编码。
使用方式：在整个项目中 `from knowresearch.config import settings`。

设计原则（来自开发计划书 §1）：
- 先定接口、后定存储：配置只定义"是什么"，不绑定"怎么实现"
- 低耦合：业务代码只依赖 settings 对象，不直接读环境变量
- 可替换：换模型/换路径/换数据库，只改 .env
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field

# 在模块加载时读取 .env，确保 os.environ 中有所有变量
_env_file = Path(__file__).resolve().parents[3] / ".env"
load_dotenv(_env_file)


def _env(key: str, default: str = "") -> str:
    """读取环境变量，缺失时立刻报错（除非有默认值）。"""
    value = os.getenv(key, default)
    if not value and default == "":
        raise ValueError(
            f"缺少必需的环境变量: {key}。"
            f"请检查项目根目录的 .env 文件，或参考 .env.example 模板。"
        )
    return value


# ================================================================
# 各子模块配置
# ================================================================


class LLMConfig(BaseModel):
    """大语言模型配置（当前：DeepSeek API）。"""

    api_key: str = Field(default_factory=lambda: _env("DEEPSEEK_API_KEY"))
    base_url: str = Field(default_factory=lambda: _env("DEEPSEEK_BASE_URL"))
    model: str = Field(default_factory=lambda: _env("DEEPSEEK_MODEL"))


class VLMConfig(BaseModel):
    """视觉语言模型配置（当前：qwen3.8-omni-flash）。"""

    api_key: str = Field(default_factory=lambda: _env("VLM_API_KEY"))
    base_url: str = Field(default_factory=lambda: _env("VLM_BASE_URL"))
    model: str = Field(default_factory=lambda: _env("VLM_MODEL"))


class EmbeddingConfig(BaseModel):
    """Embedding 模型配置（本地 BGE 系列）。"""

    model_name: str = Field(
        default_factory=lambda: _env("EMBEDDING_MODEL_NAME")
    )
    model_dir: str = Field(
        default_factory=lambda: _env("EMBEDDING_MODEL_DIR")
    )
    dimension: int = Field(
        default_factory=lambda: int(_env("EMBEDDING_DIMENSION", "768"))
    )


class RerankerConfig(BaseModel):
    """Reranker 模型配置（本地 BGE Reranker）。"""

    model_name: str = Field(
        default_factory=lambda: _env("RERANKER_MODEL_NAME")
    )
    model_dir: str = Field(
        default_factory=lambda: _env("RERANKER_MODEL_DIR")
    )


class DataConfig(BaseModel):
    """数据目录配置（raw → processed → index 三级流水线）。"""

    raw_dir: str = Field(default_factory=lambda: _env("DATA_RAW_DIR"))
    processed_dir: str = Field(default_factory=lambda: _env("DATA_PROCESSED_DIR"))
    index_dir: str = Field(default_factory=lambda: _env("DATA_INDEX_DIR"))


class StorageConfig(BaseModel):
    """持久化存储配置（MVP: SQLite + 本地文件目录）。"""

    database_url: str = Field(default_factory=lambda: _env("DATABASE_URL"))
    postgres_url: str = Field(
        default_factory=lambda: _env(
            "POSTGRES_URL",
            "postgresql+psycopg2://knowresearch:knowresearch@localhost:5432/knowresearch",
        )
    )
    file_storage_dir: str = Field(default_factory=lambda: _env("FILE_STORAGE_DIR"))
    redis_url: str = Field(
        default_factory=lambda: _env("REDIS_URL", "redis://localhost:6379/0")
    )


class LogConfig(BaseModel):
    """日志配置。"""

    level: str = Field(default_factory=lambda: _env("LOG_LEVEL"))
    dir: str = Field(default_factory=lambda: _env("LOG_DIR"))


class AppConfig(BaseModel):
    """应用运行模式配置。"""

    demo_mode: bool = Field(
        default_factory=lambda: _env("DEMO_MODE", "false").lower()
        in {"1", "true", "yes", "on"}
    )


# ================================================================
# 顶层设置（聚合所有子模块）
# ================================================================


class Settings(BaseModel):
    """全局设置，聚合所有子模块配置。"""

    llm: LLMConfig = Field(default_factory=LLMConfig)
    vlm: VLMConfig = Field(default_factory=VLMConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    log: LogConfig = Field(default_factory=LogConfig)
    app: AppConfig = Field(default_factory=AppConfig)


# 全局单例——整个项目只 import 这一个对象
settings = Settings()
