"""
统一数据模型：Record 基类 + Document IR（Document → Section → Chunk）。

设计原则（来自开发计划书 §1.3）：
- 统一 8 字段外壳：id / kind / content / metadata / source / created_at / updated_at / relations
- relations 和 timestamp 从第一天预留，M4 做图谱时直接用
- metadata 用 dict 承载不同 kind 的扩展字段，实现前向兼容

层级关系：
    Document (1) ── SectionBlock (N) ── Chunk (N)
"""

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


def _now() -> datetime:
    return datetime.now()


def _new_id() -> str:
    return uuid.uuid4().hex


# ================================================================
# Record 基类：所有记录的统一外壳
# ================================================================


class Record(BaseModel):
    """
    所有持久化记录的统一基类。

    8 个字段覆盖所有 kind 的共同需求；扩展字段放进 metadata，
    避免为每种 kind 定义独立 schema 造成接口膨胀。
    """

    id: str = Field(default_factory=_new_id, description="全局唯一 ID（uuid4 hex）")
    kind: str = Field(..., description="记录类型：document / section / chunk / ...")
    content: str = Field(default="", description="文本内容，Document 可为空字符串")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="扩展字段：页码、标题层级、文件大小等"
    )
    source: str = Field(default="", description="来源标识：文件路径 / 外部 URL")
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)
    relations: list[str] = Field(
        default_factory=list,
        description="关联的其他记录 ID 列表，M4 图谱时填充",
    )


# ================================================================
# Document IR：三层结构
# ================================================================


class Document(Record):
    """
    一份原始文档（PDF / Word / 图片）。

    content 字段留空——文档的实际内容拆到 Section / Chunk 里，
    Document 只存文档级元数据。
    """

    kind: str = Field(default="document", frozen=True)

    @property
    def file_name(self) -> str:
        return self.metadata.get("file_name", "")

    @property
    def file_hash(self) -> str:
        return self.metadata.get("file_hash", "")

    @property
    def file_type(self) -> str:
        return self.metadata.get("file_type", "")

    @property
    def total_pages(self) -> int:
        return self.metadata.get("total_pages", 0)


class SectionBlock(Record):
    """
    章节父块：文档的一个一级/二级/...标题及其下属内容。

    检索时优先返回高分 Section（整篇章节作为上下文），
    低分时才下钻到 Chunk 级别。
    """

    kind: str = Field(default="section", frozen=True)

    @property
    def document_id(self) -> str:
        return self.metadata.get("document_id", "")

    @property
    def section_title(self) -> str:
        return self.metadata.get("section_title", "")

    @property
    def section_level(self) -> int:
        return self.metadata.get("section_level", 1)

    @property
    def page_start(self) -> int:
        return self.metadata.get("page_start", 0)

    @property
    def page_end(self) -> int:
        return self.metadata.get("page_end", 0)


class Chunk(Record):
    """
    子块：检索的最小单位。

    带精确的定位元数据（页码 / 段落 / 字符偏移），
    用于生成答案时的引用溯源和原文高亮。
    """

    kind: str = Field(default="chunk", frozen=True)

    @property
    def document_id(self) -> str:
        return self.metadata.get("document_id", "")

    @property
    def section_id(self) -> str:
        return self.metadata.get("section_id", "")

    @property
    def page(self) -> int:
        return self.metadata.get("page", 0)

    @property
    def paragraph_index(self) -> int:
        return self.metadata.get("paragraph_index", 0)

    @property
    def char_start(self) -> int:
        return self.metadata.get("char_start", 0)

    @property
    def char_end(self) -> int:
        return self.metadata.get("char_end", 0)

    @property
    def token_count(self) -> int:
        return self.metadata.get("token_count", 0)