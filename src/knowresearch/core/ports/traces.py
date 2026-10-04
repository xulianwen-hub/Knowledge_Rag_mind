"""轨迹端口：任务执行轨迹的存取，是技能沉淀的数据源。

每次 RAG 问答的完整链路（query 改写、召回、rerank、上下文组装、生成、后处理）
都记录成一条轨迹，供 M5 技能沉淀模块分析。
"""

from abc import ABC, abstractmethod
from typing import Any

from knowresearch.core.schemas import Record


class TraceStorePort(ABC):
    """任务轨迹存储端口。"""

    @abstractmethod
    def save(self, trace: Record) -> None:
        """保存一条任务轨迹。"""

    @abstractmethod
    def get(self, trace_id: str) -> Record | None:
        """按 ID 获取轨迹，不存在返回 None。"""

    @abstractmethod
    def list(self, filters: dict[str, Any] | None = None) -> list[Record]:
        """按条件筛选轨迹。filters 匹配 metadata 中的字段。"""

    @abstractmethod
    def delete(self, trace_id: str) -> None:
        """删除轨迹。"""