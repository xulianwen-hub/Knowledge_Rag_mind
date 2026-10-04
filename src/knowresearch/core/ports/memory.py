"""记忆端口：短期 / 工作 / 长期记忆的抽象接口。

三层记忆的职责不同：
- SessionStore：对话历史，有 TTL，过期自动清理
- WorkingMemory：当前任务的中间状态（ReAct 多步推理的 scratchpad）
- LongTermMemory：跨会话的持久知识（用户画像、术语表、实体关系）
"""

from abc import ABC, abstractmethod
from typing import Any

from knowresearch.core.schemas import MemoryCandidate, Record


class SessionStorePort(ABC):
    """短期记忆端口：会话级对话历史。"""

    @abstractmethod
    def save_messages(
        self,
        session_id: str,
        messages: list[dict[str, str]],
        ttl_seconds: int = 3600,
        summary: str = "",
    ) -> None:
        """保存会话消息。ttl_seconds 后自动过期。

        Args:
            summary: 历史对话摘要（溢出滑动窗口的早期对话压缩结果），
                     为空字符串表示无摘要。
        """

    @abstractmethod
    def load_messages(self, session_id: str) -> list[dict[str, str]]:
        """加载会话消息。"""

    @abstractmethod
    def load_summary(self, session_id: str) -> str:
        """加载会话历史摘要，不存在返回空字符串。"""

    @abstractmethod
    def delete_session(self, session_id: str) -> None:
        """删除整个会话。"""


class WorkingMemoryPort(ABC):
    """工作记忆端口：任务执行过程中的中间状态。"""

    @abstractmethod
    def save_state(self, task_id: str, state: dict[str, Any]) -> None:
        """保存任务状态。"""

    @abstractmethod
    def load_state(self, task_id: str) -> dict[str, Any] | None:
        """加载任务状态，不存在返回 None。"""

    @abstractmethod
    def update_state(self, task_id: str, **kwargs: Any) -> None:
        """更新任务状态的部分字段。"""

    @abstractmethod
    def delete_task(self, task_id: str) -> None:
        """删除任务状态。"""


class LongTermMemoryPort(ABC):
    """长期记忆端口：跨会话的持久知识。

    用 Record 承载，kind 区分记忆类型。当前阶段只实现 profile。
    user_id 用于多用户隔离；没传时由适配器使用默认用户。
    """

    @abstractmethod
    def add(self, memory: Record, user_id: str | None = None) -> None:
        """写入一条长期记忆。"""

    @abstractmethod
    def search(
        self,
        query: str,
        top_k: int = 5,
        user_id: str | None = None,
    ) -> list[Record]:
        """按查询检索相关记忆。"""

    @abstractmethod
    def list(
        self,
        user_id: str | None = None,
        kind: str | None = None,
    ) -> list[Record]:
        """列出用户的长期记忆。"""

    @abstractmethod
    def update(
        self,
        memory_id: str,
        user_id: str | None = None,
        **kwargs: Any,
    ) -> None:
        """更新记忆的 metadata 字段。"""

    @abstractmethod
    def delete(self, memory_id: str, user_id: str | None = None) -> None:
        """删除记忆。"""


class MemoryCandidateStorePort(ABC):
    """候选记忆存储端口：保存待审核画像并更新审核状态。"""

    @abstractmethod
    def save(self, candidate: MemoryCandidate) -> None:
        """保存或更新候选记忆。"""

    @abstractmethod
    def get(self, candidate_id: str) -> MemoryCandidate | None:
        """按 ID 获取候选记忆。"""

    @abstractmethod
    def list(
        self,
        user_id: str | None = None,
        status: str | None = None,
        limit: int = 100,
    ) -> list[MemoryCandidate]:
        """列出候选记忆。"""

    @abstractmethod
    def update_review(
        self,
        candidate_id: str,
        status: str,
        reviewed_by: str = "",
        review_note: str = "",
        target_memory_id: str | None = None,
    ) -> None:
        """更新候选记忆的审核状态。"""

    @abstractmethod
    def exists_for_source(self, user_id: str, source: str) -> bool:
        """判断某个 trace/source 是否已经抽取过候选记忆。"""


class MemoryContextPort(ABC):
    """记忆上下文端口：向 RAG 主链路提供画像改写和上下文注入。"""

    @abstractmethod
    def rewrite_query(self, user_id: str, query: str) -> str:
        """用用户画像改写或扩展查询。"""

    @abstractmethod
    def build_context(self, user_id: str, query: str, max_chars: int = 800) -> str:
        """构建注入 prompt 的画像上下文。"""
