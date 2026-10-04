"""技能端口：从任务轨迹中沉淀的可复用技能的存取。

技能的本质是"触发条件 + 分析步骤 + 检索参数"的结构化记录，
用 Record 承载，kind = "skill"。
M5 接入时填充，当前阶段只定接口。
"""

from abc import ABC, abstractmethod
from typing import Any

from knowresearch.core.schemas import Record, SkillCandidate


class SkillStorePort(ABC):
    """技能存储端口。"""

    @abstractmethod
    def save(self, skill: Record) -> None:
        """保存技能。"""

    @abstractmethod
    def get(self, skill_id: str) -> Record | None:
        """按 ID 获取技能，不存在返回 None。"""

    @abstractmethod
    def search(self, query: str, top_k: int = 5) -> list[Record]:
        """按问题意图匹配技能。"""

    @abstractmethod
    def list_all(
        self,
        user_id: str | None = None,
        status: str | None = None,
        limit: int = 100,
    ) -> list[Record]:
        """列出技能，可按用户和状态过滤。"""

    @abstractmethod
    def update(self, skill_id: str, **kwargs: Any) -> None:
        """更新技能字段。"""

    @abstractmethod
    def delete(self, skill_id: str) -> None:
        """删除技能。"""

    @abstractmethod
    def update_status(
        self,
        skill_id: str,
        status: str,
        user_id: str | None = None,
    ) -> None:
        """更新技能状态（approved / disabled 等）。"""


class SkillCandidateStorePort(ABC):
    """候选技能存储端口。"""

    @abstractmethod
    def save(self, candidate: SkillCandidate) -> None:
        """保存或更新候选技能。"""

    @abstractmethod
    def get(self, candidate_id: str) -> SkillCandidate | None:
        """按 ID 获取候选技能。"""

    @abstractmethod
    def list(
        self,
        user_id: str | None = None,
        status: str | None = None,
        limit: int = 100,
    ) -> list[SkillCandidate]:
        """列出候选技能。"""

    @abstractmethod
    def update_review(
        self,
        candidate_id: str,
        status: str,
        reviewed_by: str = "",
        review_note: str = "",
        target_skill_id: str | None = None,
    ) -> None:
        """更新候选技能审核状态。"""

    @abstractmethod
    def exists_for_source(self, user_id: str, source: str) -> bool:
        """判断某个 trace/source 是否已抽取过技能。"""

    @abstractmethod
    def delete(self, candidate_id: str) -> None:
        """删除候选技能。"""


class SkillContextPort(ABC):
    """技能上下文端口：向 RAG 主链路提供技能匹配和注入内容。"""

    @abstractmethod
    def match(
        self,
        user_id: str,
        query: str,
        top_k: int = 3,
    ) -> list[Record]:
        """按问题意图匹配技能。"""

    @abstractmethod
    def build_context(
        self,
        user_id: str,
        query: str,
        max_chars: int = 1200,
    ) -> str:
        """构建技能注入上下文。"""


class SkillLifecyclePort(ABC):
    """技能生命周期端口：记录使用/反馈，供 QAEngine 写入。"""

    @abstractmethod
    def record_usage(
        self,
        skill_id: str,
        user_id: str,
        success: bool,
    ) -> None:
        """记录一次技能使用。"""

    @abstractmethod
    def record_feedback(
        self,
        skill_id: str,
        user_id: str,
        positive: bool,
    ) -> None:
        """记录一次用户反馈。"""
