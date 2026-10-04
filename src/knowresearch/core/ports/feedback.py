"""反馈存储端口。"""

from __future__ import annotations

from abc import ABC, abstractmethod

from knowresearch.core.schemas import UserFeedback


class FeedbackStorePort(ABC):
    @abstractmethod
    def save(self, feedback: UserFeedback) -> None:
        """保存反馈。"""

    @abstractmethod
    def list(
        self,
        user_id: str | None = None,
        target_type: str | None = None,
        processed: bool | None = None,
        limit: int = 100,
    ) -> list[UserFeedback]:
        """列出反馈。"""
