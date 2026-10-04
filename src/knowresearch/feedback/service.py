"""用户反馈处理服务。"""

from __future__ import annotations

from knowresearch.core.ports import FeedbackStorePort, SkillLifecyclePort
from knowresearch.core.schemas import UserFeedback


class FeedbackService:
    def __init__(
        self,
        store: FeedbackStorePort,
        skill_lifecycle: SkillLifecyclePort | None = None,
    ):
        self.store = store
        self.skill_lifecycle = skill_lifecycle

    def submit(self, feedback: UserFeedback) -> None:
        self.store.save(feedback)
        if (
            feedback.target_type == "skill"
            and feedback.target_id
            and self.skill_lifecycle is not None
        ):
            self.skill_lifecycle.record_feedback(
                feedback.target_id,
                feedback.user_id,
                positive=feedback.rating == "up",
            )
