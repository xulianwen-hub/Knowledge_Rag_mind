"""M7-8 反馈闭环测试。"""

from knowresearch.adapters.feedback import PostgresFeedbackStore
from knowresearch.core.schemas import UserFeedback
from knowresearch.feedback import FeedbackService


class _FakeLifecycle:
    def __init__(self):
        self.feedback = []

    def record_feedback(self, skill_id, user_id, positive):
        self.feedback.append((skill_id, user_id, positive))


def test_feedback_service_records_skill_feedback():
    store = PostgresFeedbackStore("sqlite:///:memory:")
    lifecycle = _FakeLifecycle()
    service = FeedbackService(store, lifecycle)
    feedback = UserFeedback(
        user_id="user-a",
        target_type="skill",
        target_id="skill-1",
        rating="up",
    )

    service.submit(feedback)

    assert store.list(user_id="user-a")[0].id == feedback.id
    assert lifecycle.feedback == [("skill-1", "user-a", True)]


def test_feedback_store_filters(tmp_path):
    store = PostgresFeedbackStore(f"sqlite:///{tmp_path / 'feedback.db'}")
    store.save(UserFeedback(user_id="user-a", rating="up", target_type="answer"))
    store.save(UserFeedback(user_id="user-b", rating="down", target_type="skill"))

    assert len(store.list(user_id="user-b")) == 1
    assert len(store.list(target_type="skill")) == 1
    assert len(store.list(processed=False)) == 2
