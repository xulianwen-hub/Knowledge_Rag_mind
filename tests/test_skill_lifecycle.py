"""M5-5 技能使用反馈与生命周期测试。"""

from knowresearch.adapters.skills import PostgresSkillStore
from knowresearch.core.schemas import Record
from knowresearch.skills import SkillLifecycleService


def _make_skill_store(tmp_path) -> tuple[PostgresSkillStore, Record]:
    store = PostgresSkillStore(f"sqlite:///{tmp_path / 'skills.db'}")
    skill = Record(
        kind="skill",
        content="第一版技能说明",
        metadata={
            "user_id": "user-a",
            "name": "艇型对比分析",
            "description": "第一版技能说明",
            "trigger": "比较艇型",
            "steps": ["步骤1"],
            "retrieval_params": {"top_k": 5},
            "tags": ["艇型"],
            "status": "approved",
            "version": 1,
            "usage_count": 0,
            "success_count": 0,
        },
    )
    store.save(skill)
    return store, skill


def test_record_usage_and_feedback(tmp_path):
    store, skill = _make_skill_store(tmp_path)
    service = SkillLifecycleService(store)

    service.record_usage(skill.id, "user-a", success=True)
    service.record_usage(skill.id, "user-a", success=False)
    service.record_feedback(skill.id, "user-a", positive=True)
    service.record_feedback(skill.id, "user-a", positive=False)

    got = store.get(skill.id)
    assert got.metadata["usage_count"] == 2
    assert got.metadata["success_count"] == 1
    assert got.metadata["feedback_up"] == 1
    assert got.metadata["feedback_down"] == 1
    assert got.metadata["last_used_at"]
    assert got.metadata["last_feedback_at"]


def test_disable_and_enable(tmp_path):
    store, skill = _make_skill_store(tmp_path)
    service = SkillLifecycleService(store)

    service.disable(skill.id, "user-a")
    assert store.get(skill.id).metadata["status"] == "disabled"

    service.enable(skill.id, "user-a")
    assert store.get(skill.id).metadata["status"] == "approved"


def test_update_version_and_rollback(tmp_path):
    store, skill = _make_skill_store(tmp_path)
    service = SkillLifecycleService(store)

    updated = service.update_skill(
        skill.id,
        "user-a",
        description="第二版技能说明",
        steps=["步骤1", "步骤2"],
        retrieval_params={"top_k": 10},
    )

    assert updated is not None
    assert updated.metadata["version"] == 2
    assert updated.content == "第二版技能说明"
    assert updated.metadata["versions"][0]["version"] == 1

    rolled_back = service.rollback(skill.id, "user-a", target_version=1)
    assert rolled_back is not None
    assert rolled_back.metadata["version"] == 1
    assert rolled_back.content == "第一版技能说明"
    assert rolled_back.metadata["steps"] == ["步骤1"]


def test_lifecycle_user_isolation(tmp_path):
    store, skill = _make_skill_store(tmp_path)
    service = SkillLifecycleService(store)

    service.record_usage(skill.id, "user-b", success=True)
    assert store.get(skill.id).metadata["usage_count"] == 0
