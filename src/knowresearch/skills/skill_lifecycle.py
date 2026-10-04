"""技能使用反馈与生命周期管理。"""

from __future__ import annotations

from datetime import datetime

from knowresearch.core.ports import SkillLifecyclePort, SkillStorePort
from knowresearch.core.schemas import Record


class SkillLifecycleService(SkillLifecyclePort):
    """记录技能使用、反馈，并支持版本回滚。"""

    def __init__(self, skill_store: SkillStorePort):
        self.skill_store = skill_store

    def record_usage(
        self,
        skill_id: str,
        user_id: str,
        success: bool,
    ) -> None:
        skill = self._get(skill_id, user_id)
        if skill is None:
            return
        skill.metadata["usage_count"] = int(skill.metadata.get("usage_count", 0)) + 1
        if success:
            skill.metadata["success_count"] = (
                int(skill.metadata.get("success_count", 0)) + 1
            )
        skill.metadata["last_used_at"] = datetime.now().isoformat()
        self._update_skill(skill, user_id)

    def record_feedback(
        self,
        skill_id: str,
        user_id: str,
        positive: bool,
    ) -> None:
        skill = self._get(skill_id, user_id)
        if skill is None:
            return
        key = "feedback_up" if positive else "feedback_down"
        skill.metadata[key] = int(skill.metadata.get(key, 0)) + 1
        skill.metadata["last_feedback_at"] = datetime.now().isoformat()
        self._update_skill(skill, user_id)

    def disable(self, skill_id: str, user_id: str) -> None:
        self.skill_store.update_status(skill_id, "disabled", user_id=user_id)

    def enable(self, skill_id: str, user_id: str) -> None:
        self.skill_store.update_status(skill_id, "approved", user_id=user_id)

    def update_skill(
        self,
        skill_id: str,
        user_id: str,
        description: str | None = None,
        trigger: str | None = None,
        steps: list[str] | None = None,
        retrieval_params: dict | None = None,
        tags: list[str] | None = None,
    ) -> Record | None:
        skill = self._get(skill_id, user_id)
        if skill is None:
            return None

        history = list(skill.metadata.get("versions", []))
        history.append(self._snapshot(skill))
        version = int(skill.metadata.get("version", 1)) + 1

        if description is not None:
            skill.content = description
            skill.metadata["description"] = description
        if trigger is not None:
            skill.metadata["trigger"] = trigger
        if steps is not None:
            skill.metadata["steps"] = steps
        if retrieval_params is not None:
            skill.metadata["retrieval_params"] = retrieval_params
        if tags is not None:
            skill.metadata["tags"] = tags
        skill.metadata["version"] = version
        skill.metadata["versions"] = history
        skill.updated_at = datetime.now()
        self._update_skill(skill, user_id)
        return skill

    def rollback(
        self,
        skill_id: str,
        user_id: str,
        target_version: int,
    ) -> Record | None:
        skill = self._get(skill_id, user_id)
        if skill is None:
            return None

        history = list(skill.metadata.get("versions", []))
        target = next(
            (snapshot for snapshot in history if snapshot.get("version") == target_version),
            None,
        )
        if target is None:
            return None

        history.append(self._snapshot(skill))
        skill.content = target.get("content", skill.content)
        skill.metadata.update(
            {
                "description": target.get("description", ""),
                "trigger": target.get("trigger", ""),
                "steps": target.get("steps", []),
                "retrieval_params": target.get("retrieval_params", {}),
                "tags": target.get("tags", []),
                "version": target_version,
                "versions": history,
            }
        )
        skill.updated_at = datetime.now()
        self._update_skill(skill, user_id)
        return skill

    @staticmethod
    def _snapshot(skill: Record) -> dict:
        return {
            "version": int(skill.metadata.get("version", 1)),
            "content": skill.content,
            "description": skill.metadata.get("description", ""),
            "trigger": skill.metadata.get("trigger", ""),
            "steps": list(skill.metadata.get("steps", [])),
            "retrieval_params": dict(skill.metadata.get("retrieval_params", {})),
            "tags": list(skill.metadata.get("tags", [])),
        }

    def _get(self, skill_id: str, user_id: str) -> Record | None:
        skill = self.skill_store.get(skill_id)
        if skill is None:
            return None
        if skill.metadata.get("user_id") != user_id:
            return None
        return skill

    def _update_skill(self, skill: Record, user_id: str) -> None:
        metadata = dict(skill.metadata)
        metadata.pop("user_id", None)
        self.skill_store.update(skill.id, user_id=user_id, **metadata)
