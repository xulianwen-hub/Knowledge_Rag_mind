"""候选技能审核服务。"""

from __future__ import annotations

from knowresearch.core.ports import SkillCandidateStorePort, SkillStorePort
from knowresearch.core.schemas import Record, SkillCandidate


class SkillReviewService:
    """把候选技能审核并转化为正式技能。"""

    def __init__(
        self,
        candidate_store: SkillCandidateStorePort,
        skill_store: SkillStorePort,
    ):
        self.candidate_store = candidate_store
        self.skill_store = skill_store

    def list_pending(
        self,
        user_id: str,
        limit: int = 50,
    ) -> list[SkillCandidate]:
        return self.candidate_store.list(
            user_id=user_id,
            status="pending",
            limit=limit,
        )

    def approve(
        self,
        candidate_id: str,
        user_id: str,
        review_note: str = "",
    ) -> Record | None:
        candidate = self._get_pending(candidate_id, user_id)
        if candidate is None:
            return None

        skill = self._to_skill(candidate)
        self.skill_store.save(skill)
        self.candidate_store.update_review(
            candidate_id,
            status="approved",
            reviewed_by=user_id,
            review_note=review_note,
            target_skill_id=skill.id,
        )
        return skill

    def reject(
        self,
        candidate_id: str,
        user_id: str,
        review_note: str = "",
    ) -> bool:
        candidate = self._get_pending(candidate_id, user_id)
        if candidate is None:
            return False
        self.candidate_store.update_review(
            candidate_id,
            status="rejected",
            reviewed_by=user_id,
            review_note=review_note,
        )
        return True

    def edit_and_approve(
        self,
        candidate_id: str,
        user_id: str,
        name: str | None = None,
        description: str | None = None,
        trigger: str | None = None,
        steps: list[str] | None = None,
        review_note: str = "",
    ) -> Record | None:
        candidate = self._get_pending(candidate_id, user_id)
        if candidate is None:
            return None
        if name is not None:
            candidate.name = name
        if description is not None:
            candidate.description = description
        if trigger is not None:
            candidate.trigger = trigger
        if steps is not None:
            candidate.steps = steps
        self.candidate_store.save(candidate)
        return self.approve(candidate_id, user_id, review_note=review_note)

    def _get_pending(
        self,
        candidate_id: str,
        user_id: str,
    ) -> SkillCandidate | None:
        candidate = self.candidate_store.get(candidate_id)
        if candidate is None:
            return None
        if candidate.user_id != user_id or candidate.status != "pending":
            return None
        return candidate

    @staticmethod
    def _to_skill(candidate: SkillCandidate) -> Record:
        return Record(
            kind="skill",
            content=candidate.description or candidate.name,
            source=",".join(candidate.source_trace_ids),
            metadata={
                "user_id": candidate.user_id,
                "name": candidate.name,
                "description": candidate.description,
                "trigger": candidate.trigger,
                "steps": candidate.steps,
                "retrieval_params": candidate.retrieval_params,
                "tags": candidate.tags,
                "source_trace_ids": candidate.source_trace_ids,
                "status": "approved",
                "version": 1,
                "usage_count": 0,
                "success_count": 0,
            },
        )
