"""候选画像审核服务。"""

from __future__ import annotations

from knowresearch.core.ports import (
    LongTermMemoryPort,
    MemoryCandidateStorePort,
)
from knowresearch.core.schemas import MemoryCandidate, Record


class ProfileReviewService:
    """把候选画像审核并转化为正式长期记忆。"""

    def __init__(
        self,
        candidate_store: MemoryCandidateStorePort,
        long_term_memory: LongTermMemoryPort,
    ):
        self.candidate_store = candidate_store
        self.long_term_memory = long_term_memory

    def list_pending(
        self,
        user_id: str,
        limit: int = 50,
    ) -> list[MemoryCandidate]:
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

        dimension = str(candidate.metadata.get("dimension", "general"))
        profile = self.long_term_memory.upsert_profile(
            user_id=user_id,
            content=candidate.content,
            dimension=dimension,
            source=candidate.source,
            confidence=candidate.confidence,
            explicit=False,
            metadata=candidate.metadata,
        )
        self.candidate_store.update_review(
            candidate_id,
            status="approved",
            reviewed_by=user_id,
            review_note=review_note,
            target_memory_id=profile.id,
        )
        return profile

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
        content: str,
        metadata: dict | None = None,
        review_note: str = "",
    ) -> Record | None:
        candidate = self._get_pending(candidate_id, user_id)
        if candidate is None:
            return None
        candidate.content = content
        if metadata:
            candidate.metadata.update(metadata)
        self.candidate_store.save(candidate)
        return self.approve(candidate_id, user_id, review_note=review_note)

    def _get_pending(
        self,
        candidate_id: str,
        user_id: str,
    ) -> MemoryCandidate | None:
        candidate = self.candidate_store.get(candidate_id)
        if candidate is None:
            return None
        if candidate.user_id != user_id or candidate.status != "pending":
            return None
        return candidate
