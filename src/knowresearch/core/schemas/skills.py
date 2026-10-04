"""技能相关 schema：候选技能与正式技能结构。"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


def _now() -> datetime:
    return datetime.now()


def _new_id() -> str:
    return uuid.uuid4().hex


class SkillCandidate(BaseModel):
    """待审核的技能候选。"""

    id: str = Field(default_factory=_new_id)
    user_id: str
    name: str
    description: str = ""
    trigger: str = ""
    steps: list[str] = Field(default_factory=list)
    retrieval_params: dict[str, Any] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    source_trace_ids: list[str] = Field(default_factory=list)
    evidence: str = ""
    confidence: float = 0.0
    status: str = "pending"
    proposed_action: str = "add"
    target_skill_id: str | None = None
    created_at: datetime = Field(default_factory=_now)
    reviewed_at: datetime | None = None
    reviewed_by: str = ""
    review_note: str = ""
