"""记忆相关 schema：候选记忆与注入上下文。"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


def _now() -> datetime:
    return datetime.now()


def _new_id() -> str:
    return uuid.uuid4().hex


class MemoryCandidate(BaseModel):
    """待审核的候选记忆。当前只用于 profile。"""

    id: str = Field(default_factory=_new_id)
    user_id: str
    kind: str = "profile"
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    source: str = ""
    evidence: str = ""
    confidence: float = 0.0
    status: str = "pending"
    proposed_action: str = "add"
    target_memory_id: str | None = None
    created_at: datetime = Field(default_factory=_now)
    reviewed_at: datetime | None = None
    reviewed_by: str = ""
    review_note: str = ""
