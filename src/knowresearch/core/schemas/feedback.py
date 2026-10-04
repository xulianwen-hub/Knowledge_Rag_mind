"""用户反馈 schema。"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


def _now() -> datetime:
    return datetime.now()


class UserFeedback(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    user_id: str
    trace_id: str = ""
    target_type: Literal["answer", "memory", "skill"] = "answer"
    target_id: str = ""
    rating: Literal["up", "down"]
    correction: str = ""
    created_at: datetime = Field(default_factory=_now)
    processed: bool = False
