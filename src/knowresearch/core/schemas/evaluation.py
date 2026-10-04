"""评估集 schema。"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class EvalCase(BaseModel):
    """一条评估样本。"""

    id: str
    question: str
    category: Literal[
        "factual",
        "method",
        "survey",
        "comparison",
        "unanswerable",
        "memory",
        "skill",
    ]
    difficulty: Literal["easy", "medium", "hard"] = "medium"
    should_answer: bool = True
    expected_document_titles: list[str] = Field(default_factory=list)
    expected_keywords: list[str] = Field(default_factory=list)
    history: list[dict[str, str]] = Field(default_factory=list)
    notes: str = ""
