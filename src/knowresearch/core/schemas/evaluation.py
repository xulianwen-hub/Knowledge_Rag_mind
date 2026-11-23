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


class AgentEvalCase(BaseModel):
    """Agent 能力评估样本：路由决策与多跳推理。

    task=routing 时校验 expected_intent；
    task=multihop 时校验 expected_document_titles / sub_questions / answer_points。
    """

    id: str
    task: Literal["routing", "multihop"]
    question: str
    expected_intent: str = ""
    should_retrieve: bool = True
    expected_document_titles: list[str] = Field(default_factory=list)
    sub_questions: list[str] = Field(default_factory=list)
    answer_points: list[str] = Field(default_factory=list)
    should_answer: bool = True
    difficulty: Literal["easy", "medium", "hard"] = "medium"
    history: list[dict[str, str]] = Field(default_factory=list)
    notes: str = ""
