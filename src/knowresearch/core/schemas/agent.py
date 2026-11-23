"""Agent 相关 schema：ReAct 的步骤、状态与决策。

设计说明（M8 复杂问题处理）：
- AgentStep：一次 ReAct 循环的 Thought / Action / Observation。
- AgentState：整个任务的中间状态，落到 WorkingMemoryPort（可恢复、可审计）。
- AgentDecision：一次 LLM 决策的结果，要么给出动作，要么给出最终答案。
"""

from __future__ import annotations

import uuid
from typing import Any

from pydantic import BaseModel, Field


class AgentStep(BaseModel):
    """ReAct 单步：思考 → 动作 → 观察。"""

    index: int
    thought: str = ""
    action: str = ""
    action_input: dict[str, Any] = Field(default_factory=dict)
    observation: str = ""
    success: bool = True


class AgentState(BaseModel):
    """ReAct 任务状态，持久化到工作记忆。"""

    task_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    goal: str = ""
    plan: list[str] = Field(default_factory=list)
    steps: list[AgentStep] = Field(default_factory=list)
    reflections: list[str] = Field(default_factory=list)
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    final_answer: str = ""
    status: str = "pending"
    stop_reason: str = ""


class AgentDecision(BaseModel):
    """一次 LLM 决策：给动作，或给最终答案。"""

    thought: str = ""
    action: str = ""
    action_input: dict[str, Any] = Field(default_factory=dict)
    final_answer: str = ""

    @property
    def is_final(self) -> bool:
        return bool(self.final_answer.strip())

    @property
    def has_action(self) -> bool:
        return bool(self.action.strip())
