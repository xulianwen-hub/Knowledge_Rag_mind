"""Agent 端口：动作提案（Action Planner）的抽象。

ReAct 循环本身依赖这个端口，而不依赖具体是「文本 JSON 解析」还是
「原生 function calling」。当前默认实现是 JSONReActPlanner；
将来模型支持原生 tool calling 时，只需新增一个实现，循环不变。
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from knowresearch.core.schemas import AgentDecision, ToolDefinition


class ActionPlannerPort(ABC):
    """根据目标、可用工具和历史步骤，给出下一步决策。"""

    @abstractmethod
    def propose(
        self,
        goal: str,
        tools: list[ToolDefinition],
        scratchpad: str,
        plan: list[str] | None = None,
    ) -> AgentDecision | None:
        """返回下一步决策；无法解析时返回 None（由调用方决定如何收尾）。"""
