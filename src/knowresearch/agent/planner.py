"""JSON 版 ReAct 动作提案器。

用一次 LLM 调用同时产出 Thought 和 Action（或 Final Answer），
不拆分「思考节点 / 行动节点」——这是 ReAct 在结构化输出下的常规做法，
也避免了每步两次 LLM 调用的开销。
"""

from __future__ import annotations

from knowresearch.agent.parser import parse_decision
from knowresearch.agent.prompt import build_react_messages
from knowresearch.core.ports import ActionPlannerPort, LLMProvider
from knowresearch.core.schemas import AgentDecision, ToolDefinition


class JSONReActPlanner(ActionPlannerPort):
    """提示词 + JSON 解析的 ReAct 提案器。"""

    def __init__(
        self,
        llm: LLMProvider,
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> None:
        self.llm = llm
        self.temperature = temperature
        self.max_tokens = max_tokens

    def propose(
        self,
        goal: str,
        tools: list[ToolDefinition],
        scratchpad: str,
        plan: list[str] | None = None,
    ) -> AgentDecision | None:
        messages = build_react_messages(goal, tools, scratchpad, plan=plan)
        try:
            raw = self.llm.generate_with_messages(
                messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        except Exception:
            return None
        return parse_decision(raw)
