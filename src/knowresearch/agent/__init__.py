"""Agent 层：ReAct 循环、复杂度路由与引擎。

这一层只做「编排」——动作提案走 ActionPlannerPort，工具执行复用
tools 层，状态落 WorkingMemoryPort，最终答案复用 AnswerGenerator /
CitationBuilder，因此不侵入原有 RAG 主链。
"""

from knowresearch.agent.engine import AgentEngine
from knowresearch.agent.parser import parse_decision
from knowresearch.agent.planner import JSONReActPlanner
from knowresearch.agent.react_agent import AgentRunResult, ReActAgent
from knowresearch.agent.router import (
    ComplexityRouter,
    Intent,
    IntentDecision,
    IntentRouter,
    LLMIntentClassifier,
    RouteDecision,
    RuleIntentClassifier,
)
from knowresearch.agent.reflector import (
    HeuristicReflector,
    LLMReflector,
    ReflectorPort,
    ReflectionResult,
)

__all__ = [
    "AgentEngine",
    "ReActAgent",
    "AgentRunResult",
    "JSONReActPlanner",
    "parse_decision",
    "ComplexityRouter",
    "RouteDecision",
    "Intent",
    "IntentDecision",
    "IntentRouter",
    "LLMIntentClassifier",
    "RuleIntentClassifier",
    "HeuristicReflector",
    "LLMReflector",
    "ReflectorPort",
    "ReflectionResult",
]
