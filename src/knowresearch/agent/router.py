"""路由模块：保持向后兼容的薄封装。

真正的意图识别与路由逻辑见 `knowresearch.agent.intents`。
此文件继续导出旧接口（ComplexityRouter / RouteDecision），
同时新增 IntentRouter / IntentDecision / Intent。
"""

from __future__ import annotations

from knowresearch.agent.intents import (
    ComplexityRouter,
    Intent,
    IntentDecision,
    IntentRouter,
    LLMIntentClassifier,
    RouteDecision,
    RuleIntentClassifier,
)

__all__ = [
    "ComplexityRouter",
    "Intent",
    "IntentDecision",
    "IntentRouter",
    "LLMIntentClassifier",
    "RouteDecision",
    "RuleIntentClassifier",
]
