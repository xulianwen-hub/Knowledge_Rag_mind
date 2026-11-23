"""意图识别与任务分解。

面试叙事：从「复杂度两档路由」升级为「六类意图路由」，
每一类意图对应不同的处理链路与成本策略：
- chitchat：不检索，直接对话
- single_doc_qa：走快路径（RAG 检索 + 引用生成）
- comparison：拆解后走 ReAct 多步检索 + 综合推理
- survey：拆解后走 ReAct，多轮本地 + 外部检索
- memory：走长期阅读/画像记忆工具
- assist：BibTeX / 推荐等辅助产出
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum


class Intent(str, Enum):
    """六类用户意图。"""

    CHITCHAT = "chitchat"
    SINGLE_DOC_QA = "single_doc_qa"
    COMPARISON = "comparison"
    SURVEY = "survey"
    MEMORY = "memory"
    ASSIST = "assist"


@dataclass
class IntentDecision:
    """意图路由结果。"""

    intent: Intent = Intent.SINGLE_DOC_QA
    complex: bool = False
    reason: str = ""
    sub_queries: list[str] = field(default_factory=list)


class RuleIntentClassifier:
    """零成本、可解释、可测试的规则意图分类器。

    生产环境可替换为 LLMIntentClassifier（同一接口），
    RuleIntentClassifier 作为兜底，避免 LLM 调用失败时整条链路不可用。
    """

    CHITCHAT_MARKERS = (
        "你好",
        "您好",
        "谢谢",
        "你是谁",
        "你能做什么",
        "介绍一下",
        "嗨",
        "hello",
        "hi",
        "thanks",
    )
    COMPARISON_MARKERS = (
        "比较",
        "对比",
        "区别",
        "差异",
        "异同",
        "优缺点",
        "不同",
        "分别",
        "哪个更好",
        "相比之下",
        "compare",
        "versus",
        " vs ",
    )
    SURVEY_MARKERS = (
        "调研",
        "综述",
        "进展",
        "最新",
        "概览",
        "有哪些方法",
        "发展趋势",
        "研究现状",
        "前沿",
        "survey",
        "overview",
    )
    MEMORY_MARKERS = (
        "我读过",
        "我上次",
        "我之前",
        "最近读过",
        "阅读历史",
        "阅读记录",
        "我看过",
        "读过的",
        "历史论文",
        "记得",
    )
    ASSIST_MARKERS = (
        "bibtex",
        "参考文献",
        "引用格式",
        "生成引用",
        "推荐相关论文",
        "推荐论文",
        "相关文献推荐",
    )

    def classify(self, question: str) -> tuple[Intent, str]:
        text = question.strip().lower()
        if not text:
            return Intent.CHITCHAT, "empty"

        if any(marker in text for marker in self.CHITCHAT_MARKERS):
            return Intent.CHITCHAT, "chitchat_marker"
        if any(marker in text for marker in self.ASSIST_MARKERS):
            return Intent.ASSIST, "assist_marker"
        if any(marker in text for marker in self.MEMORY_MARKERS):
            return Intent.MEMORY, "memory_marker"
        if any(marker in text for marker in self.SURVEY_MARKERS):
            return Intent.SURVEY, "survey_marker"
        if any(marker in text for marker in self.COMPARISON_MARKERS):
            return Intent.COMPARISON, "comparison_marker"
        if len(question.strip()) > 40 or question.count("？") + question.count("?") > 1:
            return Intent.COMPARISON, "complex_shape"
        return Intent.SINGLE_DOC_QA, "default"


class LLMIntentClassifier:
    """用 LLM 做意图分类，失败时降级到规则分类器。"""

    INTENT_LABELS = [intent.value for intent in Intent]

    def __init__(self, llm, fallback: RuleIntentClassifier | None = None) -> None:
        self.llm = llm
        self.fallback = fallback or RuleIntentClassifier()

    def classify(self, question: str) -> tuple[Intent, str]:
        try:
            prompt = (
                "把用户问题分为以下六类之一："
                f"{', '.join(self.INTENT_LABELS)}。\n"
                f"问题：{question}\n"
                "只输出一个类别名，不要解释。"
            )
            raw = self.llm.generate(prompt).strip().lower()
            for intent in Intent:
                if intent.value in raw:
                    return intent, "llm"
        except Exception:
            pass
        return self.fallback.classify(question)


class IntentRouter:
    """意图路由 + 查询分解。

    router_enabled=False 时只做「是否复杂」判断，行为与旧 ComplexityRouter 对齐；
    router_enabled=True 时输出完整意图与分解后的子问题。
    """

    def __init__(
        self,
        classifier: RuleIntentClassifier | None = None,
        max_simple_length: int = 40,
    ) -> None:
        self.classifier = classifier or RuleIntentClassifier()
        self.max_simple_length = max_simple_length

    def route(self, question: str) -> IntentDecision:
        intent, reason = self.classifier.classify(question)
        complex_ = intent in {Intent.COMPARISON, Intent.SURVEY, Intent.MEMORY}
        if not complex_ and len(question.strip()) > self.max_simple_length:
            intent, reason = Intent.COMPARISON, "length"
            complex_ = True
        sub_queries = self.decompose(question, intent)
        return IntentDecision(
            intent=intent,
            complex=complex_,
            reason=reason,
            sub_queries=sub_queries,
        )

    def decompose(self, question: str, intent: Intent) -> list[str]:
        """按意图生成子目标，作为 ReAct 的执行计划。"""
        if intent == Intent.COMPARISON:
            return [
                f"分别检索并概括「{question}」涉及的各个对象的定义、方法与结论",
                f"提炼「{question}」中各对象之间的方法差异与优缺点",
                "综合上面的证据，给出对比结论",
            ]
        if intent == Intent.SURVEY:
            return [
                f"先检索「{question}」的代表性工作与核心方法",
                f"再检索「{question}」的最新进展与关键结论",
                "归纳该主题的挑战、趋势与未来方向",
            ]
        return []


class ComplexityRouter:
    """兼容旧接口的复杂度路由，内部委托 IntentRouter。"""

    def __init__(
        self,
        markers: tuple[str, ...] | None = None,
        max_simple_length: int = 40,
    ) -> None:
        self.max_simple_length = max_simple_length
        self._intent_router = IntentRouter(max_simple_length=max_simple_length)

    def route(self, question: str):
        decision = self._intent_router.route(question)
        return RouteDecision(complex=decision.complex, reason=decision.reason)

    def is_complex(self, question: str) -> bool:
        return self.route(question).complex


@dataclass
class RouteDecision:
    """兼容旧接口的路由结果。"""

    complex: bool
    reason: str = ""
