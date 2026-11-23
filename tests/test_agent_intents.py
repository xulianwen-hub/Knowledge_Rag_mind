"""意图路由与任务分解测试。"""

from knowresearch.agent.router import (
    ComplexityRouter,
    Intent,
    IntentRouter,
)


def test_intent_router_chitchat():
    decision = IntentRouter().route("你好，你是谁？")
    assert decision.intent == Intent.CHITCHAT
    assert decision.complex is False


def test_intent_router_single_doc_qa():
    decision = IntentRouter().route("PINN 是什么？")
    assert decision.intent == Intent.SINGLE_DOC_QA
    assert decision.complex is False


def test_intent_router_comparison_decomposes():
    decision = IntentRouter().route("比较 A 和 B 的方法差异")
    assert decision.intent == Intent.COMPARISON
    assert decision.complex is True
    assert len(decision.sub_queries) == 3


def test_intent_router_survey_decomposes():
    decision = IntentRouter().route("调研 LLM 长上下文的最新进展")
    assert decision.intent == Intent.SURVEY
    assert len(decision.sub_queries) == 3


def test_intent_router_memory_and_assist():
    assert IntentRouter().route("我上个月读过的 MoE 论文有哪些").intent == Intent.MEMORY
    assert IntentRouter().route("生成这篇论文的 bibtex").intent == Intent.ASSIST


def test_complexity_router_backward_compat():
    router = ComplexityRouter()
    assert router.is_complex("PINN 是什么？") is False
    assert router.is_complex("比较 A 和 B 的方法差异") is True
