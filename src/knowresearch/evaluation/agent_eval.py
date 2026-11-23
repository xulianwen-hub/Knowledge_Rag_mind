"""Agent 能力评估：路由准确率 + 多跳端到端检索。

路由集只验证「意图分类对不对」，不跑真实问答；多跳集转成 EvalCase 后
复用 EvaluationRunner 计算检索命中与引用指标。
"""

from __future__ import annotations

from typing import Any

from knowresearch.agent.router import IntentRouter
from knowresearch.core.schemas import AgentEvalCase, EvalCase
from knowresearch.evaluation.runner import EvaluationRunner


def evaluate_routing(cases: list[AgentEvalCase]) -> dict[str, Any]:
    """用规则意图路由器评估路由准确率，按意图给出细项。"""
    router = IntentRouter()
    total = 0
    correct = 0
    per_intent: dict[str, dict[str, int]] = {}
    errors: list[dict[str, str]] = []

    for case in cases:
        if case.task != "routing":
            continue
        total += 1
        predicted = router.route(case.question).intent.value
        per_intent.setdefault(
            case.expected_intent, {"correct": 0, "total": 0}
        )["total"] += 1
        if predicted == case.expected_intent:
            correct += 1
            per_intent[case.expected_intent]["correct"] += 1
        else:
            errors.append(
                {
                    "id": case.id,
                    "question": case.question,
                    "expected": case.expected_intent,
                    "predicted": predicted,
                }
            )

    return {
        "total": total,
        "correct": correct,
        "accuracy": round(correct / total, 4) if total else 0.0,
        "per_intent": per_intent,
        "errors": errors,
    }


def to_eval_cases(cases: list[AgentEvalCase]) -> list[EvalCase]:
    """把多跳 Agent 样本转成现有 EvalCase，复用 EvaluationRunner。"""
    result: list[EvalCase] = []
    for case in cases:
        if case.task != "multihop":
            continue
        result.append(
            EvalCase(
                id=case.id,
                question=case.question,
                category="comparison" if case.should_answer else "unanswerable",
                difficulty=case.difficulty,
                should_answer=case.should_answer,
                expected_document_titles=case.expected_document_titles,
                expected_keywords=[],
                history=case.history,
                notes=case.notes,
            )
        )
    return result


def run_multihop(
    cases: list[AgentEvalCase],
    qa_engine,
    trace_store=None,
    judge=None,
    top_k: int = 5,
):
    """把多跳集交给 EvaluationRunner 跑端到端检索与引用指标。"""
    runner = EvaluationRunner(
        qa_engine=qa_engine,
        trace_store=trace_store,
        judge=judge,
        top_k=top_k,
    )
    return runner.run(to_eval_cases(cases))
