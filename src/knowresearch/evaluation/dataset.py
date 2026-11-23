"""评估集加载和校验。"""

from __future__ import annotations

import json
from pathlib import Path

from knowresearch.core.schemas import AgentEvalCase, EvalCase


def load_eval_cases(path: str | Path) -> list[EvalCase]:
    """从 JSONL 文件加载评估样本。"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"评估集不存在: {path}")

    cases: list[EvalCase] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                cases.append(EvalCase(**json.loads(stripped)))
            except Exception as e:
                raise ValueError(
                    f"评估集第 {line_number} 行解析失败: {e}"
                ) from e
    return cases


def validate_eval_cases(cases: list[EvalCase]) -> list[str]:
    """返回评估集问题列表；如果评估集无效则抛异常。"""
    errors: list[str] = []
    ids = [case.id for case in cases]
    if len(ids) != len(set(ids)):
        errors.append("存在重复的评估样本 id")

    for case in cases:
        if not case.question.strip():
            errors.append(f"{case.id}: question 不能为空")
        if case.should_answer and not case.expected_document_titles:
            errors.append(f"{case.id}: 应回答样本缺少 expected_document_titles")
        if not case.should_answer and case.expected_document_titles:
            errors.append(f"{case.id}: 无答案样本不应设置 expected_document_titles")
    if errors:
        raise ValueError("; ".join(errors))
    return errors


def dataset_stats(cases: list[EvalCase]) -> dict:
    """返回评估集统计信息。"""
    by_category: dict[str, int] = {}
    by_difficulty: dict[str, int] = {}
    for case in cases:
        by_category[case.category] = by_category.get(case.category, 0) + 1
        by_difficulty[case.difficulty] = (
            by_difficulty.get(case.difficulty, 0) + 1
        )
    return {
        "total": len(cases),
        "answerable": sum(1 for case in cases if case.should_answer),
        "unanswerable": sum(1 for case in cases if not case.should_answer),
        "by_category": by_category,
        "by_difficulty": by_difficulty,
    }


def load_agent_eval_cases(path: str | Path) -> list[AgentEvalCase]:
    """从 JSONL 文件加载 Agent 能力评估样本。"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"评估集不存在: {path}")

    cases: list[AgentEvalCase] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                cases.append(AgentEvalCase(**json.loads(stripped)))
            except Exception as e:
                raise ValueError(
                    f"评估集第 {line_number} 行解析失败: {e}"
                ) from e
    return cases


def validate_agent_eval_cases(cases: list[AgentEvalCase]) -> list[str]:
    """返回 Agent 评估集问题列表；无效则抛异常。"""
    errors: list[str] = []
    ids = [case.id for case in cases]
    if len(ids) != len(set(ids)):
        errors.append("存在重复的评估样本 id")

    valid_intents = {
        "chitchat",
        "single_doc_qa",
        "comparison",
        "survey",
        "memory",
        "assist",
    }
    for case in cases:
        if not case.question.strip():
            errors.append(f"{case.id}: question 不能为空")
        if case.task == "routing":
            if case.expected_intent not in valid_intents:
                errors.append(f"{case.id}: expected_intent 非法")
        elif case.task == "multihop":
            if case.should_answer and not case.expected_document_titles:
                errors.append(f"{case.id}: 应回答多跳样本缺少 expected_document_titles")
            if case.should_answer and len(case.sub_questions) < 2:
                errors.append(f"{case.id}: 应回答多跳样本 sub_questions 至少 2 条")
            if case.should_answer and not case.answer_points:
                errors.append(f"{case.id}: 应回答多跳样本缺少 answer_points")
            if not case.should_answer and (
                case.expected_document_titles
                or case.sub_questions
                or case.answer_points
            ):
                errors.append(f"{case.id}: 无答案多跳样本不应设置文档/子问题/答案要点")
        else:
            errors.append(f"{case.id}: task 非法")
    if errors:
        raise ValueError("; ".join(errors))
    return errors


def agent_dataset_stats(cases: list[AgentEvalCase]) -> dict:
    """返回 Agent 评估集统计信息。"""
    by_task: dict[str, int] = {}
    by_difficulty: dict[str, int] = {}
    by_intent: dict[str, int] = {}
    for case in cases:
        by_task[case.task] = by_task.get(case.task, 0) + 1
        by_difficulty[case.difficulty] = (
            by_difficulty.get(case.difficulty, 0) + 1
        )
        if case.task == "routing":
            by_intent[case.expected_intent] = (
                by_intent.get(case.expected_intent, 0) + 1
            )
    return {
        "total": len(cases),
        "by_task": by_task,
        "by_difficulty": by_difficulty,
        "routing_by_intent": by_intent,
        "multihop_answerable": sum(
            1 for case in cases if case.task == "multihop" and case.should_answer
        ),
        "multihop_unanswerable": sum(
            1 for case in cases if case.task == "multihop" and not case.should_answer
        ),
    }
