"""Agent 能力评估集加载与路由准确率测试。"""

from pathlib import Path

from knowresearch.evaluation import (
    agent_dataset_stats,
    evaluate_routing,
    load_agent_eval_cases,
    validate_agent_eval_cases,
)


ROOT = Path(__file__).resolve().parents[1]


def test_load_and_validate_agent_eval_sets():
    routing = load_agent_eval_cases(ROOT / "data" / "eval" / "routing_set.jsonl")
    multihop = load_agent_eval_cases(ROOT / "data" / "eval" / "multihop_set.jsonl")

    assert len(routing) == 25
    assert len(multihop) == 25
    assert validate_agent_eval_cases(routing + multihop) == []

    stats = agent_dataset_stats(routing + multihop)
    assert stats["total"] == 50
    assert stats["by_task"]["routing"] == 25
    assert stats["by_task"]["multihop"] == 25


def test_routing_accuracy_on_rule_classifier():
    routing = load_agent_eval_cases(ROOT / "data" / "eval" / "routing_set.jsonl")
    report = evaluate_routing(routing)
    assert report["total"] == 25
    assert report["accuracy"] == 1.0
