"""CRAG 式反思器测试。"""

from knowresearch.agent.reflector import HeuristicReflector


def test_reflector_flags_empty_observation():
    reflector = HeuristicReflector()
    result = reflector.evaluate("PINN 是什么", "", "knowledge_base_search", {})
    assert result.sufficient is False
    assert result.quality == 0.0
    assert "改写查询" in result.feedback


def test_reflector_accepts_relevant_content():
    reflector = HeuristicReflector(min_quality=0.1)
    result = reflector.evaluate(
        "PINN 是什么",
        "PINN 是一种物理信息神经网络，常用于求解偏微分方程，保留了物理约束。",
        "knowledge_base_search",
        {"query": "PINN"},
    )
    assert result.sufficient is True
    assert result.quality >= 0.1
