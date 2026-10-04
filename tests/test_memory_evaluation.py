"""M7-4 记忆评估测试。"""

from knowresearch.core.schemas import MemoryCandidate
from knowresearch.evaluation import (
    MemoryEvalCase,
    MemoryEvalProfile,
    MemoryEvaluationRunner,
    load_memory_eval_cases,
)


def test_memory_evaluation_metrics():
    case = MemoryEvalCase(
        id="m1",
        user_id="user-a",
        transcript="我研究船舶水动力学，喜欢公式。",
        expected_profiles=[
            MemoryEvalProfile(dimension="research_field", keywords=["船舶水动力学"]),
            MemoryEvalProfile(dimension="output_preference", keywords=["公式"]),
        ],
    )

    def extract_fn(case):
        return [
            MemoryCandidate(
                user_id=case.user_id,
                content="用户研究船舶水动力学",
                metadata={"dimension": "research_field"},
            ),
            MemoryCandidate(
                user_id=case.user_id,
                content="用户喜欢表格",
                metadata={"dimension": "output_preference"},
            ),
        ]

    report = MemoryEvaluationRunner(extract_fn).run([case])

    assert report.total == 1
    assert report.precision == 0.5
    assert report.recall == 0.5
    assert report.f1 == 0.5
    assert report.false_positive_rate == 0.5


def test_memory_eval_dataset_loads():
    cases = load_memory_eval_cases("data/eval/memory_set.jsonl")
    assert len(cases) == 5
    assert all(case.expected_profiles for case in cases)
