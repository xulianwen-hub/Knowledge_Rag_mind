"""M7-5 技能评估测试。"""

from knowresearch.core.schemas import SkillCandidate
from knowresearch.evaluation import (
    SkillEvalCase,
    SkillEvaluationRunner,
    load_skill_eval_cases,
)


def test_skill_evaluation_metrics():
    case = SkillEvalCase(
        id="s1",
        user_id="user-a",
        transcript="比较不同艇型优化方法",
        expected_skill_keywords=["艇型", "对比"],
        expected_replay_pass=True,
    )

    def extract_fn(case):
        return [
            SkillCandidate(
                user_id=case.user_id,
                name="艇型对比分析",
                description="比较艇型优化方法",
                trigger="用户要求比较不同艇型",
                tags=["艇型", "对比"],
            ),
            SkillCandidate(
                user_id=case.user_id,
                name="无关技能",
                description="其他",
            ),
        ]

    report = SkillEvaluationRunner(
        extract_fn,
        replay_fn=lambda candidate: True,
    ).run([case])

    assert report.total == 1
    assert report.candidate_hit_rate == 1.0
    assert report.candidate_precision == 0.5
    assert report.replay_accuracy == 1.0


def test_skill_eval_dataset_loads():
    cases = load_skill_eval_cases("data/eval/skill_set.jsonl")
    assert len(cases) == 5
    assert any(not case.expected_replay_pass for case in cases)
