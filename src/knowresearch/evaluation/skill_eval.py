"""M7-5 技能系统评估。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from pydantic import BaseModel, Field

from knowresearch.core.schemas import SkillCandidate


class SkillEvalCase(BaseModel):
    id: str
    user_id: str
    transcript: str
    expected_skill_keywords: list[str] = Field(default_factory=list)
    expected_replay_pass: bool = True


class SkillCaseMetrics(BaseModel):
    id: str
    extracted_count: int
    matched_count: int
    candidate_hit: bool
    candidate_precision: float
    replay_passed: bool
    replay_correct: bool


class SkillEvalReport(BaseModel):
    total: int
    candidate_hit_rate: float
    candidate_precision: float
    replay_accuracy: float
    cases: list[SkillCaseMetrics]


class SkillEvaluationRunner:
    def __init__(
        self,
        extract_fn: Callable[[SkillEvalCase], list[SkillCandidate]],
        replay_fn: Callable[[SkillCandidate], bool],
    ):
        self.extract_fn = extract_fn
        self.replay_fn = replay_fn

    def run(self, cases: list[SkillEvalCase]) -> SkillEvalReport:
        metrics = [self._score_case(case) for case in cases]
        return SkillEvalReport(
            total=len(metrics),
            candidate_hit_rate=_mean(
                [1.0 if metric.candidate_hit else 0.0 for metric in metrics]
            ),
            candidate_precision=_mean(
                [metric.candidate_precision for metric in metrics]
            ),
            replay_accuracy=_mean(
                [1.0 if metric.replay_correct else 0.0 for metric in metrics]
            ),
            cases=metrics,
        )

    def _score_case(self, case: SkillEvalCase) -> SkillCaseMetrics:
        candidates = self.extract_fn(case)
        matched = [
            candidate
            for candidate in candidates
            if _candidate_matches(candidate, case.expected_skill_keywords)
        ]
        candidate_hit = bool(matched)
        candidate_precision = len(matched) / len(candidates) if candidates else 0.0
        replay_passed = self.replay_fn(matched[0]) if matched else False
        return SkillCaseMetrics(
            id=case.id,
            extracted_count=len(candidates),
            matched_count=len(matched),
            candidate_hit=candidate_hit,
            candidate_precision=round(candidate_precision, 4),
            replay_passed=replay_passed,
            replay_correct=replay_passed == case.expected_replay_pass,
        )


def load_skill_eval_cases(path: str | Path) -> list[SkillEvalCase]:
    path = Path(path)
    cases = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                cases.append(SkillEvalCase(**json.loads(stripped)))
            except Exception as e:
                raise ValueError(
                    f"技能评估集第 {line_number} 行解析失败: {e}"
                ) from e
    return cases


def _candidate_matches(
    candidate: SkillCandidate,
    keywords: list[str],
) -> bool:
    if not keywords:
        return False
    text = " ".join(
        [
            candidate.name,
            candidate.description,
            candidate.trigger,
            " ".join(candidate.tags),
            " ".join(candidate.steps),
        ]
    )
    return any(keyword in text for keyword in keywords)


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return round(sum(values) / len(values), 4)
