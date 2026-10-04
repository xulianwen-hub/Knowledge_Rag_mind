"""M7-4 记忆系统评估。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from pydantic import BaseModel, Field

from knowresearch.core.schemas import MemoryCandidate


class MemoryEvalProfile(BaseModel):
    dimension: str
    keywords: list[str] = Field(default_factory=list)


class MemoryEvalCase(BaseModel):
    id: str
    user_id: str
    transcript: str
    expected_profiles: list[MemoryEvalProfile]


class MemoryCaseMetrics(BaseModel):
    id: str
    actual_profiles: int
    expected_profiles: int
    matched_profiles: int
    precision: float
    recall: float
    f1: float
    false_positive_rate: float


class MemoryEvalReport(BaseModel):
    total: int
    precision: float
    recall: float
    f1: float
    false_positive_rate: float
    cases: list[MemoryCaseMetrics]


class MemoryEvaluationRunner:
    def __init__(
        self,
        extract_fn: Callable[[MemoryEvalCase], list[MemoryCandidate]],
    ):
        self.extract_fn = extract_fn

    def run(self, cases: list[MemoryEvalCase]) -> MemoryEvalReport:
        case_metrics = []
        for case in cases:
            candidates = self.extract_fn(case)
            case_metrics.append(self._score_case(case, candidates))

        return MemoryEvalReport(
            total=len(case_metrics),
            precision=_mean([m.precision for m in case_metrics]),
            recall=_mean([m.recall for m in case_metrics]),
            f1=_mean([m.f1 for m in case_metrics]),
            false_positive_rate=_mean(
                [m.false_positive_rate for m in case_metrics]
            ),
            cases=case_metrics,
        )

    @staticmethod
    def _score_case(
        case: MemoryEvalCase,
        candidates: list[MemoryCandidate],
    ) -> MemoryCaseMetrics:
        matched_expected = 0
        for expected in case.expected_profiles:
            if any(
                _candidate_matches(candidate, expected)
                for candidate in candidates
            ):
                matched_expected += 1

        matched_actual = 0
        for candidate in candidates:
            if any(
                _candidate_matches(candidate, expected)
                for expected in case.expected_profiles
            ):
                matched_actual += 1

        actual_count = len(candidates)
        expected_count = len(case.expected_profiles)
        precision = matched_actual / actual_count if actual_count else 0.0
        recall = matched_expected / expected_count if expected_count else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        false_positive_rate = (
            (actual_count - matched_actual) / actual_count
            if actual_count
            else 0.0
        )
        return MemoryCaseMetrics(
            id=case.id,
            actual_profiles=actual_count,
            expected_profiles=expected_count,
            matched_profiles=matched_expected,
            precision=round(precision, 4),
            recall=round(recall, 4),
            f1=round(f1, 4),
            false_positive_rate=round(false_positive_rate, 4),
        )


def load_memory_eval_cases(path: str | Path) -> list[MemoryEvalCase]:
    path = Path(path)
    cases = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                cases.append(MemoryEvalCase(**json.loads(stripped)))
            except Exception as e:
                raise ValueError(
                    f"记忆评估集第 {line_number} 行解析失败: {e}"
                ) from e
    return cases


def _candidate_matches(
    candidate: MemoryCandidate,
    expected: MemoryEvalProfile,
) -> bool:
    dimension = candidate.metadata.get("dimension", "")
    if dimension != expected.dimension:
        return False
    if not expected.keywords:
        return True
    return any(keyword in candidate.content for keyword in expected.keywords)


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return round(sum(values) / len(values), 4)
