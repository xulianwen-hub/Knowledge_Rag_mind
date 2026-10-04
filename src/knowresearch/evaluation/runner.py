"""评估运行器。"""

from __future__ import annotations

import math
import time
from typing import Callable

from knowresearch.core.generation.answer_generator import NO_ANSWER_TEXT
from knowresearch.core.qa_engine import SERVICE_UNAVAILABLE_TEXT
from knowresearch.core.schemas import EvalCase
from knowresearch.evaluation.judge import LLMJudge
from knowresearch.evaluation.report import CaseMetrics, EvaluationReport


class EvaluationRunner:
    def __init__(
        self,
        qa_engine,
        trace_store=None,
        judge: LLMJudge | None = None,
        top_k: int = 5,
        user_id: str = "eval-user",
        progress_callback: Callable[[int, int], None] | None = None,
    ):
        self.qa_engine = qa_engine
        self.trace_store = trace_store
        self.judge = judge
        self.top_k = top_k
        self.user_id = user_id
        self.progress_callback = progress_callback

    def run(self, cases: list[EvalCase]) -> EvaluationReport:
        case_metrics: list[CaseMetrics] = []
        for index, case in enumerate(cases, start=1):
            case_metrics.append(self.run_case(case))
            if self.progress_callback:
                self.progress_callback(index, len(cases))

        answerable = [m for m in case_metrics if m.should_answer]
        unanswerable = [m for m in case_metrics if not m.should_answer]
        latencies = [m.latency_ms for m in case_metrics]

        report = EvaluationReport(
            total=len(case_metrics),
            answerable=len(answerable),
            unanswerable=len(unanswerable),
            hit_rate=_mean([1.0 if m.hit else 0.0 for m in answerable]),
            recall_at_k=_mean([m.recall for m in answerable]),
            mrr=_mean([m.reciprocal_rank for m in answerable]),
            ndcg_at_10=_mean([m.ndcg for m in answerable]),
            citation_accuracy=_mean([m.citation_accuracy for m in answerable]),
            citation_coverage=_mean([m.citation_coverage for m in answerable]),
            no_answer_accuracy=_mean(
                [1.0 if m.no_answer_correct else 0.0 for m in unanswerable]
            ),
            failure_rate=_mean([1.0 if m.failure else 0.0 for m in case_metrics]),
            avg_latency_ms=_mean(latencies),
            p95_latency_ms=_percentile(latencies, 95),
            faithfulness=_optional_mean([m.faithfulness for m in case_metrics]),
            relevancy=_optional_mean([m.relevancy for m in case_metrics]),
            cases=case_metrics,
        )
        return report

    def run_case(self, case: EvalCase) -> CaseMetrics:
        started_at = time.perf_counter()
        answer = None
        failure = False
        error = ""
        try:
            answer = self.qa_engine.answer(
                case.question,
                session_id=f"eval-{case.id}",
                user_id=self.user_id,
            )
            failure = answer.text == SERVICE_UNAVAILABLE_TEXT
        except Exception as e:
            failure = True
            error = str(e)

        latency_ms = round((time.perf_counter() - started_at) * 1000, 2)
        retrieved_titles: list[str] = []
        answer_text = ""
        if answer is not None:
            answer_text = answer.text
            retrieved_titles = [
                citation.document_title
                for citation in answer.citations
                if citation.document_title
            ]
            if self.trace_store is not None and answer.trace_id:
                trace = self.trace_store.get(answer.trace_id)
                if trace is not None:
                    latency_ms = float(trace.metadata.get("duration_ms", latency_ms))

        expected = case.expected_document_titles
        retrieved_top = retrieved_titles[: self.top_k]
        matched = [title for title in expected if title in retrieved_top]

        hit = bool(matched) if case.should_answer else False
        recall = len(matched) / len(expected) if expected else 0.0
        reciprocal_rank = _reciprocal_rank(expected, retrieved_top)
        ndcg = _ndcg(expected, retrieved_top, k=10)
        citation_accuracy = (
            len([title for title in retrieved_titles if title in expected])
            / len(retrieved_titles)
            if retrieved_titles and case.should_answer
            else 0.0
        )
        citation_coverage = recall
        no_answer_correct = (
            not case.should_answer and answer_text == NO_ANSWER_TEXT
        )

        faithfulness = None
        relevancy = None
        if self.judge is not None and case.should_answer and not failure and answer is not None:
            score = self.judge.score(case.question, answer_text, expected)
            faithfulness = score.faithfulness
            relevancy = score.relevancy

        return CaseMetrics(
            id=case.id,
            question=case.question,
            category=case.category,
            should_answer=case.should_answer,
            answered=bool(answer_text) and answer_text != NO_ANSWER_TEXT,
            failure=failure,
            expected_titles=expected,
            retrieved_titles=retrieved_titles,
            hit=hit,
            recall=recall,
            reciprocal_rank=reciprocal_rank,
            ndcg=ndcg,
            citation_accuracy=citation_accuracy,
            citation_coverage=citation_coverage,
            no_answer_correct=no_answer_correct,
            latency_ms=latency_ms,
            faithfulness=faithfulness,
            relevancy=relevancy,
            error=error,
        )


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return round(sum(values) / len(values), 4)


def _optional_mean(values: list[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return _mean(present) if present else None


def _percentile(values: list[float], percentile: int) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(percentile / 100 * len(ordered)) - 1)
    return round(ordered[index], 2)


def _reciprocal_rank(expected: list[str], retrieved: list[str]) -> float:
    for rank, title in enumerate(retrieved, start=1):
        if title in expected:
            return round(1.0 / rank, 4)
    return 0.0


def _ndcg(expected: list[str], retrieved: list[str], k: int) -> float:
    if not expected:
        return 0.0
    dcg = 0.0
    for index, title in enumerate(retrieved[:k], start=1):
        if title in expected:
            dcg += 1.0 / math.log2(index + 1)
    ideal_hits = min(len(expected), k)
    idcg = sum(1.0 / math.log2(index + 1) for index in range(1, ideal_hits + 1))
    return round(dcg / idcg, 4) if idcg else 0.0
