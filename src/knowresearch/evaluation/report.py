"""评估报告数据模型。"""

from __future__ import annotations

from pydantic import BaseModel, Field


class CaseMetrics(BaseModel):
    """单条评估样本的指标。"""

    id: str
    question: str
    category: str
    should_answer: bool
    answered: bool
    failure: bool
    expected_titles: list[str] = Field(default_factory=list)
    retrieved_titles: list[str] = Field(default_factory=list)
    hit: bool = False
    recall: float = 0.0
    reciprocal_rank: float = 0.0
    ndcg: float = 0.0
    citation_accuracy: float = 0.0
    citation_coverage: float = 0.0
    no_answer_correct: bool = False
    latency_ms: float = 0.0
    faithfulness: float | None = None
    relevancy: float | None = None
    error: str = ""


class EvaluationReport(BaseModel):
    """一次评估运行的汇总报告。"""

    total: int
    answerable: int
    unanswerable: int
    hit_rate: float = 0.0
    recall_at_k: float = 0.0
    mrr: float = 0.0
    ndcg_at_10: float = 0.0
    citation_accuracy: float = 0.0
    citation_coverage: float = 0.0
    no_answer_accuracy: float = 0.0
    failure_rate: float = 0.0
    avg_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    faithfulness: float | None = None
    relevancy: float | None = None
    cases: list[CaseMetrics] = Field(default_factory=list)
