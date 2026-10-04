"""M7-3 实验框架：对多组模型/检索配置跑同一评估集。"""

from __future__ import annotations

from typing import Callable

from pydantic import BaseModel, Field

from knowresearch.core.schemas import EvalCase
from knowresearch.evaluation.report import EvaluationReport
from knowresearch.evaluation.runner import EvaluationRunner


class ExperimentConfig(BaseModel):
    """一组可复现实验配置。"""

    name: str
    embedding_model: str
    reranker_model: str = ""
    use_reranker: bool = True
    top_k: int = 5
    notes: str = ""


class ExperimentResult(BaseModel):
    config: ExperimentConfig
    report: EvaluationReport


class ExperimentRunner:
    """对多组配置复用同一个评估集。"""

    def __init__(
        self,
        engine_factory: Callable[[ExperimentConfig], object],
        trace_store_factory: Callable[[ExperimentConfig], object] | None = None,
        judge=None,
    ):
        self.engine_factory = engine_factory
        self.trace_store_factory = trace_store_factory
        self.judge = judge

    def run(
        self,
        configs: list[ExperimentConfig],
        cases: list[EvalCase],
    ) -> list[ExperimentResult]:
        results: list[ExperimentResult] = []
        for config in configs:
            engine = self.engine_factory(config)
            trace_store = (
                self.trace_store_factory(config)
                if self.trace_store_factory
                else None
            )
            runner = EvaluationRunner(
                qa_engine=engine,
                trace_store=trace_store,
                judge=self.judge,
                top_k=config.top_k,
            )
            results.append(
                ExperimentResult(
                    config=config,
                    report=runner.run(cases),
                )
            )
        return results


def render_experiment_markdown(results: list[ExperimentResult]) -> str:
    """把多组实验结果渲染成对比表。"""
    lines = [
        "# 模型 / 检索配置对比",
        "",
        "| 配置 | Embedding | Reranker | HitRate | Recall@K | MRR | nDCG@10 | 引用准确率 | 平均延迟 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for result in results:
        report = result.report
        lines.append(
            "| {name} | {emb} | {rerank} | {hit} | {recall} | {mrr} | "
            "{ndcg} | {cite} | {latency} |".format(
                name=result.config.name,
                emb=result.config.embedding_model,
                rerank=result.config.reranker_model
                if result.config.use_reranker
                else "off",
                hit=report.hit_rate,
                recall=report.recall_at_k,
                mrr=report.mrr,
                ndcg=report.ndcg_at_10,
                cite=report.citation_accuracy,
                latency=report.avg_latency_ms,
            )
        )
    return "\n".join(lines) + "\n"
