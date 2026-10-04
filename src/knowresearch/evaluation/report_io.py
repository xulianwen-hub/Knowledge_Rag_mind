"""评估报告保存、加载和对比。"""

from __future__ import annotations

import json
from pathlib import Path

from knowresearch.evaluation.report import EvaluationReport


DEFAULT_METRICS = [
    "hit_rate",
    "recall_at_k",
    "mrr",
    "ndcg_at_10",
    "citation_accuracy",
    "citation_coverage",
    "no_answer_accuracy",
    "failure_rate",
    "avg_latency_ms",
    "p95_latency_ms",
]


def save_report(report: EvaluationReport, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        report.model_dump_json(indent=2),
        encoding="utf-8",
    )
    return path


def load_report(path: str | Path) -> EvaluationReport:
    path = Path(path)
    return EvaluationReport(**json.loads(path.read_text(encoding="utf-8")))


def compare_reports(
    current: EvaluationReport,
    baseline: EvaluationReport,
    metrics: list[str] | None = None,
) -> dict[str, float]:
    """返回 current - baseline 的指标差值。"""
    metrics = metrics or DEFAULT_METRICS
    deltas: dict[str, float] = {}
    for metric in metrics:
        current_value = getattr(current, metric, 0.0) or 0.0
        baseline_value = getattr(baseline, metric, 0.0) or 0.0
        deltas[metric] = round(current_value - baseline_value, 4)
    return deltas


def render_markdown(report: EvaluationReport) -> str:
    """把评估报告渲染为 Markdown。"""
    lines = [
        "# 评估报告",
        "",
        f"- 样本总数：{report.total}",
        f"- 应回答：{report.answerable}",
        f"- 无答案：{report.unanswerable}",
        "",
        "| 指标 | 值 |",
        "|---|---|",
        f"| Hit Rate | {report.hit_rate} |",
        f"| Recall@K | {report.recall_at_k} |",
        f"| MRR | {report.mrr} |",
        f"| nDCG@10 | {report.ndcg_at_10} |",
        f"| 引用准确率 | {report.citation_accuracy} |",
        f"| 引用覆盖率 | {report.citation_coverage} |",
        f"| 无答案准确率 | {report.no_answer_accuracy} |",
        f"| 失败率 | {report.failure_rate} |",
        f"| 平均延迟 ms | {report.avg_latency_ms} |",
        f"| P95 延迟 ms | {report.p95_latency_ms} |",
    ]
    if report.faithfulness is not None:
        lines.append(f"| 忠实度 | {report.faithfulness} |")
    if report.relevancy is not None:
        lines.append(f"| 相关性 | {report.relevancy} |")

    lines.extend(
        [
            "",
            "## 失败样本",
            "",
            "| ID | 问题 | 错误 |",
            "|---|---|---|",
        ]
    )
    for case in report.cases:
        if case.failure:
            lines.append(f"| {case.id} | {case.question} | {case.error} |")
    return "\n".join(lines) + "\n"
