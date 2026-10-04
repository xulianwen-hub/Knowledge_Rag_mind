"""运行 M7 评估集。"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from knowresearch.bootstrap import (
    build_llm,
    build_production_qa_engine,
    build_production_trace_store,
)
from knowresearch.evaluation import (
    EvaluationRunner,
    LLMJudge,
    load_eval_cases,
    render_markdown,
    save_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="运行 KnowResearch 评估集")
    parser.add_argument(
        "--eval-path",
        default="data/eval/qa_set.jsonl",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--judge", action="store_true")
    parser.add_argument("--output-dir", default="data/eval/reports")
    args = parser.parse_args()

    cases = load_eval_cases(args.eval_path)
    if args.limit is not None:
        cases = cases[: args.limit]

    qa_engine = build_production_qa_engine()
    trace_store = build_production_trace_store()
    judge = LLMJudge(build_llm()) if args.judge else None
    runner = EvaluationRunner(
        qa_engine=qa_engine,
        trace_store=trace_store,
        judge=judge,
    )
    report = runner.run(cases)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir)
    json_path = save_report(report, output_dir / f"eval_{stamp}.json")
    markdown_path = output_dir / f"eval_{stamp}.md"
    markdown_path.write_text(render_markdown(report), encoding="utf-8")

    print(f"报告已保存: {json_path}")
    print(f"Markdown: {markdown_path}")
    print(
        f"hit_rate={report.hit_rate} recall_at_k={report.recall_at_k} "
        f"mrr={report.mrr} citation_accuracy={report.citation_accuracy} "
        f"failure_rate={report.failure_rate}"
    )


if __name__ == "__main__":
    main()
