"""运行 Agent 能力评估：路由准确率 + 多跳端到端检索。"""

from __future__ import annotations

import argparse

from knowresearch.bootstrap import (
    build_llm,
    build_production_qa_engine,
    build_production_trace_store,
)
from knowresearch.evaluation import (
    LLMJudge,
    evaluate_routing,
    load_agent_eval_cases,
    run_multihop,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="运行 Agent 能力评估集")
    parser.add_argument(
        "--routing-path",
        default="data/eval/routing_set.jsonl",
    )
    parser.add_argument(
        "--multihop-path",
        default="data/eval/multihop_set.jsonl",
    )
    parser.add_argument("--judge", action="store_true")
    parser.add_argument("--skip-multihop", action="store_true")
    args = parser.parse_args()

    routing_cases = load_agent_eval_cases(args.routing_path)
    routing_report = evaluate_routing(routing_cases)
    print("== 路由集 ==")
    print(
        f"total={routing_report['total']} correct={routing_report['correct']} "
        f"accuracy={routing_report['accuracy']}"
    )
    if routing_report["errors"]:
        print("误分类样本：")
        for error in routing_report["errors"]:
            print(
                f"- {error['id']} expected={error['expected']} "
                f"predicted={error['predicted']} question={error['question']}"
            )

    if args.skip_multihop:
        return

    multihop_cases = load_agent_eval_cases(args.multihop_path)
    qa_engine = build_production_qa_engine()
    trace_store = build_production_trace_store()
    judge = LLMJudge(build_llm()) if args.judge else None
    report = run_multihop(
        multihop_cases,
        qa_engine,
        trace_store=trace_store,
        judge=judge,
    )
    print("\n== 多跳集 ==")
    print(
        f"total={report.total} hit_rate={report.hit_rate} "
        f"recall_at_k={report.recall_at_k} mrr={report.mrr} "
        f"citation_accuracy={report.citation_accuracy} "
        f"no_answer_accuracy={report.no_answer_accuracy} "
        f"failure_rate={report.failure_rate}"
    )


if __name__ == "__main__":
    main()
