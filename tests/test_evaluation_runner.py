"""M7-2 评估运行器和报告测试。"""

from knowresearch.core.generation.answer_generator import NO_ANSWER_TEXT
from knowresearch.core.ports import LLMProvider
from knowresearch.core.retrieval.citation import Citation
from knowresearch.core.schemas import EvalCase, Record
from knowresearch.evaluation import (
    EvaluationRunner,
    LLMJudge,
    compare_reports,
    load_report,
    render_markdown,
    save_report,
)


class _FakeAnswer:
    def __init__(self, text, citations=None, trace_id="trace-1"):
        self.text = text
        self.citations = citations or []
        self.trace_id = trace_id


class _FakeQAEngine:
    def __init__(self, answers):
        self.answers = answers

    def answer(self, question, session_id=None, user_id=None):
        return self.answers[question]


class _FakeTraceStore:
    def __init__(self, traces):
        self.traces = traces

    def get(self, trace_id):
        return self.traces.get(trace_id)


class _FakeJudgeLLM(LLMProvider):
    def generate(self, prompt, system_prompt="", temperature=0.7, max_tokens=2048):
        return '{"faithfulness": 0.9, "relevancy": 0.8, "reason": "ok"}'

    def generate_with_messages(self, messages, temperature=0.7, max_tokens=2048):
        return ""


def _citation(title: str, index: int) -> Citation:
    return Citation(
        chunk_id=f"c{index}",
        document_id=f"d{index}",
        document_title=title,
        content="内容",
        index=index,
    )


def test_evaluation_runner_metrics():
    case_hit = EvalCase(
        id="e1",
        question="q1",
        category="factual",
        expected_document_titles=["A.pdf"],
    )
    case_partial = EvalCase(
        id="e2",
        question="q2",
        category="comparison",
        expected_document_titles=["A.pdf", "B.pdf"],
    )
    case_unanswerable = EvalCase(
        id="e3",
        question="q3",
        category="unanswerable",
        should_answer=False,
    )
    answers = {
        "q1": _FakeAnswer("答案", [_citation("A.pdf", 1)], "t1"),
        "q2": _FakeAnswer("答案", [_citation("A.pdf", 1)], "t2"),
        "q3": _FakeAnswer(NO_ANSWER_TEXT, [], "t3"),
    }
    trace_store = _FakeTraceStore(
        {
            "t1": Record(kind="trace", metadata={"duration_ms": 100}),
            "t2": Record(kind="trace", metadata={"duration_ms": 200}),
            "t3": Record(kind="trace", metadata={"duration_ms": 50}),
        }
    )
    runner = EvaluationRunner(
        _FakeQAEngine(answers),
        trace_store=trace_store,
        judge=LLMJudge(_FakeJudgeLLM()),
    )

    report = runner.run([case_hit, case_partial, case_unanswerable])

    assert report.total == 3
    assert report.answerable == 2
    assert report.unanswerable == 1
    assert report.hit_rate == 1.0
    assert report.recall_at_k == 0.75
    assert report.mrr == 1.0
    assert report.citation_accuracy == 1.0
    assert report.citation_coverage == 0.75
    assert report.no_answer_accuracy == 1.0
    assert report.faithfulness == 0.9
    assert report.relevancy == 0.8
    assert report.avg_latency_ms == 116.6667


def test_report_save_load_compare_and_markdown(tmp_path):
    report = EvaluationRunner(
        _FakeQAEngine(
            {
                "q1": _FakeAnswer(
                    "答案",
                    [_citation("A.pdf", 1)],
                )
            }
        )
    ).run(
        [
            EvalCase(
                id="e1",
                question="q1",
                category="factual",
                expected_document_titles=["A.pdf"],
            )
        ]
    )
    path = save_report(report, tmp_path / "report.json")
    loaded = load_report(path)
    assert loaded.hit_rate == 1.0
    deltas = compare_reports(loaded, report)
    assert deltas["hit_rate"] == 0.0
    assert "评估报告" in render_markdown(loaded)
