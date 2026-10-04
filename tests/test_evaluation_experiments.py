"""M7-3 实验框架测试。"""

from knowresearch.core.retrieval.citation import Citation
from knowresearch.core.schemas import EvalCase
from knowresearch.evaluation import (
    ExperimentConfig,
    ExperimentRunner,
    render_experiment_markdown,
)


class _FakeAnswer:
    def __init__(self, title):
        self.text = "答案"
        self.trace_id = "trace-1"
        self.citations = [
            Citation(
                chunk_id="c1",
                document_id="d1",
                document_title=title,
                content="内容",
                index=1,
            )
        ]


class _FakeEngine:
    def __init__(self, title):
        self.title = title

    def answer(self, question, session_id=None, user_id=None):
        return _FakeAnswer(self.title)


def test_experiment_runner_compares_configs():
    case = EvalCase(
        id="e1",
        question="问题",
        category="factual",
        expected_document_titles=["A.pdf"],
    )
    configs = [
        ExperimentConfig(
            name="hit",
            embedding_model="model-a",
            reranker_model="reranker",
        ),
        ExperimentConfig(
            name="miss",
            embedding_model="model-b",
            use_reranker=False,
        ),
    ]

    def factory(config):
        return _FakeEngine("A.pdf" if config.name == "hit" else "B.pdf")

    results = ExperimentRunner(factory).run(configs, [case])

    assert len(results) == 2
    assert results[0].report.hit_rate == 1.0
    assert results[1].report.hit_rate == 0.0
    markdown = render_experiment_markdown(results)
    assert "hit" in markdown
    assert "miss" in markdown
