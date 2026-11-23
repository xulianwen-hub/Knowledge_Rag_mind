"""图解读器与图片接口测试（方案一查询侧）。"""

from __future__ import annotations

from fastapi.testclient import TestClient

from knowresearch.api.app import create_app
from knowresearch.core.generation.answer_generator import Answer
from knowresearch.core.generation.prompt import build_messages
from knowresearch.core.ports import VLMProvider
from knowresearch.core.qa_engine import QAEngine
from knowresearch.core.retrieval.citation import Citation
from knowresearch.multimodal import FigureReader


class _FakeVLM(VLMProvider):
    def __init__(self, response="曲线对比图：A 优于 B", raise_exc=False):
        self._response = response
        self._raise = raise_exc
        self.calls = []

    def describe_image(self, image_path, prompt=""):
        self.calls.append(prompt)
        if self._raise:
            raise RuntimeError("vlm 挂了")
        return self._response

    def extract_structure(self, image_path):
        return {"raw_text": self._response, "structure": self._response}


class _FakeObjectStorage:
    def __init__(self, data=None):
        self.data = data or {}

    def get(self, key):
        if key not in self.data:
            raise FileNotFoundError(key)
        return self.data[key]


def _figure_citation(index=1, image_key="figures/d1/1.png", chunk_type="figure"):
    return Citation(
        chunk_id=f"c{index}",
        document_id="d1",
        document_title="论文.pdf",
        index=index,
        chunk_type=chunk_type,
        image_key=image_key,
    )


def test_figure_reader_reads_figure_citations():
    storage = _FakeObjectStorage({"figures/d1/1.png": b"\x89PNG"})
    reader = FigureReader(storage, _FakeVLM())
    citations = [_figure_citation(1), _figure_citation(2, chunk_type="text")]
    result = reader.read_citations(citations, question="哪个更好？")
    assert "[图 1]" in result
    assert "曲线对比图" in result


def test_figure_reader_skips_missing_image():
    reader = FigureReader(_FakeObjectStorage(), _FakeVLM())
    assert reader.read_citations([_figure_citation()], "问题") == ""


def test_figure_reader_respects_max_figures():
    storage = _FakeObjectStorage(
        {
            "figures/d1/1.png": b"\x89PNG",
            "figures/d1/2.png": b"\x89PNG",
            "figures/d1/3.png": b"\x89PNG",
        }
    )
    reader = FigureReader(storage, _FakeVLM(), max_figures=2)
    citations = [_figure_citation(1), _figure_citation(2, "figures/d1/2.png"), _figure_citation(3, "figures/d1/3.png")]
    result = reader.read_citations(citations, "问题")
    assert result.count("[图 ") == 2


def test_build_messages_injects_figure_context():
    citations = [_figure_citation(1)]
    messages = build_messages("问题", citations, figure_context="这是图片解读")
    contents = [m["content"] for m in messages]
    assert any("【图片解读】" in c and "这是图片解读" in c for c in contents)


class _FakeFigureReader:
    def read_citations(self, citations, question=""):
        return "解读文本"


class _FakeAnswerGenerator:
    def __init__(self):
        self.kwargs = None

    def generate(self, question, chunks, citations, **kwargs):
        self.kwargs = kwargs
        return Answer(text="答案", citations=[], trace_id="t")


def test_qa_engine_injects_figure_context():
    generator = _FakeAnswerGenerator()
    engine = QAEngine(
        retriever=None,
        reranker=None,
        citation_builder=None,
        answer_generator=generator,
        embedding_provider=None,
        figure_reader=_FakeFigureReader(),
    )
    engine._generate("问题", [], [_figure_citation(1)], [], "", "trace")
    assert generator.kwargs["figure_context"] == "解读文本"


def test_api_serves_figure_image():
    storage = _FakeObjectStorage({"figures/d1/1.png": b"\x89PNG-data"})
    app = create_app(object_storage=storage, allow_lazy_production=False)
    client = TestClient(app)
    response = client.get("/documents/d1/figures/1.png")
    assert response.status_code == 200
    assert response.content == b"\x89PNG-data"
    assert response.headers["content-type"] == "image/png"


def test_api_figure_image_404_when_missing():
    app = create_app(object_storage=_FakeObjectStorage(), allow_lazy_production=False)
    client = TestClient(app)
    assert client.get("/documents/d1/figures/nope.png").status_code == 404
