"""方案一收尾测试：成本控制、表格兜底、小规模端到端。"""

from __future__ import annotations

import os
import tempfile

import fitz  # PyMuPDF
from fastapi.testclient import TestClient

from knowresearch.adapters.providers.embedding import MockEmbedding
from knowresearch.adapters.stores import BM25FullTextStore, InMemoryVectorStore
from knowresearch.adapters.stores.object_storage import LocalObjectStorage
from knowresearch.api.app import create_app
from knowresearch.core.ports import DocumentStorePort, VLMProvider
from knowresearch.core.retrieval.citation import CitationBuilder
from knowresearch.core.schemas import Document
from knowresearch.ingestion.media_blocks import PDFMediaExtractor, TableBlock
from knowresearch.ingestion.pipeline import IngestPipeline
from knowresearch.ingestion.verbalizer import FigureVerbalizer, TableTranscriber
from knowresearch.multimodal import FigureReader


class _VLM(VLMProvider):
    def __init__(self, response="曲线对比图：A 优于 B"):
        self._response = response
        self.calls = 0

    def describe_image(self, image_path, prompt=""):
        self.calls += 1
        return self._response

    def extract_structure(self, image_path):
        return {"raw_text": self._response, "structure": self._response}


class _FlakyVLM(VLMProvider):
    def __init__(self, fail_times=1, response="ok"):
        self.fail_left = fail_times
        self._response = response
        self.calls = 0

    def describe_image(self, image_path, prompt=""):
        self.calls += 1
        if self.fail_left > 0:
            self.fail_left -= 1
            raise RuntimeError("vlm fail")
        return self._response

    def extract_structure(self, image_path):
        return {}


class _DocStore(DocumentStorePort):
    def __init__(self):
        self.docs = {}

    def save_document(self, doc):
        self.docs[doc.id] = doc

    def get_document(self, doc_id):
        return self.docs.get(doc_id)

    def get_document_by_hash(self, file_hash):
        return None

    def list_documents(self):
        return list(self.docs.values())

    def save_sections(self, sections):
        pass

    def get_sections(self, doc_id):
        return []

    def delete_document(self, doc_id):
        self.docs.pop(doc_id, None)


def _write_one_page_pdf(path: str) -> None:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 60), "PINN 是一种神经网络方法。")

    # 2x2 表格
    page.draw_rect(fitz.Rect(72, 90, 200, 150), width=1)
    for x in (72, 136, 200):
        page.draw_line((x, 90), (x, 150))
    for y in (90, 120, 150):
        page.draw_line((72, y), (200, y))
    page.insert_text((80, 110), "A")
    page.insert_text((144, 110), "B")
    page.insert_text((80, 142), "1")
    page.insert_text((144, 142), "2")

    # 一张图
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 40, 40))
    pix.clear_with(255)
    page.insert_image(fitz.Rect(100, 200, 180, 280), stream=pix.tobytes("png"))

    doc.save(path)
    doc.close()


# ----------------------------------------------------------------------
# 成本控制
# ----------------------------------------------------------------------


def test_verbalizer_caches_identical_image():
    vlm = _VLM()
    verbalizer = FigureVerbalizer(vlm=vlm, retries=0)
    first = verbalizer.describe(b"\x89PNG", "图1", "正文")
    second = verbalizer.describe(b"\x89PNG", "图1", "正文")
    assert first == second == "曲线对比图：A 优于 B"
    assert vlm.calls == 1  # 相同图片只调一次


def test_verbalizer_retries_once_on_failure():
    vlm = _FlakyVLM(fail_times=1, response="ok")
    verbalizer = FigureVerbalizer(vlm=vlm, retries=1)
    assert verbalizer.describe(b"\x89PNG", "图1", "正文") == "ok"
    assert vlm.calls == 2


class _ManyFiguresExtractor:
    def extract_tables(self, file_path):
        return []

    def extract_figures(self, file_path):
        from knowresearch.ingestion.media_blocks import FigureBlock

        return [FigureBlock(page=1, bbox=(0, 0, 10, 10), xref=1) for _ in range(3)]

    def render_figure(self, file_path, figure):
        return b"\x89PNG"

    def render_region(self, file_path, page, bbox, dpi=144):
        return b"\x89PNG"


def test_max_figures_per_doc():
    with tempfile.TemporaryDirectory() as tmp:
        pdf = os.path.join(tmp, "d.pdf")
        _write_one_page_pdf(pdf)
        pipeline = IngestPipeline(
            document_store=_DocStore(),
            vector_store=InMemoryVectorStore(dimension=384),
            fulltext_store=BM25FullTextStore(),
            object_storage=LocalObjectStorage(os.path.join(tmp, "files")),
            embedding_provider=MockEmbedding(),
            media_extractor=_ManyFiguresExtractor(),
            figure_verbalizer=FigureVerbalizer(vlm=_VLM(), retries=0),
            max_figures_per_doc=2,
        )
        result = pipeline.ingest_file(pdf)
        assert result.figure_count == 2


# ----------------------------------------------------------------------
# 表格兜底：结构识别失败时用 VLM 转 Markdown
# ----------------------------------------------------------------------


class _EmptyTableExtractor:
    def extract_tables(self, file_path):
        return [TableBlock(page=1, bbox=(0, 0, 10, 10), markdown="", caption="表1")]

    def extract_figures(self, file_path):
        return []

    def render_figure(self, file_path, figure):
        return b"\x89PNG"

    def render_region(self, file_path, page, bbox, dpi=144):
        return b"\x89PNG"


class _FakeTableTranscriber:
    def transcribe(self, image_bytes):
        return "| X |\n| --- |\n| 1 |"


def test_table_transcriber_fallback():
    with tempfile.TemporaryDirectory() as tmp:
        pdf = os.path.join(tmp, "d.pdf")
        _write_one_page_pdf(pdf)
        pipeline = IngestPipeline(
            document_store=_DocStore(),
            vector_store=InMemoryVectorStore(dimension=384),
            fulltext_store=BM25FullTextStore(),
            object_storage=LocalObjectStorage(os.path.join(tmp, "files")),
            embedding_provider=MockEmbedding(),
            media_extractor=_EmptyTableExtractor(),
            table_transcriber=_FakeTableTranscriber(),
        )
        result = pipeline.ingest_file(pdf)
        assert result.table_count == 1
        table = next(
            rec for rec in pipeline.fulltext_store._records.values()
            if rec.metadata.get("chunk_type") == "table"
        )
        assert "| X |" in table.content


# ----------------------------------------------------------------------
# 小规模端到端：正文 + 表格 + 图，走完整链路
# ----------------------------------------------------------------------


def test_small_pdf_end_to_end():
    with tempfile.TemporaryDirectory() as tmp:
        pdf = os.path.join(tmp, "paper.pdf")
        _write_one_page_pdf(pdf)

        object_storage = LocalObjectStorage(os.path.join(tmp, "files"))
        doc_store = _DocStore()
        vlm = _VLM()
        pipeline = IngestPipeline(
            document_store=doc_store,
            vector_store=InMemoryVectorStore(dimension=384),
            fulltext_store=BM25FullTextStore(),
            object_storage=object_storage,
            embedding_provider=MockEmbedding(),
            media_extractor=PDFMediaExtractor(min_image_side=10),
            figure_verbalizer=FigureVerbalizer(vlm=vlm, retries=0),
        )

        result = pipeline.ingest_file(pdf)
        assert not result.error
        assert result.table_count == 1
        assert result.figure_count == 1

        records = pipeline.fulltext_store._records
        table = next(r for r in records.values() if r.metadata.get("chunk_type") == "table")
        figure = next(r for r in records.values() if r.metadata.get("chunk_type") == "figure")

        assert "| A |" in table.content
        image_key = figure.metadata["image_key"]
        assert image_key.startswith("figures/")
        assert "曲线对比图" in figure.content  # VLM 描述进入文本
        assert object_storage.get(image_key).startswith(b"\x89PNG")

        # 引用携带 chunk_type / image_key，可回取原图
        document = next(iter(doc_store.docs.values()))
        citation = CitationBuilder(doc_store).build([figure])[0]
        assert citation.chunk_type == "figure"
        assert citation.image_key == image_key

        # 查询侧：命中的图交给 VLM 解读
        reader = FigureReader(object_storage, vlm)
        interpretation = reader.read_citations([citation], question="哪个更好？")
        assert "[图 1]" in interpretation

        # 图片接口可回取
        app = create_app(object_storage=object_storage, allow_lazy_production=False)
        client = TestClient(app)
        response = client.get(f"/documents/{document.id}/figures/{os.path.basename(image_key)}")
        assert response.status_code == 200
        assert response.content.startswith(b"\x89PNG")
