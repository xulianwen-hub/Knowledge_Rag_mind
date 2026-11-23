"""图表摄取测试：表格转 Markdown、图提取/渲染、图像描述化、管线集成。"""

from __future__ import annotations

import os
import tempfile

import fitz  # PyMuPDF

from knowresearch.adapters.providers.embedding import MockEmbedding
from knowresearch.adapters.stores import BM25FullTextStore, InMemoryVectorStore
from knowresearch.adapters.stores.object_storage import LocalObjectStorage
from knowresearch.core.ports import DocumentStorePort, VLMProvider
from knowresearch.ingestion.media_blocks import (
    FigureBlock,
    PDFMediaExtractor,
    TableBlock,
    table_to_markdown,
)
from knowresearch.ingestion.pipeline import IngestPipeline
from knowresearch.ingestion.verbalizer import FigureVerbalizer


def _write_minimal_pdf(path: str, with_image: bool = False, with_table: bool = False) -> None:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 60), "PINN 是一种神经网络方法。")

    if with_table:
        page.draw_rect(fitz.Rect(72, 90, 200, 150), width=1)
        for x in (72, 136, 200):
            page.draw_line((x, 90), (x, 150))
        for y in (90, 120, 150):
            page.draw_line((72, y), (200, y))
        page.insert_text((80, 110), "A")
        page.insert_text((144, 110), "B")
        page.insert_text((80, 142), "1")
        page.insert_text((144, 142), "2")

    if with_image:
        pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 40, 40))
        pix.clear_with(255)
        page.insert_image(fitz.Rect(100, 200, 180, 280), stream=pix.tobytes("png"))

    doc.save(path)
    doc.close()


class _FakeDocumentStore(DocumentStorePort):
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


class _FakeMediaExtractor:
    def extract_tables(self, file_path):
        return [
            TableBlock(
                page=1,
                bbox=(0, 0, 10, 10),
                markdown="| A | B |\n| --- | --- |\n| 1 | 2 |",
                caption="表1 参数",
            )
        ]

    def extract_figures(self, file_path):
        return [FigureBlock(page=1, bbox=(0, 0, 100, 100), xref=1, caption="图1 流程")]

    def render_figure(self, file_path, figure):
        return b"\x89PNGfake-image"


class _FakeVLM(VLMProvider):
    def __init__(self, response="这是描述", raise_exc=False):
        self._response = response
        self._raise = raise_exc
        self.calls = []

    def describe_image(self, image_path, prompt=""):
        self.calls.append((image_path, prompt))
        if self._raise:
            raise RuntimeError("vlm 挂了")
        return self._response

    def extract_structure(self, image_path):
        return {"raw_text": self._response, "structure": self._response}


# ----------------------------------------------------------------------
# table_to_markdown
# ----------------------------------------------------------------------


def test_table_to_markdown_basic():
    markdown = table_to_markdown([["A", "B"], ["1", "2"]])
    assert markdown.startswith("| A | B |")
    assert "| 1 | 2 |" in markdown
    assert "---" in markdown


def test_table_to_markdown_ragged_rows():
    markdown = table_to_markdown([["A", "B", "C"], ["1", "2"]])
    lines = markdown.splitlines()
    assert lines[0].count("|") == 4  # 3 列
    assert lines[2].count("|") == 4  # 补齐空列


def test_table_to_markdown_empty():
    assert table_to_markdown([]) == ""
    assert table_to_markdown([["", ""]]) == ""


# ----------------------------------------------------------------------
# PDFMediaExtractor（真实 PDF）
# ----------------------------------------------------------------------


def test_extract_figures_and_render():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "fig.pdf")
        _write_minimal_pdf(path, with_image=True)
        extractor = PDFMediaExtractor(min_image_side=10)
        figures = extractor.extract_figures(path)
        assert len(figures) == 1
        assert figures[0].page == 1
        rendered = extractor.render_figure(path, figures[0])
        assert rendered.startswith(b"\x89PNG")


def test_extract_tables_detects_grid():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "table.pdf")
        _write_minimal_pdf(path, with_table=True)
        extractor = PDFMediaExtractor()
        tables = extractor.extract_tables(path)
        assert len(tables) >= 1
        assert "A" in tables[0].markdown
        assert "B" in tables[0].markdown


# ----------------------------------------------------------------------
# FigureVerbalizer
# ----------------------------------------------------------------------


def test_verbalizer_without_vlm_returns_empty():
    verbalizer = FigureVerbalizer(vlm=None)
    assert verbalizer.describe(b"img", "图题", "上下文") == ""


def test_verbalizer_calls_vlm():
    vlm = _FakeVLM(response="曲线对比图")
    verbalizer = FigureVerbalizer(vlm=vlm)
    result = verbalizer.describe(b"\x89PNG", "图1", "正文")
    assert result == "曲线对比图"
    assert len(vlm.calls) == 1
    assert "图1" in vlm.calls[0][1]
    assert "正文" in vlm.calls[0][1]


def test_verbalizer_degrades_on_exception():
    verbalizer = FigureVerbalizer(vlm=_FakeVLM(raise_exc=True))
    assert verbalizer.describe(b"\x89PNG", "图1", "正文") == ""


# ----------------------------------------------------------------------
# IngestPipeline 集成（注入假提取器，验证图/表入库与图片存储）
# ----------------------------------------------------------------------


def test_pipeline_ingests_tables_and_figures():
    with tempfile.TemporaryDirectory() as tmp:
        pdf_path = os.path.join(tmp, "doc.pdf")
        _write_minimal_pdf(pdf_path)

        embedding = MockEmbedding()
        vector_store = InMemoryVectorStore(dimension=embedding.dimension)
        fulltext_store = BM25FullTextStore()
        object_storage = LocalObjectStorage(os.path.join(tmp, "files"))
        doc_store = _FakeDocumentStore()

        pipeline = IngestPipeline(
            document_store=doc_store,
            vector_store=vector_store,
            fulltext_store=fulltext_store,
            object_storage=object_storage,
            embedding_provider=embedding,
            media_extractor=_FakeMediaExtractor(),
            figure_verbalizer=FigureVerbalizer(vlm=_FakeVLM()),
        )

        result = pipeline.ingest_file(pdf_path)
        assert not result.error
        assert result.table_count == 1
        assert result.figure_count == 1

        # 图/表 chunk 进入全文索引，带 chunk_type 元数据
        kinds = {
            rec.metadata.get("chunk_type") for rec in fulltext_store._records.values()
        }
        assert "table" in kinds
        assert "figure" in kinds

        # 原图写入对象存储，可回取
        figure_record = next(
            rec for rec in fulltext_store._records.values()
            if rec.metadata.get("chunk_type") == "figure"
        )
        image_key = figure_record.metadata["image_key"]
        assert image_key.startswith("figures/")
        assert object_storage.get(image_key) == b"\x89PNGfake-image"
