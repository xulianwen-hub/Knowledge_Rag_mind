"""摄取流水线：扫描目录 → 解析 → 切块 → 入库。

依赖注入：构造函数接收各端口实例，不硬编码实现。
流程：
1. 扫描目录下的 .pdf / .docx 文件
2. 逐文件：hash 去重 → 选 parser → 解析 → 切块 → 存储
"""

import os
from dataclasses import dataclass, field

from knowresearch.core.ports import (
    DocumentStorePort,
    EmbeddingProvider,
    FullTextStorePort,
    ObjectStoragePort,
    VectorStorePort,
)
from knowresearch.core.schemas import Chunk
from knowresearch.ingestion.chunker import Chunker
from knowresearch.ingestion.media_blocks import PDFMediaExtractor
from knowresearch.ingestion.parsers import PDFParser, WordParser
from knowresearch.ingestion.verbalizer import FigureVerbalizer, TableTranscriber


@dataclass
class IngestResult:
    """单文件摄取结果。"""

    file_path: str
    document_id: str
    section_count: int
    chunk_count: int
    table_count: int = 0
    figure_count: int = 0
    skipped: bool = False
    error: str = ""


@dataclass
class IngestSummary:
    """批量摄取汇总。"""

    total: int = 0
    success: int = 0
    skipped: int = 0
    failed: int = 0
    results: list[IngestResult] = field(default_factory=list)


class IngestPipeline:
    """文档摄取流水线。"""

    SUPPORTED_EXTENSIONS = {".pdf", ".docx"}

    def __init__(
        self,
        document_store: DocumentStorePort,
        vector_store: VectorStorePort,
        fulltext_store: FullTextStorePort,
        object_storage: ObjectStoragePort,
        embedding_provider: EmbeddingProvider,
        chunker: Chunker | None = None,
        media_extractor: PDFMediaExtractor | None = None,
        figure_verbalizer: FigureVerbalizer | None = None,
        table_transcriber: TableTranscriber | None = None,
        max_figures_per_doc: int = 20,
    ) -> None:
        self.document_store = document_store
        self.vector_store = vector_store
        self.fulltext_store = fulltext_store
        self.object_storage = object_storage
        self.embedding_provider = embedding_provider
        self.chunker = chunker or Chunker()
        self.media_extractor = media_extractor
        self.figure_verbalizer = figure_verbalizer
        self.table_transcriber = table_transcriber
        self.max_figures_per_doc = max(0, max_figures_per_doc)
        self._parsers = {
            ".pdf": PDFParser(),
            ".docx": WordParser(),
        }

    def ingest_directory(self, dir_path: str) -> IngestSummary:
        """批量摄取目录下所有支持的文件。"""
        summary = IngestSummary()
        for root, _, files in os.walk(dir_path):
            for fname in sorted(files):
                ext = os.path.splitext(fname)[1].lower()
                if ext not in self.SUPPORTED_EXTENSIONS:
                    continue
                file_path = os.path.join(root, fname)
                result = self.ingest_file(file_path)
                summary.results.append(result)
                summary.total += 1
                if result.skipped:
                    summary.skipped += 1
                elif result.error:
                    summary.failed += 1
                else:
                    summary.success += 1
        return summary

    def ingest_file(self, file_path: str) -> IngestResult:
        """摄取单个文件。"""
        ext = os.path.splitext(file_path)[1].lower()
        parser = self._parsers.get(ext)
        if parser is None:
            return IngestResult(
                file_path=file_path,
                document_id="",
                section_count=0,
                chunk_count=0,
                error=f"不支持的文件类型: {ext}",
            )

        try:
            document, sections = parser.parse(file_path)

            existing = self.document_store.get_document_by_hash(document.file_hash)
            if existing is not None:
                return IngestResult(
                    file_path=file_path,
                    document_id=existing.id,
                    section_count=0,
                    chunk_count=0,
                    skipped=True,
                )

            self.document_store.save_document(document)
            self.document_store.save_sections(sections)

            with open(file_path, "rb") as f:
                self.object_storage.put(document.file_name, f.read())

            chunks = self.chunker.chunk_sections(sections)
            media_chunks, table_count, figure_count = self._ingest_media(
                document, sections, file_path
            )
            chunks = list(chunks) + media_chunks
            if chunks:
                texts = [c.content for c in chunks]
                embeddings = self.embedding_provider.embed_documents(texts)
                self.vector_store.add(chunks, embeddings)
                self.fulltext_store.add(chunks)

            return IngestResult(
                file_path=file_path,
                document_id=document.id,
                section_count=len(sections),
                chunk_count=len(chunks),
                table_count=table_count,
                figure_count=figure_count,
            )
        except Exception as e:
            return IngestResult(
                file_path=file_path,
                document_id="",
                section_count=0,
                chunk_count=0,
                error=str(e),
            )

    # ------------------------------------------------------------------
    # 图表摄取（方案一：图像描述化 + 表格转 Markdown）
    # ------------------------------------------------------------------

    def _ingest_media(self, document, sections, file_path: str):
        """提取图/表并转成 Chunk，返回 (chunks, table_count, figure_count)。"""
        if self.media_extractor is None:
            return [], 0, 0

        try:
            tables = self.media_extractor.extract_tables(file_path)
        except Exception:
            tables = []
        try:
            figures = self.media_extractor.extract_figures(file_path)
        except Exception:
            figures = []

        chunks: list[Chunk] = []
        table_added = 0
        figure_added = 0

        for table in tables:
            markdown = table.markdown
            if not markdown and self.table_transcriber is not None:
                try:
                    image_bytes = self.media_extractor.render_region(
                        file_path, table.page, table.bbox
                    )
                    markdown = self.table_transcriber.transcribe(image_bytes)
                except Exception:
                    markdown = ""
            if not markdown:
                continue
            metadata = self._media_metadata(
                document, sections, table.page, "table", caption=table.caption, bbox=table.bbox
            )
            chunks.append(Chunk(content=markdown, source=file_path, metadata=metadata))
            table_added += 1

        for index, figure in enumerate(figures, start=1):
            if index > self.max_figures_per_doc:
                break
            try:
                image_bytes = self.media_extractor.render_figure(file_path, figure)
            except Exception:
                image_bytes = b""

            image_key = ""
            if image_bytes:
                image_key = f"figures/{document.id}/{index}.png"
                try:
                    self.object_storage.put(image_key, image_bytes)
                except Exception:
                    image_key = ""

            caption = figure.caption or ""
            context = self._section_context(sections, figure.page)
            description = ""
            if self.figure_verbalizer is not None and image_bytes:
                try:
                    description = self.figure_verbalizer.describe(image_bytes, caption, context)
                except Exception:
                    description = ""

            parts = [p for p in (caption, description, context) if p]
            content = "\n".join(parts) or "（图片，无文字说明）"
            metadata = self._media_metadata(
                document,
                sections,
                figure.page,
                "figure",
                caption=caption,
                bbox=figure.bbox,
                image_key=image_key,
            )
            chunks.append(Chunk(content=content, source=file_path, metadata=metadata))
            figure_added += 1

        return chunks, table_added, figure_added

    @staticmethod
    def _media_metadata(
        document,
        sections,
        page: int,
        chunk_type: str,
        caption: str = "",
        bbox=None,
        image_key: str = "",
    ) -> dict:
        section = IngestPipeline._section_for_page(sections, page)
        metadata = {
            "document_id": document.id,
            "document_title": document.file_name,
            "chunk_type": chunk_type,
            "page": page,
            "caption": caption,
            "image_key": image_key,
        }
        if bbox is not None:
            metadata["bbox"] = [float(v) for v in bbox]
        if section is not None:
            metadata["section_id"] = section.id
            metadata["section_title"] = section.metadata.get("section_title", "")
        return metadata

    @staticmethod
    def _section_for_page(sections, page: int):
        for section in sections:
            start = section.metadata.get("page_start", 0)
            end = section.metadata.get("page_end", 0)
            if start <= page <= end:
                return section
        return sections[0] if sections else None

    @staticmethod
    def _section_context(sections, page: int) -> str:
        section = IngestPipeline._section_for_page(sections, page)
        if section is None:
            return ""
        title = section.metadata.get("section_title", "")
        content = (section.content or "").strip()[:300]
        if title:
            return f"{title}：{content}" if content else title
        return content
