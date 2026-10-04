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
from knowresearch.ingestion.chunker import Chunker
from knowresearch.ingestion.parsers import PDFParser, WordParser


@dataclass
class IngestResult:
    """单文件摄取结果。"""

    file_path: str
    document_id: str
    section_count: int
    chunk_count: int
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
    ) -> None:
        self.document_store = document_store
        self.vector_store = vector_store
        self.fulltext_store = fulltext_store
        self.object_storage = object_storage
        self.embedding_provider = embedding_provider
        self.chunker = chunker or Chunker()
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
            )
        except Exception as e:
            return IngestResult(
                file_path=file_path,
                document_id="",
                section_count=0,
                chunk_count=0,
                error=str(e),
            )