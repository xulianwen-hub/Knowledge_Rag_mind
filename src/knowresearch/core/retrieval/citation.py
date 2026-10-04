"""引用溯源：根据 chunk 反查文档和章节元数据。

检索返回的 Chunk 只带 document_id / section_id，需要回溯到
Document（文件名）和 SectionBlock（章节标题、页码范围），
才能在生成答案时标注清晰的引用来源。
"""

from pydantic import BaseModel

from knowresearch.core.ports import DocumentStorePort
from knowresearch.core.schemas import Chunk


class Citation(BaseModel):
    """一条引用：chunk 对应的文档/章节/页码信息。"""

    chunk_id: str
    document_id: str
    document_title: str
    section_id: str = ""
    section_title: str = ""
    page: int = 0
    content: str = ""
    index: int = 0


class CitationBuilder:
    """引用构建器：把 Chunk 列表转换成带溯源信息的 Citation 列表。"""

    def __init__(self, document_store: DocumentStorePort) -> None:
        self.document_store = document_store
        self._doc_cache: dict[str, object] = {}
        self._section_cache: dict[str, object] = {}

    def build(self, chunks: list[Chunk]) -> list[Citation]:
        """为每个 chunk 构建 Citation，自动缓存文档和章节避免重复查询。"""
        citations: list[Citation] = []

        for index, chunk in enumerate(chunks, start=1):
            doc = self._get_document(chunk.document_id)
            section = self._get_section(chunk)

            citation = Citation(
                chunk_id=chunk.id,
                document_id=chunk.document_id,
                document_title=getattr(doc, "file_name", "") if doc else "",
                section_id=chunk.section_id,
                section_title=getattr(section, "section_title", "") if section else "",
                page=chunk.page,
                content=chunk.content,
                index=index,
            )
            citations.append(citation)

        return citations

    def _get_document(self, doc_id: str):
        if not doc_id:
            return None
        if doc_id not in self._doc_cache:
            self._doc_cache[doc_id] = self.document_store.get_document(doc_id)
        return self._doc_cache[doc_id]

    def _get_section(self, chunk: Chunk):
        section_id = chunk.section_id
        if not section_id:
            return None
        if section_id not in self._section_cache:
            self._section_cache[section_id] = self._find_section(chunk)
        return self._section_cache[section_id]

    def _find_section(self, chunk: Chunk):
        """在文档的章节列表中查找 section_id 对应的章节。"""
        doc_id = chunk.document_id
        if not doc_id:
            return None
        sections = self.document_store.get_sections(doc_id)
        for section in sections:
            if section.id == chunk.section_id:
                return section
        return None