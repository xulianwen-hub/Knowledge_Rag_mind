"""文档存储端口：Document 与 SectionBlock 的持久化。

M3 检索时需要根据 chunk 的 document_id 反查文档标题、
根据 section_id 反查章节标题，因此需要专门的文档存储端口。
"""

from abc import ABC, abstractmethod

from knowresearch.core.schemas import Document, SectionBlock


class DocumentStorePort(ABC):
    """文档与章节的持久化端口。"""

    @abstractmethod
    def save_document(self, doc: Document) -> None:
        """保存文档元数据。已存在则更新。"""

    @abstractmethod
    def get_document(self, doc_id: str) -> Document | None:
        """按 ID 获取文档。"""

    @abstractmethod
    def get_document_by_hash(self, file_hash: str) -> Document | None:
        """按文件 hash 获取文档（用于去重）。"""

    @abstractmethod
    def list_documents(self) -> list[Document]:
        """列出所有文档。"""

    @abstractmethod
    def save_sections(self, sections: list[SectionBlock]) -> None:
        """批量保存章节。已存在则更新。"""

    @abstractmethod
    def get_sections(self, doc_id: str) -> list[SectionBlock]:
        """获取指定文档的所有章节，按层级和顺序返回。"""

    @abstractmethod
    def delete_document(self, doc_id: str) -> None:
        """删除文档及其所有章节。"""