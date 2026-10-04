"""存储端口：向量库 / 全文索引 / 对象存储的抽象接口。

所有存储端口都基于 Record 基类读写，不关心具体 kind。
换存储后端（FAISS → pgvector，本地文件 → MinIO）只写新适配器。
"""

from abc import ABC, abstractmethod

from knowresearch.core.schemas import Record


class VectorStorePort(ABC):
    """向量存储端口：embedding 的存取与相似度检索。"""

    @abstractmethod
    def add(self, records: list[Record], embeddings: list[list[float]]) -> None:
        """批量写入向量。records 与 embeddings 一一对应。"""

    @abstractmethod
    def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
    ) -> list[tuple[Record, float]]:
        """相似度检索。返回 (记录, 相似度分数) 列表，按分数降序。"""

    @abstractmethod
    def delete(self, record_ids: list[str]) -> None:
        """按 ID 删除向量。"""

    @abstractmethod
    def clear(self) -> None:
        """清空所有向量。"""


class FullTextStorePort(ABC):
    """全文检索端口：基于关键词 / BM25 的检索。"""

    @abstractmethod
    def add(self, records: list[Record]) -> None:
        """批量写入全文索引。"""

    @abstractmethod
    def search(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[tuple[Record, float]]:
        """全文检索。返回 (记录, BM25 分数) 列表，按分数降序。"""

    @abstractmethod
    def delete(self, record_ids: list[str]) -> None:
        """按 ID 删除索引。"""

    @abstractmethod
    def clear(self) -> None:
        """清空索引。"""


class ObjectStoragePort(ABC):
    """对象存储端口：原始文件（PDF / Word / 图片）的存取。"""

    @abstractmethod
    def put(self, key: str, data: bytes) -> str:
        """上传文件。返回访问路径 / URL。"""

    @abstractmethod
    def get(self, key: str) -> bytes:
        """下载文件。"""

    @abstractmethod
    def delete(self, key: str) -> None:
        """删除文件。"""

    @abstractmethod
    def exists(self, key: str) -> bool:
        """判断文件是否存在。"""

    @abstractmethod
    def list_keys(self, prefix: str = "") -> list[str]:
        """列出指定前缀下的所有 key。"""