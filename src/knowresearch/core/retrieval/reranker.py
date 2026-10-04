"""重排器：对检索候选按相关性重新排序。

调用 RerankerProvider 对候选 chunks 打分排序，返回 top-k。
M3 阶段用 MockReranker（返回原始顺序），M5 换真实 bge-reranker。
"""

from knowresearch.core.ports import RerankerProvider
from knowresearch.core.schemas import Chunk


class Reranker:
    """重排器。"""

    def __init__(self, provider: RerankerProvider) -> None:
        self.provider = provider

    def rerank(self, query: str, chunks: list[Chunk], top_k: int = 5) -> list[Chunk]:
        """对候选 chunks 重排，返回 top-k。

        Args:
            query: 查询语句
            chunks: 候选 chunks
            top_k: 返回前 k 个

        Returns:
            按相关性降序排列的 top-k chunks
        """
        if not chunks:
            return []

        documents = [c.content for c in chunks]
        ranked = self.provider.rerank(query, documents, top_k=top_k)

        results: list[Chunk] = []
        for index, _score in ranked:
            if 0 <= index < len(chunks):
                results.append(chunks[index])
            if len(results) >= top_k:
                break

        return results