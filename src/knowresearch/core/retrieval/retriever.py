"""混合检索器：向量检索 + 全文检索 + RRF 融合。

RRF（Reciprocal Rank Fusion）公式：
    score(d) = Σ 1 / (k + rank_i(d))

其中 k 通常取 60，rank_i 是文档 d 在第 i 个结果集中的排名（从 1 开始）。
RRF 的优势：不需要调权重，对不同检索器的分数尺度不敏感。
"""

from knowresearch.core.ports import EmbeddingProvider, FullTextStorePort, VectorStorePort
from knowresearch.core.schemas import Chunk


class Retriever:
    """混合检索器。"""

    def __init__(
        self,
        vector_store: VectorStorePort,
        fulltext_store: FullTextStorePort,
        embedding_provider: EmbeddingProvider,
        rrf_k: int = 60,
        vector_top_k: int = 20,
        fulltext_top_k: int = 20,
        final_top_k: int = 10,
    ) -> None:
        self.vector_store = vector_store
        self.fulltext_store = fulltext_store
        self.embedding_provider = embedding_provider
        self.rrf_k = rrf_k
        self.vector_top_k = vector_top_k
        self.fulltext_top_k = fulltext_top_k
        self.final_top_k = final_top_k

    def retrieve(self, query: str) -> list[Chunk]:
        """执行混合检索，返回融合排序后的 top-N chunks。

        Args:
            query: 用户查询

        Returns:
            按 RRF 分数降序排列的 Chunk 列表，最多 final_top_k 个
        """
        query_embedding = self.embedding_provider.embed_query(query)

        vector_results = self.vector_store.search(
            query_embedding, top_k=self.vector_top_k
        )
        fulltext_results = self.fulltext_store.search(
            query, top_k=self.fulltext_top_k
        )

        rrf_scores: dict[str, float] = {}
        chunk_map: dict[str, Chunk] = {}

        for rank, (record, _score) in enumerate(vector_results, start=1):
            chunk = self._to_chunk(record)
            if chunk is None:
                continue
            chunk_map[chunk.id] = chunk
            rrf_scores[chunk.id] = rrf_scores.get(chunk.id, 0.0) + 1.0 / (
                self.rrf_k + rank
            )

        for rank, (record, _score) in enumerate(fulltext_results, start=1):
            chunk = self._to_chunk(record)
            if chunk is None:
                continue
            chunk_map[chunk.id] = chunk
            rrf_scores[chunk.id] = rrf_scores.get(chunk.id, 0.0) + 1.0 / (
                self.rrf_k + rank
            )

        ranked_ids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)
        top_ids = ranked_ids[: self.final_top_k]
        return [chunk_map[cid] for cid in top_ids]

    @staticmethod
    def _to_chunk(record) -> Chunk | None:
        """Record → Chunk。如果 record 不是 chunk 类型返回 None。"""
        try:
            return Chunk(**record.model_dump())
        except Exception:
            return None