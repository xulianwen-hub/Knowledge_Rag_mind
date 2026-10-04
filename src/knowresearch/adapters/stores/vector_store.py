"""VectorStorePort 适配器：内存版（余弦相似度）。

首版用纯 Python + numpy 实现，跑通接口；M3 阶段换 FAISS 适配器。
向量存在内存 dict 中，重启丢失——MVP 阶段可接受，因为解析流水线会重建索引。
"""

import numpy as np

from knowresearch.core.ports import VectorStorePort
from knowresearch.core.schemas import Record


class InMemoryVectorStore(VectorStorePort):
    """内存向量存储，余弦相似度检索。"""

    def __init__(self, dimension: int):
        self._dimension = dimension
        self._records: dict[str, Record] = {}
        self._vectors: dict[str, np.ndarray] = {}

    @property
    def dimension(self) -> int:
        return self._dimension

    def add(self, records: list[Record], embeddings: list[list[float]]) -> None:
        if len(records) != len(embeddings):
            raise ValueError("records 和 embeddings 数量必须一致")
        for record, emb in zip(records, embeddings):
            vec = np.array(emb, dtype=np.float32)
            if vec.shape[0] != self._dimension:
                raise ValueError(
                    f"向量维度不匹配：期望 {self._dimension}，实际 {vec.shape[0]}"
                )
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            self._records[record.id] = record
            self._vectors[record.id] = vec

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
    ) -> list[tuple[Record, float]]:
        if not self._vectors:
            return []
        query_vec = np.array(query_embedding, dtype=np.float32)
        norm = np.linalg.norm(query_vec)
        if norm > 0:
            query_vec = query_vec / norm

        ids = list(self._vectors.keys())
        matrix = np.stack([self._vectors[i] for i in ids])
        scores = matrix @ query_vec

        top_k = min(top_k, len(ids))
        top_indices = np.argsort(-scores)[:top_k]

        results: list[tuple[Record, float]] = []
        for idx in top_indices:
            record_id = ids[idx]
            results.append((self._records[record_id], float(scores[idx])))
        return results

    def delete(self, record_ids: list[str]) -> None:
        for record_id in record_ids:
            self._records.pop(record_id, None)
            self._vectors.pop(record_id, None)

    def clear(self) -> None:
        self._records.clear()
        self._vectors.clear()