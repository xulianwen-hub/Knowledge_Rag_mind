"""EmbeddingProvider 适配器：占位实现。

M1 阶段用伪向量（固定维度，基于文本 hash 生成）跑通接口。
M3 阶段换本地 bge-base-zh-v1.5。
"""

import hashlib

from knowresearch.core.ports import EmbeddingProvider


class MockEmbedding(EmbeddingProvider):
    """占位 embedding：基于文本 hash 生成伪向量，维度固定。"""

    def __init__(self, dimension: int = 384):
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def _text_to_vector(self, text: str) -> list[float]:
        """用文本 hash 生成确定性伪向量（同一文本总是同一向量）。"""
        vec = [0.0] * self._dimension
        h = hashlib.md5(text.encode("utf-8")).digest()
        for i in range(self._dimension):
            byte = h[i % len(h)]
            vec[i] = (byte / 255.0) * 2 - 1
        norm = sum(v * v for v in vec) ** 0.5
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._text_to_vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._text_to_vector(text)