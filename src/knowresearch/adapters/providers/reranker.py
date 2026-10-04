"""RerankerProvider 适配器：占位实现。

M1 阶段不做真正的重排，直接返回原顺序和均匀分数。
M3 阶段换本地 bge-reranker-base。
"""

from knowresearch.core.ports import RerankerProvider


class MockReranker(RerankerProvider):
    """占位 reranker：返回原顺序，分数为 1.0 / (i+1) 递减。"""

    def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int = 5,
    ) -> list[tuple[int, float]]:
        results = [
            (i, 1.0 / (i + 1))
            for i in range(min(len(documents), top_k))
        ]
        return results