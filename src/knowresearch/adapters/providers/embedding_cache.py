"""Embedding 缓存装饰器。"""

from __future__ import annotations

from collections import OrderedDict

from knowresearch.core.ports import EmbeddingProvider


class CachedEmbeddingProvider(EmbeddingProvider):
    """用 LRU 缓存包装 EmbeddingProvider。"""

    def __init__(self, provider: EmbeddingProvider, maxsize: int = 2048):
        self.provider = provider
        self.maxsize = maxsize
        self._cache: OrderedDict[str, list[float]] = OrderedDict()
        self.hits = 0
        self.misses = 0

    @property
    def dimension(self) -> int:
        return self.provider.dimension

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed_many(texts, prefix="d")

    def embed_query(self, text: str) -> list[float]:
        return self._embed_many([text], prefix="q")[0]

    def _embed_many(self, texts: list[str], prefix: str) -> list[list[float]]:
        results: list[list[float] | None] = [None] * len(texts)
        misses: list[tuple[int, str, str]] = []
        for index, text in enumerate(texts):
            key = f"{prefix}:{text}"
            cached = self._cache.get(key)
            if cached is not None:
                self._cache.move_to_end(key)
                self.hits += 1
                results[index] = cached
            else:
                self.misses += 1
                misses.append((index, key, text))

        if misses:
            vectors = self.provider.embed_documents([text for _, _, text in misses])
            for (index, key, _), vector in zip(misses, vectors):
                results[index] = vector
                self._cache[key] = vector
                self._cache.move_to_end(key)
                if len(self._cache) > self.maxsize:
                    self._cache.popitem(last=False)
        return [result or [] for result in results]
