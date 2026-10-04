"""Embedding 缓存测试。"""

from knowresearch.adapters.providers.embedding_cache import CachedEmbeddingProvider
from knowresearch.core.ports import EmbeddingProvider


class _CountingEmbedding(EmbeddingProvider):
    def __init__(self):
        self.calls = 0

    @property
    def dimension(self):
        return 2

    def embed_documents(self, texts):
        self.calls += len(texts)
        return [[float(len(text)), 1.0] for text in texts]

    def embed_query(self, text):
        return self.embed_documents([text])[0]


def test_embedding_cache_hits():
    provider = _CountingEmbedding()
    cache = CachedEmbeddingProvider(provider)

    first = cache.embed_documents(["a", "b"])
    second = cache.embed_documents(["a", "b"])

    assert first == second
    assert provider.calls == 2
    assert cache.hits == 2
    assert cache.misses == 2


def test_embedding_cache_separates_query_and_document():
    provider = _CountingEmbedding()
    cache = CachedEmbeddingProvider(provider)
    cache.embed_documents(["x"])
    cache.embed_query("x")
    assert provider.calls == 2
