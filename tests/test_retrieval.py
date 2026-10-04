"""M3 Step 1+2 测试：Retriever 混合检索 + Reranker 重排。"""

from knowresearch.core.ports import (
    EmbeddingProvider,
    FullTextStorePort,
    RerankerProvider,
    VectorStorePort,
)
from knowresearch.core.retrieval.reranker import Reranker
from knowresearch.core.retrieval.retriever import Retriever
from knowresearch.core.schemas import Chunk, Record


# ================================================================
# Mock 实现
# ================================================================


class _MockVectorStore(VectorStorePort):
    def __init__(self, results: list[tuple[Record, float]]):
        self._results = results

    def add(self, records, embeddings):
        pass

    def search(self, query_embedding, top_k=10):
        return self._results[:top_k]

    def delete(self, record_ids):
        pass

    def clear(self):
        pass


class _MockFullTextStore(FullTextStorePort):
    def __init__(self, results: list[tuple[Record, float]]):
        self._results = results

    def add(self, records):
        pass

    def search(self, query, top_k=10):
        return self._results[:top_k]

    def delete(self, record_ids):
        pass

    def clear(self):
        pass


class _MockEmbedding(EmbeddingProvider):
    @property
    def dimension(self) -> int:
        return 384

    def embed_documents(self, texts):
        return [[0.0] * 384 for _ in texts]

    def embed_query(self, text):
        return [0.0] * 384


class _MockReranker(RerankerProvider):
    """按指定顺序重排。返回的 (index, score) 列表决定重排结果。"""

    def __init__(self, order: list[int] | None = None):
        self._order = order

    def rerank(self, query, documents, top_k=5):
        if self._order is not None:
            return [(i, 1.0 - i * 0.1) for i in self._order]
        return [(i, 1.0 - i * 0.1) for i in range(min(top_k, len(documents)))]


def _make_chunk(cid: str, content: str = "content") -> Chunk:
    return Chunk(id=cid, content=content, metadata={"document_id": "doc1"})


# ================================================================
# Step 1: Retriever
# ================================================================


def test_retriever_vector_only():
    """只有向量检索有结果时，能正常返回。"""
    chunks = [_make_chunk(f"c{i}") for i in range(3)]
    vec_store = _MockVectorStore([(c, 0.9 - i * 0.1) for i, c in enumerate(chunks)])
    ft_store = _MockFullTextStore([])

    retriever = Retriever(vec_store, ft_store, _MockEmbedding(), final_top_k=5)
    results = retriever.retrieve("test query")

    assert len(results) == 3
    assert [r.id for r in results] == ["c0", "c1", "c2"]


def test_retriever_fulltext_only():
    """只有全文检索有结果时，能正常返回。"""
    chunks = [_make_chunk(f"c{i}") for i in range(3)]
    vec_store = _MockVectorStore([])
    ft_store = _MockFullTextStore([(c, 5.0 - i) for i, c in enumerate(chunks)])

    retriever = Retriever(vec_store, ft_store, _MockEmbedding(), final_top_k=5)
    results = retriever.retrieve("test query")

    assert len(results) == 3
    assert [r.id for r in results] == ["c0", "c1", "c2"]


def test_retriever_rrf_fusion():
    """两个检索器都有结果，RRF 融合后排序正确。"""
    c0 = _make_chunk("c0")
    c1 = _make_chunk("c1")
    c2 = _make_chunk("c2")

    vec_store = _MockVectorStore([(c0, 0.9), (c1, 0.8), (c2, 0.7)])
    ft_store = _MockFullTextStore([(c2, 5.0), (c0, 4.0), (c1, 3.0)])

    retriever = Retriever(vec_store, ft_store, _MockEmbedding(), rrf_k=10, final_top_k=5)
    results = retriever.retrieve("test query")

    assert len(results) == 3
    ranks = {c.id: i for i, c in enumerate(results)}
    assert ranks["c0"] < ranks["c1"]
    assert ranks["c0"] < ranks["c2"]


def test_retriever_dedup():
    """同一个 chunk 出现在两个结果集时，去重后只返回一个。"""
    c0 = _make_chunk("c0")
    c1 = _make_chunk("c1")

    vec_store = _MockVectorStore([(c0, 0.9), (c1, 0.8)])
    ft_store = _MockFullTextStore([(c0, 5.0), (c1, 4.0)])

    retriever = Retriever(vec_store, ft_store, _MockEmbedding(), final_top_k=5)
    results = retriever.retrieve("test query")

    assert len(results) == 2


def test_retriever_final_top_k():
    """结果数量超过 final_top_k 时截断。"""
    chunks = [_make_chunk(f"c{i}") for i in range(10)]
    vec_store = _MockVectorStore([(c, 1.0 - i * 0.05) for i, c in enumerate(chunks)])
    ft_store = _MockFullTextStore([])

    retriever = Retriever(vec_store, ft_store, _MockEmbedding(), final_top_k=3)
    results = retriever.retrieve("test query")

    assert len(results) == 3


# ================================================================
# Step 2: Reranker
# ================================================================


def test_reranker_returns_top_k():
    """重排后返回 top_k 个。"""
    chunks = [_make_chunk(f"c{i}") for i in range(8)]
    reranker = Reranker(_MockReranker())

    results = reranker.rerank("query", chunks, top_k=5)

    assert len(results) == 5


def test_reranker_respects_provider_order():
    """重排顺序由 RerankerProvider 返回的索引决定。"""
    chunks = [_make_chunk(f"c{i}") for i in range(5)]
    mock_rerank = _MockReranker(order=[3, 1, 4, 0, 2])
    reranker = Reranker(mock_rerank)

    results = reranker.rerank("query", chunks, top_k=5)

    assert [r.id for r in results] == ["c3", "c1", "c4", "c0", "c2"]


def test_reranker_truncates_to_top_k():
    """provider 返回超过 top_k 个时截断。"""
    chunks = [_make_chunk(f"c{i}") for i in range(10)]
    mock_rerank = _MockReranker(order=[9, 8, 7, 6, 5, 4, 3, 2, 1, 0])
    reranker = Reranker(mock_rerank)

    results = reranker.rerank("query", chunks, top_k=3)

    assert len(results) == 3
    assert [r.id for r in results] == ["c9", "c8", "c7"]


def test_reranker_empty_input():
    """空输入返回空列表。"""
    reranker = Reranker(_MockReranker())
    results = reranker.rerank("query", [], top_k=5)
    assert results == []