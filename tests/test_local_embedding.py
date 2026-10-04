"""LocalBGEZhEmbedding 的单元测试。

为避免在单测里下载真实模型，这里注入一个假的 SentenceTransformer 对象，
重点验证适配器的调用约定、维度、返回结构和懒加载边界。
"""

import numpy as np

from knowresearch.adapters.providers.local_embedding import LocalBGEZhEmbedding


class _FakeSentenceTransformer:
    def __init__(self, dimension: int = 3):
        self._dimension = dimension

    def get_sentence_embedding_dimension(self) -> int:
        return self._dimension

    def encode(self, texts, normalize_embeddings=True):
        return np.array(
            [[float(i + 1) / 10.0] * self._dimension for i in range(len(texts))]
        )


def _make_embedding(dimension: int = 3) -> LocalBGEZhEmbedding:
    provider = LocalBGEZhEmbedding()
    provider._model = _FakeSentenceTransformer(dimension)
    return provider


def test_dimension_from_model():
    provider = _make_embedding(dimension=8)
    assert provider.dimension == 8


def test_embed_documents_returns_right_shape():
    provider = _make_embedding(dimension=4)
    vectors = provider.embed_documents(["文本一", "文本二"])

    assert len(vectors) == 2
    assert all(len(v) == 4 for v in vectors)
    assert all(isinstance(x, float) for v in vectors for x in v)


def test_embed_query_returns_one_vector():
    provider = _make_embedding(dimension=4)
    vector = provider.embed_query("查询文本")

    assert len(vector) == 4
    assert isinstance(vector[0], float)


def test_embed_documents_empty_does_not_require_model():
    provider = LocalBGEZhEmbedding()
    assert provider.embed_documents([]) == []


def test_local_model_dir_detection(tmp_path):
    valid_dir = tmp_path / "valid"
    valid_dir.mkdir()
    (valid_dir / "config.json").write_text("{}", encoding="utf-8")
    assert LocalBGEZhEmbedding._is_local_model_dir(valid_dir) is True

    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    assert LocalBGEZhEmbedding._is_local_model_dir(empty_dir) is False
