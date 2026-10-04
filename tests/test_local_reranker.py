"""LocalBGEReranker 的单元测试。

同样注入假 CrossEncoder，避免真实模型下载，重点验证端口约定和排序逻辑。
"""

from knowresearch.adapters.providers.local_reranker import LocalBGEReranker


class _FakeCrossEncoder:
    def __init__(self, scores=None):
        self._scores = scores or [0.2, 0.8, 0.5]

    def predict(self, pairs):
        return self._scores[: len(pairs)]


def _make_reranker(scores=None) -> LocalBGEReranker:
    provider = LocalBGEReranker()
    provider._model = _FakeCrossEncoder(scores)
    return provider


def test_rerank_returns_descending_scores():
    provider = _make_reranker(scores=[0.2, 0.9, 0.5])
    result = provider.rerank("query", ["a", "b", "c"], top_k=3)

    assert result == [(1, 0.9), (2, 0.5), (0, 0.2)]


def test_rerank_respects_top_k():
    provider = _make_reranker(scores=[0.1, 0.9, 0.7, 0.8])
    result = provider.rerank("query", ["a", "b", "c", "d"], top_k=2)

    assert len(result) == 2
    assert result[0] == (1, 0.9)
    assert result[1] == (3, 0.8)


def test_rerank_empty_documents_does_not_require_model():
    provider = LocalBGEReranker()
    assert provider.rerank("query", []) == []


def test_local_model_dir_detection(tmp_path):
    valid_dir = tmp_path / "valid"
    valid_dir.mkdir()
    (valid_dir / "config.json").write_text("{}", encoding="utf-8")
    assert LocalBGEReranker._is_local_model_dir(valid_dir) is True

    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    assert LocalBGEReranker._is_local_model_dir(empty_dir) is False
