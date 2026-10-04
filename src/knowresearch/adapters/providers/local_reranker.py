"""本地 BGE Reranker 适配器。

用 sentence-transformers 的 CrossEncoder 加载 bge-reranker 系列模型，
实现 RerankerProvider 端口。

设计要点：
- 延迟加载：只在第一次 rerank 时加载模型。
- 优先使用本地模型目录，目录不存在时回退到 model_name 并缓存到 model_dir。
- 返回结果保持 (原文档索引, 相关性分数) 的端口约定，分数降序。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from knowresearch.config import settings
from knowresearch.core.ports import RerankerProvider


class LocalBGEReranker(RerankerProvider):
    """基于 CrossEncoder 的本地中文 Reranker。"""

    def __init__(
        self,
        model_name: str | None = None,
        model_dir: str | None = None,
        device: str | None = None,
        max_length: int = 512,
    ) -> None:
        self._model_name = model_name or settings.reranker.model_name
        self._model_dir = Path(model_dir or settings.reranker.model_dir)
        self._device = device
        self._max_length = max_length
        self._model: Any | None = None

    def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int = 5,
    ) -> list[tuple[int, float]]:
        if not documents:
            return []

        if self._model is None:
            self._load_model()
        assert self._model is not None

        pairs = [(query, document) for document in documents]
        scores = self._model.predict(pairs)
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: float(scores[index]),
            reverse=True,
        )[:top_k]
        return [(index, float(scores[index])) for index in ranked_indices]

    def _load_model(self) -> None:
        from sentence_transformers import CrossEncoder

        if self._is_local_model_dir(self._model_dir):
            model_source: str | Path = self._model_dir
            self._model = CrossEncoder(
                str(model_source),
                device=self._device,
                max_length=self._max_length,
                tokenizer_args={"use_fast": False},
            )
            return

        self._model = CrossEncoder(
            self._model_name,
            device=self._device,
            max_length=self._max_length,
            cache_dir=str(self._model_dir),
            tokenizer_args={"use_fast": False},
        )

    @staticmethod
    def _is_local_model_dir(model_dir: Path) -> bool:
        """判断 model_dir 是否是已经下载好的 CrossEncoder 模型目录。"""
        if not model_dir.exists() or not model_dir.is_dir():
            return False
        if (model_dir / "config.json").exists():
            return True
        return any(model_dir.glob("*.safetensors")) or any(
            model_dir.glob("pytorch_model*.bin")
        )
