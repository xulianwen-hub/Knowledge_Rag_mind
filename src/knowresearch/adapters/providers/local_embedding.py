"""本地 BGE Embedding 适配器。

用 sentence-transformers 加载本地或 Hugging Face 上的 BGE 中文模型，
实现 EmbeddingProvider 端口。

设计要点：
- 延迟加载：实例化时不加载模型，第一次真正需要向量或维度时才加载。
- 优先使用配置的本地模型目录；目录不存在或不完整时回退到 model_name，
  并把模型缓存到 model_dir，方便后续离线复用。
- 文档向量和查询向量都做归一化，与 InMemoryVectorStore 的余弦相似度约定一致。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from knowresearch.config import settings
from knowresearch.core.ports import EmbeddingProvider


class LocalBGEZhEmbedding(EmbeddingProvider):
    """基于 sentence-transformers 的本地中文 Embedding。"""

    def __init__(
        self,
        model_name: str | None = None,
        model_dir: str | None = None,
        device: str | None = None,
        normalize_embeddings: bool = True,
    ) -> None:
        self._model_name = model_name or settings.embedding.model_name
        self._model_dir = Path(model_dir or settings.embedding.model_dir)
        self._device = device
        self._normalize_embeddings = normalize_embeddings
        self._model: Any | None = None

    @property
    def dimension(self) -> int:
        if self._model is None:
            self._load_model()
        assert self._model is not None
        return int(self._model.get_sentence_embedding_dimension())

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if self._model is None:
            self._load_model()
        assert self._model is not None
        vectors = self._model.encode(
            texts,
            normalize_embeddings=self._normalize_embeddings,
        )
        return [vector.tolist() for vector in vectors]

    def embed_query(self, text: str) -> list[float]:
        vectors = self.embed_documents([text])
        return vectors[0]

    def _load_model(self) -> None:
        from sentence_transformers import SentenceTransformer

        if self._is_local_model_dir(self._model_dir):
            model_source: str | Path = self._model_dir
            self._model = SentenceTransformer(
                str(model_source),
                device=self._device,
            )
            return

        self._model = SentenceTransformer(
            self._model_name,
            device=self._device,
            cache_folder=str(self._model_dir),
        )

    @staticmethod
    def _is_local_model_dir(model_dir: Path) -> bool:
        """判断 model_dir 是否是一个已经下载好的 SentenceTransformer 模型目录。"""
        if not model_dir.exists() or not model_dir.is_dir():
            return False
        required_files = ("config.json", "sentence_bert_config.json")
        if any((model_dir / name).exists() for name in required_files):
            return True
        return any(model_dir.glob("*.safetensors")) or any(
            model_dir.glob("pytorch_model*.bin")
        )
