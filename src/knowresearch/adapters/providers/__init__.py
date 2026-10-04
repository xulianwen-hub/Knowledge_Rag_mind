"""模型提供商适配器：Embedding / Reranker / LLM / VLM / OCR。"""

from knowresearch.adapters.providers.embedding import MockEmbedding
from knowresearch.adapters.providers.embedding_cache import CachedEmbeddingProvider
from knowresearch.adapters.providers.local_embedding import LocalBGEZhEmbedding
from knowresearch.adapters.providers.local_reranker import LocalBGEReranker
from knowresearch.adapters.providers.llm import DeepSeekLLM
from knowresearch.adapters.providers.demo_llm import DemoLLM
from knowresearch.adapters.providers.ocr import MockOCR
from knowresearch.adapters.providers.reranker import MockReranker
from knowresearch.adapters.providers.vlm import QwenVLM

__all__ = [
    "MockEmbedding",
    "CachedEmbeddingProvider",
    "LocalBGEZhEmbedding",
    "LocalBGEReranker",
    "MockReranker",
    "DeepSeekLLM",
    "DemoLLM",
    "QwenVLM",
    "MockOCR",
]
