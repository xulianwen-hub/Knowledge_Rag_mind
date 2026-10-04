"""模型提供商端口：LLM / VLM / Embedding / Reranker / OCR 的抽象接口。

业务代码只依赖这些抽象类，不依赖具体实现（DeepSeek / 本地 bge / PaddleOCR 等）。
换实现 = 写一个新的子类，业务代码零改动。
"""

from abc import ABC, abstractmethod
from typing import Any


class EmbeddingProvider(ABC):
    """文本向量化端口。

    文档和查询可能走不同的编码策略（对称 vs 非对称 embedding），
    所以拆成 embed_documents 和 embed_query 两个方法。
    """

    @property
    @abstractmethod
    def dimension(self) -> int:
        """向量维度，建索引时需要。"""

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """批量向量化文档片段。"""

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """向量化查询语句。"""


class RerankerProvider(ABC):
    """重排端口：对召回的候选片段按相关性打分排序。"""

    @abstractmethod
    def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int = 5,
    ) -> list[tuple[int, float]]:
        """重排。

        Args:
            query: 查询语句
            documents: 候选文档文本列表
            top_k: 返回前 k 个

        Returns:
            list of (原文档索引, 相关性分数)，按分数降序
        """


class LLMProvider(ABC):
    """大语言模型端口：文本生成。"""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str:
        """单次生成。"""

    @abstractmethod
    def generate_with_messages(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str:
        """多轮对话生成。messages 格式：[{"role": "user"/"assistant"/"system", "content": "..."}]"""


class VLMProvider(ABC):
    """视觉语言模型端口：图表 / 笔记 / 图片的结构化理解。"""

    @abstractmethod
    def describe_image(self, image_path: str, prompt: str = "") -> str:
        """对图片进行自由描述。"""

    @abstractmethod
    def extract_structure(self, image_path: str) -> dict[str, Any]:
        """从图片中提取结构化信息（表格、图表数据等）。"""


class OCRProvider(ABC):
    """OCR 端口：扫描件 / 手写文字识别。"""

    @abstractmethod
    def recognize(self, image_path: str) -> list[dict[str, Any]]:
        """识别图片中的文字。

        Returns:
            list of {"text": str, "bbox": [x1, y1, x2, y2], "confidence": float}
        """