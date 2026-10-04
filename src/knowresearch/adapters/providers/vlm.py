"""VLMProvider 适配器：qwen3.8-omni-flash API 实现。

通过 DashScope 兼容模式调用，用 openai SDK。
支持图片输入（base64 或 URL）和文本 prompt。
"""

import base64
from pathlib import Path
from typing import Any

from openai import OpenAI

from knowresearch.config import settings
from knowresearch.core.ports import VLMProvider


class QwenVLM(VLMProvider):
    """基于 qwen3.8-omni-flash API 的 VLM 适配器。"""

    def __init__(self):
        self._client = OpenAI(
            api_key=settings.vlm.api_key,
            base_url=settings.vlm.base_url,
        )
        self._model = settings.vlm.model

    def _encode_image(self, image_path: str) -> str:
        """图片 → base64 字符串。"""
        data = Path(image_path).read_bytes()
        return base64.b64encode(data).decode("utf-8")

    def _call(self, image_path: str, prompt: str) -> str:
        """调用 VLM API，返回文本结果。"""
        image_base64 = self._encode_image(image_path)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
        )
        return response.choices[0].message.content or ""

    def describe_image(self, image_path: str, prompt: str = "") -> str:
        if not prompt:
            prompt = "请详细描述这张图片的内容。"
        return self._call(image_path, prompt)

    def extract_structure(self, image_path: str) -> dict[str, Any]:
        """提取结构化信息。返回 {"raw_text": str, "structure": ...}。"""
        prompt = (
            "请提取这张图片中的结构化信息（表格数据、图表数据等）。"
            "如果是表格，返回行列数据；如果是图表，返回数据点。"
            "用 JSON 格式输出。"
        )
        raw = self._call(image_path, prompt)
        return {"raw_text": raw, "structure": raw}