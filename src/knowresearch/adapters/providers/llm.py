"""LLMProvider 适配器：DeepSeek API 实现。

DeepSeek 兼容 OpenAI API 格式，用 openai SDK 调用。
从 settings.llm 读取 api_key / base_url / model。
"""

from typing import Iterator

from openai import OpenAI

from knowresearch.config import settings
from knowresearch.core.ports import LLMProvider


class DeepSeekLLM(LLMProvider):
    """基于 DeepSeek API 的 LLM 适配器。"""

    def __init__(self):
        self._client = OpenAI(
            api_key=settings.llm.api_key,
            base_url=settings.llm.base_url,
        )
        self._model = settings.llm.model
        self.last_usage: dict = {}

    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return self.generate_with_messages(messages, temperature, max_tokens)

    def generate_with_messages(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        usage = getattr(response, "usage", None)
        if usage is not None:
            self.last_usage = usage.model_dump()
        else:
            self.last_usage = {}
        return response.choices[0].message.content or ""

    def stream_with_messages(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> Iterator[str]:
        """逐 token 流式生成，yield 每个增量文本片段。

        开启 stream=True 后，服务端会边生成边返回；这里只取 delta.content 非空的片段。
        流式响应默认不带 usage，因此流式调用后 last_usage 可能为空。
        """
        stream = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        self.last_usage = {}
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta