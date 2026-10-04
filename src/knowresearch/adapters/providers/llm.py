"""LLMProvider 适配器：DeepSeek API 实现。

DeepSeek 兼容 OpenAI API 格式，用 openai SDK 调用。
从 settings.llm 读取 api_key / base_url / model。
"""

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
