"""演示模式 LLM：不调用外部 API，从参考资料中生成简单答案。"""

from __future__ import annotations

import re

from knowresearch.core.ports import LLMProvider


class DemoLLM(LLMProvider):
    """离线演示用 LLM。"""

    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str:
        if "只返回 JSON 数组" in prompt or '"dimension"' in prompt:
            return "[]"
        if "只返回 JSON 对象" in prompt or '"retrieval_params"' in prompt:
            return "{}"
        if "faithfulness" in prompt and "relevancy" in prompt:
            return (
                '{"faithfulness": 0.85, "relevancy": 0.85, '
                '"reason": "演示模式固定评分"}'
            )
        if "压缩成一段摘要" in prompt or "历史对话摘要" in prompt:
            return "历史对话摘要"
        return self._answer_from_reference(prompt)

    def generate_with_messages(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str:
        user_text = ""
        for message in reversed(messages):
            if message.get("role") == "user":
                user_text = message.get("content", "")
                break
        return self._answer_from_reference(user_text)

    @staticmethod
    def _answer_from_reference(prompt: str) -> str:
        if "【参考资料】" not in prompt:
            return "这是演示模式回答。请配置真实 LLM 以获得完整生成能力。"

        snippets = re.findall(
            r"\[(\d+)\]\s*《(.*?)》.*?\n(.*?)(?=\n\n\[\d+\]|\n\n【问题】|$)",
            prompt,
            flags=re.S,
        )
        if not snippets:
            return "未找到相关内容。"

        lines = ["根据演示知识库中的参考资料，可以得到以下信息："]
        for index, title, content in snippets[:3]:
            clean = _clean_content(content)[:180]
            lines.append(f"- {clean} [{index}]")
        lines.append("以上内容来自演示资料，仅用于展示完整链路。")
        return "\n".join(lines)


def _clean_content(content: str) -> str:
    return re.sub(r"\s+", " ", content).strip()
