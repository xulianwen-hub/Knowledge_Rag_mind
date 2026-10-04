"""答案生成器：调用 LLM 生成带引用的回答。

流程：
1. 接收检索到的 chunks 和对应的 citations
2. 用 prompt 模板拼装 messages
3. 调用 LLMProvider.generate_with_messages
4. 返回 Answer（文本 + 引用列表）

无答案场景：如果没有任何检索结果，直接返回"未找到相关内容"。
"""

import re

from pydantic import BaseModel

from knowresearch.core.generation.prompt import build_messages
from knowresearch.core.ports import LLMProvider
from knowresearch.core.retrieval.citation import Citation

NO_ANSWER_TEXT = "未找到相关内容，无法回答这个问题。"

_CITATION_PATTERN = re.compile(r"\[(\d+)\]")


class Answer(BaseModel):
    """生成的答案。"""

    text: str
    citations: list[Citation]
    trace_id: str = ""
    usage: dict = {}


class AnswerGenerator:
    """答案生成器。"""

    def __init__(
        self,
        llm: LLMProvider,
        temperature: float = 0.3,
        max_tokens: int = 2048,
        max_chunk_chars: int = 500,
    ) -> None:
        self.llm = llm
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_chunk_chars = max_chunk_chars

    def generate(
        self,
        question: str,
        chunks: list,
        citations: list[Citation],
        history: list[dict[str, str]] | None = None,
        summary: str = "",
        memory_context: str = "",
        skill_context: str = "",
        tool_context: str = "",
    ) -> Answer:
        """生成答案。

        Args:
            question: 用户问题
            chunks: 检索到的 chunks（保留以备扩展）
            citations: 带溯源的引用列表
            history: 历史对话消息（多轮对话用）
            summary: 早期对话压缩摘要（多轮对话用）
            memory_context: 长期记忆注入内容（当前主要是用户画像）
            skill_context: 匹配到的技能上下文
            tool_context: 工具执行结果上下文

        Returns:
            Answer 对象
        """
        if not citations:
            return Answer(text=NO_ANSWER_TEXT, citations=[])

        messages = build_messages(
            question,
            citations,
            max_chunk_chars=self.max_chunk_chars,
            history=history,
            summary=summary,
            memory_context=memory_context,
            skill_context=skill_context,
            tool_context=tool_context,
        )

        raw_text = self.llm.generate_with_messages(
            messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        used_citations = self._extract_used_citations(raw_text, citations)

        return Answer(
            text=raw_text,
            citations=used_citations,
            usage=getattr(self.llm, "last_usage", {}) or {},
        )

    @staticmethod
    def _extract_used_citations(
        text: str, all_citations: list[Citation]
    ) -> list[Citation]:
        """从生成文本中提取实际引用到的编号，返回对应的 Citation 列表。

        如果文本中没有引用标记，则返回全部 citations（保守策略）。
        """
        matches = _CITATION_PATTERN.findall(text)
        if not matches:
            return all_citations

        used_indices = set()
        for m in matches:
            try:
                idx = int(m)
                if 1 <= idx <= len(all_citations):
                    used_indices.add(idx)
            except ValueError:
                continue

        if not used_indices:
            return all_citations

        return [
            cite for cite in all_citations if cite.index in used_indices
        ]
