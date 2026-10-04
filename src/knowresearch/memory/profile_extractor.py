"""用户画像候选抽取服务。

从 PostgreSQL traces 中读取问答轨迹，用 LLM 抽取稳定用户画像，
写入 memory_candidates，等待用户审核。
"""

from __future__ import annotations

import json
import re
from typing import Any

from knowresearch.core.ports import (
    LLMProvider,
    MemoryCandidateStorePort,
    TraceStorePort,
)
from knowresearch.core.schemas import MemoryCandidate


PROFILE_EXTRACTION_SYSTEM_PROMPT = (
    "你是用户画像抽取器。只抽取长期稳定、对后续问答有帮助的用户信息，"
    "不要抽取论文事实、术语定义或一次性问题。"
    "当前只允许抽取 profile，例如研究方向、研究主题、方法偏好、工具偏好、"
    "输出偏好和长期约束。"
)

PROFILE_EXTRACTION_PROMPT = (
    "请从下面的问答轨迹中抽取用户画像候选。\n"
    "只返回 JSON 数组，不要输出解释。每个元素格式：\n"
    '{{"content": "用户画像描述", "dimension": "research_field|research_topics|'
    'method_preference|tool_preference|output_preference|constraint", '
    '"confidence": 0.0, "evidence": "对话中的依据"}}\n'
    "如果没有稳定画像，返回 []。\n\n"
    "【问答轨迹】\n{traces}\n"
)


class ProfileExtractor:
    """从 traces 抽取候选用户画像。"""

    def __init__(
        self,
        llm: LLMProvider,
        trace_store: TraceStorePort,
        candidate_store: MemoryCandidateStorePort,
    ):
        self.llm = llm
        self.trace_store = trace_store
        self.candidate_store = candidate_store

    def extract_for_user(
        self,
        user_id: str,
        trace_limit: int = 20,
    ) -> list[MemoryCandidate]:
        traces = self._load_traces(user_id, trace_limit)
        if not traces:
            return []

        candidates: list[MemoryCandidate] = []
        seen: set[tuple[str, str]] = set()
        for trace in traces:
            trace_text = self._format_traces([trace])
            prompt = PROFILE_EXTRACTION_PROMPT.format(traces=trace_text)
            try:
                raw = self.llm.generate(
                    prompt,
                    system_prompt=PROFILE_EXTRACTION_SYSTEM_PROMPT,
                    temperature=0.1,
                    max_tokens=1024,
                )
            except Exception:
                continue

            items = self._parse_items(raw)
            for item in items:
                candidate = self._to_candidate(user_id, item, trace.id)
                if candidate is None:
                    continue
                key = (
                    str(candidate.metadata.get("dimension", "")),
                    candidate.content,
                )
                if key in seen:
                    continue
                seen.add(key)
                self.candidate_store.save(candidate)
                candidates.append(candidate)
        return candidates

    def _load_traces(self, user_id: str, limit: int) -> list:
        try:
            traces = self.trace_store.list({"user_id": user_id})
        except Exception:
            return []

        usable = []
        for trace in traces:
            if trace.metadata.get("user_id") != user_id:
                continue
            if trace.metadata.get("status") != "success":
                continue
            if self.candidate_store.exists_for_source(user_id, trace.id):
                continue
            usable.append(trace)
            if len(usable) >= limit:
                break
        return usable

    @staticmethod
    def _format_traces(traces: list) -> str:
        lines: list[str] = []
        for index, trace in enumerate(traces, start=1):
            question = trace.metadata.get("question") or trace.content
            answer = trace.metadata.get("answer", "")
            lines.append(f"[{index}] 用户：{question}\n助手：{answer}")
        return "\n\n".join(lines)

    @staticmethod
    def _parse_items(raw: str) -> list[dict[str, Any]]:
        text = raw.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        start = text.find("[")
        end = text.rfind("]")
        if start == -1 or end == -1 or end <= start:
            return []
        try:
            data = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return []
        return data if isinstance(data, list) else []

    @staticmethod
    def _to_candidate(
        user_id: str,
        item: Any,
        source: str,
    ) -> MemoryCandidate | None:
        if not isinstance(item, dict):
            return None
        content = str(item.get("content", "")).strip()
        dimension = str(item.get("dimension", "")).strip()
        evidence = str(item.get("evidence", "")).strip()
        if not content or not dimension:
            return None
        try:
            confidence = float(item.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))
        return MemoryCandidate(
            user_id=user_id,
            kind="profile",
            content=content,
            metadata={"dimension": dimension},
            source=source,
            evidence=evidence,
            confidence=confidence,
            status="pending",
            proposed_action="add",
        )
