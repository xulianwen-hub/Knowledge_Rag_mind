"""CRAG 式反思：对工具观察结果做质量评估，必要时给出自纠反馈。

面试叙事：普通 RAG 是「检索 → 生成」的固定流水线；Agentic RAG 的
分水岭在于「检索后有没有判断这一步」。Reflector 在每个 ReAct 步骤
执行后评估 Observation 与目标的匹配度，低于阈值时把反思写回
scratchpad，促使 planner 改写查询或换工具，这就是自纠闭环。
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class ReflectionResult:
    """反思结果。"""

    quality: float
    sufficient: bool
    feedback: str = ""
    suggested_query: str = ""


class ReflectorPort(ABC):
    """反思端口：评估一次工具观察是否足以回答目标。"""

    @abstractmethod
    def evaluate(
        self,
        goal: str,
        observation: str,
        action: str,
        action_input: dict,
    ) -> ReflectionResult:
        """返回质量分与是否需要自纠。"""


class HeuristicReflector(ReflectorPort):
    """零成本启发式反思器，生产可替换为 LLMReflector。"""

    def __init__(self, min_quality: float = 0.2, min_chars: int = 40) -> None:
        self.min_quality = min_quality
        self.min_chars = min_chars

    def evaluate(
        self,
        goal: str,
        observation: str,
        action: str,
        action_input: dict,
    ) -> ReflectionResult:
        text = observation or ""
        if not text or "失败" in text or "无返回" in text:
            return ReflectionResult(
                quality=0.0,
                sufficient=False,
                feedback=(
                    "上一步工具没有返回有效内容。请改写查询词（更短、更聚焦、"
                    "换成核心术语）后重试，或改用其他可用工具；仍无结果则诚实说明。"
                ),
            )

        terms = _keywords(goal)
        overlap = (
            sum(1 for term in terms if term in text) / len(terms) if terms else 1.0
        )
        coverage = min(1.0, len(text) / max(self.min_chars, 1))
        quality = round(0.6 * overlap + 0.4 * coverage, 3)
        sufficient = quality >= self.min_quality

        if sufficient:
            return ReflectionResult(quality=quality, sufficient=True)

        return ReflectionResult(
            quality=quality,
            sufficient=False,
            feedback=(
                f"当前观察与目标的匹配度不足（quality={quality}）。"
                "请换一个角度改写查询词，优先覆盖目标里的关键术语；"
                "若本地知识库没有结果，可尝试外部工具或明确告知资料不足。"
            ),
        )


class LLMReflector(ReflectorPort):
    """用 LLM 打分的高质量反思器，失败时降级启发式。"""

    def __init__(
        self,
        llm,
        fallback: ReflectorPort | None = None,
    ) -> None:
        self.llm = llm
        self.fallback = fallback or HeuristicReflector()

    def evaluate(
        self,
        goal: str,
        observation: str,
        action: str,
        action_input: dict,
    ) -> ReflectionResult:
        try:
            prompt = _build_reflection_prompt(
                goal, observation, action, action_input
            )
            raw = self.llm.generate(prompt).strip()
            parsed = _parse_quality(raw)
            if parsed is None:
                return self.fallback.evaluate(
                    goal, observation, action, action_input
                )
            quality, sufficient, feedback = parsed
            return ReflectionResult(
                quality=quality,
                sufficient=sufficient,
                feedback=feedback,
            )
        except Exception:
            return self.fallback.evaluate(goal, observation, action, action_input)


_STOPWORDS = {
    "的",
    "了",
    "是",
    "在",
    "和",
    "与",
    "及",
    "或",
    "对",
    "中",
    "有",
    "这个",
    "那个",
    "什么",
    "怎么",
    "如何",
    "为什么",
}


def _keywords(goal: str) -> list[str]:
    """提取目标中的中英文关键词，用于启发式相关性打分。"""
    tokens = re.findall(r"[A-Za-z0-9_+-]+|[\u4e00-\u9fff]{2,}", goal)
    seen: set[str] = set()
    result: list[str] = []
    for token in tokens:
        token = token.lower()
        if token in _STOPWORDS or token in seen:
            continue
        seen.add(token)
        result.append(token)
    return result[:12] or [goal.strip()]


def _build_reflection_prompt(
    goal: str,
    observation: str,
    action: str,
    action_input: dict,
) -> str:
    return (
        "你是检索质量评估器。请判断下面的工具观察是否足以回答目标问题。\n"
        f"目标：{goal}\n"
        f"工具：{action}，参数：{action_input}\n"
        f"观察：{observation[:800]}\n"
        "只输出 JSON："
        '{"quality": 0到1的小数, "sufficient": true或false, "feedback": "不足时的改进建议"}'
    )


def _parse_quality(raw: str) -> tuple[float, bool, str] | None:
    import json

    raw = raw.strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        data = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return None
    quality = float(data.get("quality", -1))
    if quality < 0 or quality > 1:
        return None
    sufficient = bool(data.get("sufficient", quality >= 0.6))
    feedback = str(data.get("feedback", ""))
    return quality, sufficient, feedback
