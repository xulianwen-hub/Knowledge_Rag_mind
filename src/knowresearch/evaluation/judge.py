"""LLM 评审器：评估答案忠实度和相关性。"""

from __future__ import annotations

import json
import re

from pydantic import BaseModel

from knowresearch.core.ports import LLMProvider


class JudgeScore(BaseModel):
    faithfulness: float
    relevancy: float
    reason: str = ""


class LLMJudge:
    def __init__(self, llm: LLMProvider):
        self.llm = llm

    def score(
        self,
        question: str,
        answer_text: str,
        expected_titles: list[str],
    ) -> JudgeScore:
        prompt = (
            "请评估下面的问答结果。只返回 JSON："
            '{"faithfulness": 0.0, "relevancy": 0.0, "reason": "简短理由"}'
            "\nfaithfulness 表示答案是否忠实于参考资料，relevancy 表示是否回答了问题。"
            f"\n问题：{question}\n答案：{answer_text}\n"
            f"期望来源：{', '.join(expected_titles)}"
        )
        try:
            raw = self.llm.generate(prompt, temperature=0.0, max_tokens=256)
            data = _parse_json(raw)
            return JudgeScore(
                faithfulness=_clamp(data.get("faithfulness", 0.0)),
                relevancy=_clamp(data.get("relevancy", 0.0)),
                reason=str(data.get("reason", "")),
            )
        except Exception:
            return JudgeScore(faithfulness=0.0, relevancy=0.0)


def _parse_json(raw: str) -> dict:
    text = raw.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        return {}
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _clamp(value) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, number))
