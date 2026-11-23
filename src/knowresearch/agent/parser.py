"""ReAct 决策解析：把 LLM 输出解析成 AgentDecision。

容错点（这几处不做，ReAct 就会经常崩）：
- 模型可能把 JSON 包在 ```json 代码块里；
- 可能前后带解释性文字；
- 字段名可能是 action / action_input / arguments 等变体。
解析失败返回 None，由调用方决定收尾或降级。
"""

from __future__ import annotations

import json
import re

from knowresearch.core.schemas import AgentDecision


_FENCE_PATTERN = re.compile(r"^```[a-zA-Z]*\s*|\s*```$")


def _strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = _FENCE_PATTERN.sub("", text)
    return text.strip()


def parse_decision(raw: str) -> AgentDecision | None:
    """从模型输出中解析出一个 AgentDecision，失败返回 None。"""
    if not raw or not raw.strip():
        return None

    text = _strip_code_fence(raw)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        return None

    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None

    thought = str(data.get("thought") or data.get("reasoning") or "")
    final_answer = str(
        data.get("final_answer") or data.get("final") or ""
    ).strip()
    action = str(
        data.get("action") or data.get("tool") or data.get("action_name") or ""
    ).strip()
    action_input = (
        data.get("action_input")
        or data.get("arguments")
        or data.get("args")
        or data.get("input")
        or {}
    )
    if not isinstance(action_input, dict):
        action_input = {"input": action_input}

    if final_answer:
        return AgentDecision(thought=thought, final_answer=final_answer)
    if action:
        return AgentDecision(
            thought=thought, action=action, action_input=action_input
        )
    return None
