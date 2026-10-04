"""技能匹配与注入服务。"""

from __future__ import annotations

import json
import re
from typing import Any

import jieba

from knowresearch.core.ports import SkillContextPort, SkillStorePort
from knowresearch.core.schemas import Record, SkillCandidate


class SkillContextService(SkillContextPort):
    """从正式技能库中匹配技能，并构建注入 prompt 的上下文。"""

    def __init__(
        self,
        skill_store: SkillStorePort,
        min_score: float = 0.05,
    ):
        self.skill_store = skill_store
        self.min_score = min_score

    def match(
        self,
        user_id: str,
        query: str,
        top_k: int = 3,
    ) -> list[Record]:
        if not query:
            return []
        skills = self.skill_store.list_all(
            user_id=user_id,
            status="approved",
        )
        query_tokens = _tokenize(query)
        ranked = []
        for skill in skills:
            skill_tokens = _tokenize(_skill_text(skill))
            score = _jaccard(query_tokens, skill_tokens)
            if score >= self.min_score:
                ranked.append((score, skill))
        ranked.sort(key=lambda item: item[0], reverse=True)
        return [skill for _, skill in ranked[:top_k]]

    def build_context(
        self,
        user_id: str,
        query: str,
        max_chars: int = 1200,
    ) -> str:
        try:
            skills = self.match(user_id, query, top_k=3)
        except Exception:
            return ""
        if not skills:
            return ""

        lines: list[str] = []
        used = 0
        for skill in skills:
            line = _format_skill(skill)
            if used + len(line) > max_chars:
                break
            lines.append(line)
            used += len(line)
        return "\n\n".join(lines)


class StaticSkillContext(SkillContextPort):
    """固定注入一个技能，用于 M5-3 回放验证。"""

    def __init__(self, skill: SkillCandidate | Record):
        self.skill = skill

    def match(
        self,
        user_id: str,
        query: str,
        top_k: int = 3,
    ) -> list[Record]:
        if isinstance(self.skill, Record):
            return [self.skill]
        return []

    def build_context(
        self,
        user_id: str,
        query: str,
        max_chars: int = 1200,
    ) -> str:
        text = _format_skill(self.skill)
        return text[:max_chars]


def _skill_text(skill: SkillCandidate | Record) -> str:
    if isinstance(skill, SkillCandidate):
        parts = [
            skill.name,
            skill.description,
            skill.trigger,
            " ".join(skill.steps),
            " ".join(skill.tags),
        ]
    else:
        metadata = skill.metadata
        parts = [
            str(metadata.get("name", "")),
            str(metadata.get("description", "")),
            str(metadata.get("trigger", "")),
            " ".join(str(step) for step in metadata.get("steps", [])),
            " ".join(str(tag) for tag in metadata.get("tags", [])),
        ]
    return " ".join(part for part in parts if part)


def _format_skill(skill: SkillCandidate | Record) -> str:
    if isinstance(skill, SkillCandidate):
        name = skill.name
        description = skill.description
        trigger = skill.trigger
        steps = skill.steps
        retrieval_params = skill.retrieval_params
    else:
        name = skill.metadata.get("name", "")
        description = skill.metadata.get("description", "")
        trigger = skill.metadata.get("trigger", "")
        steps = skill.metadata.get("steps", [])
        retrieval_params = skill.metadata.get("retrieval_params", {})

    lines = [f"- 技能：{name}"]
    if description:
        lines.append(f"  说明：{description}")
    if trigger:
        lines.append(f"  触发：{trigger}")
    if steps:
        step_text = "；".join(
            f"{index}. {step}" for index, step in enumerate(steps, start=1)
        )
        lines.append(f"  步骤：{step_text}")
    if retrieval_params:
        lines.append(
            "  检索参数：" + json.dumps(retrieval_params, ensure_ascii=False)
        )
    return "\n".join(lines)


def _tokenize(text: str) -> set[str]:
    normalized = re.sub(r"[\s，。！？、；：,.!?;:（）()\[\]【】]+", "", text.lower())
    tokens = {token.strip() for token in jieba.lcut(normalized) if token.strip()}
    return {token for token in tokens if len(token) > 1}


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)
