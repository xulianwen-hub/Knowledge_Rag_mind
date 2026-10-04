"""技能候选抽取服务。

确定性代码负责聚类和统计重复次数，LLM 只负责把重复模式归纳成技能候选。
"""

from __future__ import annotations

import json
import re
from typing import Any

import jieba

from knowresearch.core.ports import (
    LLMProvider,
    SkillCandidateStorePort,
    TraceStorePort,
)
from knowresearch.core.schemas import SkillCandidate


SKILL_EXTRACTION_SYSTEM_PROMPT = (
    "你是技能沉淀器。根据一组反复出现的问答轨迹，总结一个可复用的分析技能。"
    "技能应该描述触发条件、分析步骤和推荐检索参数，不要复述具体论文内容。"
)

SKILL_EXTRACTION_PROMPT = (
    "请根据以下相似问答轨迹，总结一个可复用技能。\n"
    "只返回 JSON 对象，不要输出解释。格式：\n"
    '{{"name": "技能名", "description": "技能说明", '
    '"trigger": "什么时候使用", "steps": ["步骤1", "步骤2"], '
    '"retrieval_params": {{"top_k": 10, "use_rerank": true}}, '
    '"tags": ["标签"], "confidence": 0.0, "evidence": "依据"}}\n\n'
    "【相似问答轨迹】\n{traces}\n"
)


class SkillExtractor:
    """从 traces 中抽取候选技能。"""

    def __init__(
        self,
        llm: LLMProvider,
        trace_store: TraceStorePort,
        candidate_store: SkillCandidateStorePort,
        min_occurrences: int = 2,
        similarity_threshold: float = 0.45,
    ):
        self.llm = llm
        self.trace_store = trace_store
        self.candidate_store = candidate_store
        self.min_occurrences = min_occurrences
        self.similarity_threshold = similarity_threshold

    def extract_for_user(
        self,
        user_id: str,
        trace_limit: int = 50,
    ) -> list[SkillCandidate]:
        traces = self._load_traces(user_id, trace_limit)
        clusters = self._cluster_traces(traces)

        candidates: list[SkillCandidate] = []
        for cluster in clusters:
            if len(cluster) < self.min_occurrences:
                continue
            source = cluster[0].id
            if self.candidate_store.exists_for_source(user_id, source):
                continue
            candidate = self._summarize_cluster(user_id, cluster)
            if candidate is None:
                continue
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

    def _cluster_traces(self, traces: list) -> list[list]:
        clusters: list[list] = []
        token_sets: list[set[str]] = []
        for trace in traces:
            tokens = self._tokenize(self._question_of(trace))
            placed = False
            for index, cluster_tokens in enumerate(token_sets):
                if self._jaccard(tokens, cluster_tokens) >= self.similarity_threshold:
                    clusters[index].append(trace)
                    token_sets[index] = cluster_tokens | tokens
                    placed = True
                    break
            if not placed:
                clusters.append([trace])
                token_sets.append(tokens)
        return clusters

    def _summarize_cluster(self, user_id: str, cluster: list) -> SkillCandidate | None:
        trace_text = self._format_traces(cluster)
        prompt = SKILL_EXTRACTION_PROMPT.format(traces=trace_text)
        try:
            raw = self.llm.generate(
                prompt,
                system_prompt=SKILL_EXTRACTION_SYSTEM_PROMPT,
                temperature=0.1,
                max_tokens=1024,
            )
        except Exception:
            return None

        data = self._parse_skill(raw)
        if not data:
            return None
        name = str(data.get("name", "")).strip()
        if not name:
            return None

        steps = data.get("steps", [])
        if not isinstance(steps, list):
            steps = []
        retrieval_params = data.get("retrieval_params", {})
        if not isinstance(retrieval_params, dict):
            retrieval_params = {}
        tags = data.get("tags", [])
        if not isinstance(tags, list):
            tags = []
        try:
            confidence = float(data.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))

        return SkillCandidate(
            user_id=user_id,
            name=name,
            description=str(data.get("description", "")),
            trigger=str(data.get("trigger", "")),
            steps=[str(step) for step in steps],
            retrieval_params=retrieval_params,
            tags=[str(tag) for tag in tags],
            source_trace_ids=[trace.id for trace in cluster],
            evidence=str(data.get("evidence", "")),
            confidence=confidence,
            status="pending",
            proposed_action="add",
        )

    @staticmethod
    def _format_traces(traces: list) -> str:
        lines = []
        for index, trace in enumerate(traces, start=1):
            question = trace.metadata.get("question") or trace.content
            answer = trace.metadata.get("answer", "")
            lines.append(f"[{index}] 用户：{question}\n助手：{answer}")
        return "\n\n".join(lines)

    @staticmethod
    def _parse_skill(raw: str) -> dict[str, Any]:
        text = raw.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return {}
        try:
            data = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return {}
        return data if isinstance(data, dict) else {}

    @staticmethod
    def _question_of(trace) -> str:
        return str(trace.metadata.get("question") or trace.content or "")

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        normalized = re.sub(r"[\s，。！？、；：,.!?;:（）()\[\]【】]+", "", text.lower())
        tokens = {token.strip() for token in jieba.lcut(normalized) if token.strip()}
        return {token for token in tokens if len(token) > 1}

    @staticmethod
    def _jaccard(left: set[str], right: set[str]) -> float:
        if not left or not right:
            return 0.0
        return len(left & right) / len(left | right)
