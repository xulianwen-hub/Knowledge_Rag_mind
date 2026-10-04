"""技能回放验证服务。"""

from __future__ import annotations

from pydantic import BaseModel, Field

from knowresearch.core.generation.answer_generator import (
    NO_ANSWER_TEXT,
    Answer,
)
from knowresearch.core.ports import (
    SkillCandidateStorePort,
    TraceStorePort,
)
from knowresearch.core.qa_engine import QAEngine, SERVICE_UNAVAILABLE_TEXT
from knowresearch.core.schemas import SkillCandidate
from knowresearch.skills.skill_context import StaticSkillContext


class SkillReplayResult(BaseModel):
    """一次技能回放的对比结果。"""

    sample_size: int
    baseline_answered_rate: float
    enhanced_answered_rate: float
    baseline_citation_rate: float
    enhanced_citation_rate: float
    baseline_skill_hit_rate: float
    enhanced_skill_hit_rate: float
    improved: bool
    reason: str = ""
    details: list[dict] = Field(default_factory=list)


class SkillReplayService:
    """用历史问题回放技能，自动化验证是否有实际提升。"""

    def __init__(
        self,
        qa_engine: QAEngine,
        trace_store: TraceStorePort,
        candidate_store: SkillCandidateStorePort | None = None,
        min_skill_hit_improvement: float = 0.0,
    ):
        self.qa_engine = qa_engine
        self.trace_store = trace_store
        self.candidate_store = candidate_store
        self.min_skill_hit_improvement = min_skill_hit_improvement

    def replay(
        self,
        candidate: SkillCandidate,
        user_id: str,
        sample_size: int = 5,
    ) -> SkillReplayResult:
        traces = self._load_questions(user_id, sample_size)
        baseline_engine = self.qa_engine.with_skill_context(None)
        enhanced_engine = self.qa_engine.with_skill_context(
            StaticSkillContext(candidate)
        )

        details = []
        for trace in traces:
            question = trace.metadata.get("question") or trace.content
            baseline = baseline_engine.answer(question, user_id=user_id)
            enhanced = enhanced_engine.answer(question, user_id=user_id)
            details.append(
                {
                    "question": question,
                    "baseline_answered": _answered(baseline),
                    "enhanced_answered": _answered(enhanced),
                    "baseline_citations": len(baseline.citations),
                    "enhanced_citations": len(enhanced.citations),
                    "baseline_skill_hit": _skill_hit(baseline, candidate),
                    "enhanced_skill_hit": _skill_hit(enhanced, candidate),
                }
            )

        total = len(details)
        if total == 0:
            return SkillReplayResult(
                sample_size=0,
                baseline_answered_rate=0.0,
                enhanced_answered_rate=0.0,
                baseline_citation_rate=0.0,
                enhanced_citation_rate=0.0,
                baseline_skill_hit_rate=0.0,
                enhanced_skill_hit_rate=0.0,
                improved=False,
                reason="没有可用于回放的历史问题",
            )

        baseline_answered = sum(d["baseline_answered"] for d in details) / total
        enhanced_answered = sum(d["enhanced_answered"] for d in details) / total
        baseline_citations = sum(d["baseline_citations"] > 0 for d in details) / total
        enhanced_citations = sum(d["enhanced_citations"] > 0 for d in details) / total
        baseline_skill_hit = sum(d["baseline_skill_hit"] for d in details) / total
        enhanced_skill_hit = sum(d["enhanced_skill_hit"] for d in details) / total

        improved, reason = self._judge(
            baseline_answered=baseline_answered,
            enhanced_answered=enhanced_answered,
            baseline_citations=baseline_citations,
            enhanced_citations=enhanced_citations,
            baseline_skill_hit=baseline_skill_hit,
            enhanced_skill_hit=enhanced_skill_hit,
        )
        return SkillReplayResult(
            sample_size=total,
            baseline_answered_rate=round(baseline_answered, 4),
            enhanced_answered_rate=round(enhanced_answered, 4),
            baseline_citation_rate=round(baseline_citations, 4),
            enhanced_citation_rate=round(enhanced_citations, 4),
            baseline_skill_hit_rate=round(baseline_skill_hit, 4),
            enhanced_skill_hit_rate=round(enhanced_skill_hit, 4),
            improved=improved,
            reason=reason,
            details=details,
        )

    def replay_and_store(
        self,
        candidate_id: str,
        user_id: str,
        sample_size: int = 5,
    ) -> SkillReplayResult | None:
        if self.candidate_store is None:
            return None
        candidate = self.candidate_store.get(candidate_id)
        if candidate is None or candidate.user_id != user_id:
            return None

        result = self.replay(candidate, user_id, sample_size=sample_size)
        candidate.metadata["replay"] = result.model_dump(mode="json")
        candidate.metadata["replay_passed"] = result.improved
        self.candidate_store.save(candidate)
        return result

    def _load_questions(self, user_id: str, limit: int) -> list:
        try:
            traces = self.trace_store.list({"user_id": user_id})
        except Exception:
            return []
        usable = [
            trace
            for trace in traces
            if trace.metadata.get("user_id") == user_id
            and trace.metadata.get("status") == "success"
        ]
        return usable[:limit]

    def _judge(
        self,
        baseline_answered: float,
        enhanced_answered: float,
        baseline_citations: float,
        enhanced_citations: float,
        baseline_skill_hit: float,
        enhanced_skill_hit: float,
    ) -> tuple[bool, str]:
        if enhanced_answered < baseline_answered:
            return False, "可回答率下降"
        if enhanced_citations < baseline_citations:
            return False, "引用覆盖率下降"
        if enhanced_skill_hit < baseline_skill_hit + self.min_skill_hit_improvement:
            return False, "技能相关内容命中没有提升"
        if enhanced_skill_hit <= baseline_skill_hit:
            return False, "技能没有带来可见提升"
        return True, "技能提升且没有明显回退"


def _answered(answer: Answer) -> bool:
    return answer.text not in (NO_ANSWER_TEXT, SERVICE_UNAVAILABLE_TEXT)


def _skill_hit(answer: Answer, candidate: SkillCandidate) -> bool:
    terms = _skill_terms(candidate)
    if not terms:
        return False
    text = answer.text + "".join(citation.content for citation in answer.citations)
    return any(term in text for term in terms)


def _skill_terms(candidate: SkillCandidate) -> list[str]:
    terms = [candidate.name, candidate.trigger]
    terms.extend(candidate.tags)
    terms.extend(candidate.steps)
    return [term for term in terms if term and len(term) > 1]
