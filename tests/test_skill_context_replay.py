"""M5-3 / M5-4 技能回放与注入的单元测试。"""

from knowresearch.adapters.skills import (
    PostgresSkillCandidateStore,
    PostgresSkillStore,
)
from knowresearch.core.generation.answer_generator import Answer
from knowresearch.core.ports import TraceStorePort
from knowresearch.core.retrieval.citation import Citation
from knowresearch.core.schemas import Record, SkillCandidate
from knowresearch.skills import (
    SkillContextService,
    SkillReplayService,
    StaticSkillContext,
)


class _FakeTraceStore(TraceStorePort):
    def __init__(self, traces):
        self.traces = traces

    def save(self, trace):
        self.traces.append(trace)

    def get(self, trace_id):
        return next((t for t in self.traces if t.id == trace_id), None)

    def list(self, filters=None):
        return list(self.traces)

    def delete(self, trace_id):
        self.traces = [t for t in self.traces if t.id != trace_id]


class _FakeEngine:
    def __init__(self, skill_context=None):
        self.skill_context = skill_context

    def with_skill_context(self, skill_context):
        return _FakeEngine(skill_context=skill_context)

    def answer(self, question, user_id=None):
        if self.skill_context is None:
            content = "普通回答"
        else:
            content = "艇型对比分析：先统一评价维度，再对比优缺点"
        return Answer(
            text=content,
            citations=[
                Citation(
                    chunk_id="c1",
                    document_id="d1",
                    document_title="论文.pdf",
                    content=content,
                    index=1,
                )
            ],
        )


def _trace(trace_id: str, question: str, user_id: str = "user-a") -> Record:
    return Record(
        id=trace_id,
        kind="trace",
        content=question,
        metadata={
            "trace_id": trace_id,
            "user_id": user_id,
            "status": "success",
            "question": question,
            "answer": "旧回答",
        },
    )


def _candidate() -> SkillCandidate:
    return SkillCandidate(
        user_id="user-a",
        name="艇型对比分析",
        description="比较艇型优化方法",
        trigger="用户要求比较不同艇型",
        steps=["统一评价维度", "对比优缺点"],
        tags=["艇型", "对比"],
    )


def test_skill_context_service_matches_and_builds_context(tmp_path):
    store = PostgresSkillStore(f"sqlite:///{tmp_path / 'skills.db'}")
    store.save(
        Record(
            kind="skill",
            content="比较艇型优化方法",
            metadata={
                "user_id": "user-a",
                "name": "艇型对比分析",
                "description": "比较艇型优化方法",
                "trigger": "用户要求比较不同艇型",
                "steps": ["统一评价维度", "对比优缺点"],
                "tags": ["艇型", "对比"],
                "status": "approved",
            },
        )
    )
    service = SkillContextService(store)

    matched = service.match("user-a", "请比较不同艇型优化方法")
    context = service.build_context("user-a", "请比较不同艇型优化方法")

    assert len(matched) == 1
    assert "艇型对比分析" in context
    assert "统一评价维度" in context


def test_static_skill_context_builds_candidate_context():
    context = StaticSkillContext(_candidate()).build_context("user-a", "问题")
    assert "艇型对比分析" in context
    assert "统一评价维度" in context


def test_replay_service_detects_improvement():
    traces = [
        _trace("t1", "请比较不同艇型优化方法"),
        _trace("t2", "帮我比较不同艇型优化方法"),
    ]
    service = SkillReplayService(_FakeEngine(), _FakeTraceStore(traces))

    result = service.replay(_candidate(), "user-a", sample_size=2)

    assert result.sample_size == 2
    assert result.improved is True
    assert result.enhanced_skill_hit_rate > result.baseline_skill_hit_rate
    assert result.enhanced_answered_rate >= result.baseline_answered_rate


def test_replay_and_store_updates_candidate(tmp_path):
    candidate_store = PostgresSkillCandidateStore(
        f"sqlite:///{tmp_path / 'candidates.db'}"
    )
    candidate = _candidate()
    candidate_store.save(candidate)
    service = SkillReplayService(
        _FakeEngine(),
        _FakeTraceStore([_trace("t1", "请比较不同艇型优化方法")]),
        candidate_store=candidate_store,
    )

    result = service.replay_and_store(candidate.id, "user-a", sample_size=1)

    assert result is not None
    saved = candidate_store.get(candidate.id)
    assert saved.metadata["replay_passed"] is True
    assert saved.metadata["replay"]["sample_size"] == 1
