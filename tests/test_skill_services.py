"""M5-1 / M5-2 技能抽取、存储和审核单元测试。"""

from knowresearch.adapters.skills import (
    PostgresSkillCandidateStore,
    PostgresSkillStore,
)
from knowresearch.core.ports import LLMProvider, TraceStorePort
from knowresearch.core.schemas import Record, SkillCandidate
from knowresearch.skills import SkillExtractor, SkillReviewService


class _FakeLLM(LLMProvider):
    def __init__(self, response: str):
        self.response = response

    def generate(self, prompt, system_prompt="", temperature=0.7, max_tokens=2048):
        return self.response

    def generate_with_messages(self, messages, temperature=0.7, max_tokens=2048):
        return self.response


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


def _make_stores(tmp_path):
    db_url = f"sqlite:///{tmp_path / 'skills.db'}"
    candidate_store = PostgresSkillCandidateStore(db_url)
    skill_store = PostgresSkillStore(db_url)
    return candidate_store, skill_store


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
            "answer": "回答",
        },
    )


SKILL_RESPONSE = """
{
  "name": "艇型对比分析",
  "description": "比较不同艇型优化方法时，先拆解目标再对比优缺点。",
  "trigger": "用户要求比较不同艇型或优化方法",
  "steps": ["识别对比对象", "统一评价维度", "给出适用场景"],
  "retrieval_params": {"top_k": 10, "use_rerank": true},
  "tags": ["艇型", "对比分析"],
  "confidence": 0.85,
  "evidence": "多条轨迹都在问不同艇型优化方法的比较"
}
"""


def test_skill_extractor_clusters_repeated_questions(tmp_path):
    candidate_store, _ = _make_stores(tmp_path)
    traces = [
        _trace("t1", "请比较不同艇型优化方法的优缺点"),
        _trace("t2", "帮我比较不同艇型优化方法的优缺点"),
        _trace("t3", "今天天气怎么样"),
    ]
    extractor = SkillExtractor(
        _FakeLLM(SKILL_RESPONSE),
        _FakeTraceStore(traces),
        candidate_store,
        min_occurrences=2,
    )

    candidates = extractor.extract_for_user("user-a")

    assert len(candidates) == 1
    assert candidates[0].name == "艇型对比分析"
    assert candidates[0].status == "pending"
    assert set(candidates[0].source_trace_ids) == {"t1", "t2"}
    assert len(candidate_store.list(user_id="user-a", status="pending")) == 1


def test_skill_extractor_skips_already_extracted_source(tmp_path):
    candidate_store, _ = _make_stores(tmp_path)
    traces = [
        _trace("t1", "请比较不同艇型优化方法的优缺点"),
        _trace("t2", "帮我比较不同艇型优化方法的优缺点"),
    ]
    extractor = SkillExtractor(
        _FakeLLM(SKILL_RESPONSE),
        _FakeTraceStore(traces),
        candidate_store,
        min_occurrences=2,
    )

    assert len(extractor.extract_for_user("user-a")) == 1
    assert extractor.extract_for_user("user-a") == []


def test_skill_review_approve(tmp_path):
    candidate_store, skill_store = _make_stores(tmp_path)
    review = SkillReviewService(candidate_store, skill_store)
    candidate = SkillCandidate(
        user_id="user-a",
        name="检索式文献对比",
        description="先检索再对比",
        trigger="用户要求对比",
        steps=["检索", "对比"],
        source_trace_ids=["t1", "t2"],
    )
    candidate_store.save(candidate)

    skill = review.approve(candidate.id, "user-a", review_note="确认")

    assert skill is not None
    assert skill.metadata["status"] == "approved"
    assert skill.metadata["name"] == "检索式文献对比"
    assert skill_store.get(skill.id).metadata["status"] == "approved"
    assert candidate_store.get(candidate.id).status == "approved"
    assert candidate_store.get(candidate.id).target_skill_id == skill.id


def test_skill_review_reject_and_user_isolation(tmp_path):
    candidate_store, skill_store = _make_stores(tmp_path)
    review = SkillReviewService(candidate_store, skill_store)
    candidate = SkillCandidate(user_id="user-a", name="技能", description="说明")
    candidate_store.save(candidate)

    assert review.approve(candidate.id, "user-b") is None
    assert review.reject(candidate.id, "user-a", "不合适") is True
    assert candidate_store.get(candidate.id).status == "rejected"
    assert skill_store.list_all(user_id="user-a") == []


def test_skill_store_status_and_usage_fields(tmp_path):
    _, skill_store = _make_stores(tmp_path)
    skill = Record(
        kind="skill",
        content="技能说明",
        metadata={
            "user_id": "user-a",
            "name": "技能A",
            "status": "approved",
            "usage_count": 3,
            "success_count": 2,
        },
    )
    skill_store.save(skill)

    got = skill_store.get(skill.id)
    assert got.metadata["usage_count"] == 3
    assert len(skill_store.list_all(user_id="user-a", status="approved")) == 1

    skill_store.update_status(skill.id, "disabled", user_id="user-a")
    assert skill_store.get(skill.id).metadata["status"] == "disabled"
