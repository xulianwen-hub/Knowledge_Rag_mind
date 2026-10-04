"""P2-5 ~ P2-7 画像抽取、审核、注入的单元测试。"""

from knowresearch.adapters.memory import (
    PostgresLongTermMemory,
    PostgresMemoryCandidateStore,
)
from knowresearch.core.ports import LLMProvider, TraceStorePort
from knowresearch.core.schemas import MemoryCandidate, Record
from knowresearch.memory import (
    ProfileExtractor,
    ProfileMemoryService,
    ProfileReviewService,
)


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
    db_url = f"sqlite:///{tmp_path / 'memory.db'}"
    candidate_store = PostgresMemoryCandidateStore(db_url)
    long_term_memory = PostgresLongTermMemory(db_url)
    return candidate_store, long_term_memory


def _success_trace(trace_id="trace-1", user_id="user-a"):
    return Record(
        id=trace_id,
        kind="trace",
        content="我主要研究船舶水动力学，希望回答给出公式和页码。",
        metadata={
            "trace_id": trace_id,
            "user_id": user_id,
            "status": "success",
            "question": "我主要研究船舶水动力学，希望回答给出公式和页码。",
            "answer": "好的，我会注意。",
        },
    )


def test_profile_extractor_saves_pending_candidates(tmp_path):
    candidate_store, _ = _make_stores(tmp_path)
    trace_store = _FakeTraceStore([_success_trace()])
    llm = _FakeLLM(
        """
        [
          {"content": "用户研究船舶水动力学", "dimension": "research_field",
           "confidence": 0.9, "evidence": "用户自述研究方向"},
          {"content": "用户希望答案包含公式和页码", "dimension": "output_preference",
           "confidence": 0.8, "evidence": "用户明确表达偏好"}
        ]
        """
    )
    extractor = ProfileExtractor(llm, trace_store, candidate_store)

    candidates = extractor.extract_for_user("user-a")

    assert len(candidates) == 2
    saved = candidate_store.list(user_id="user-a", status="pending")
    assert len(saved) == 2
    assert all(c.source == "trace-1" for c in saved)
    assert {c.metadata["dimension"] for c in saved} == {
        "research_field",
        "output_preference",
    }


def test_profile_extractor_skips_extracted_source(tmp_path):
    candidate_store, _ = _make_stores(tmp_path)
    trace_store = _FakeTraceStore([_success_trace()])
    llm = _FakeLLM(
        '[{"content":"用户研究船舶水动力学","dimension":"research_field",'
        '"confidence":0.9,"evidence":"用户自述"}]'
    )
    extractor = ProfileExtractor(llm, trace_store, candidate_store)

    first = extractor.extract_for_user("user-a")
    second = extractor.extract_for_user("user-a")

    assert len(first) == 1
    assert second == []


def test_profile_review_approve_and_reject(tmp_path):
    candidate_store, long_term_memory = _make_stores(tmp_path)
    review = ProfileReviewService(candidate_store, long_term_memory)
    candidate = MemoryCandidate(
        user_id="user-a",
        content="用户研究船舶水动力学",
        metadata={"dimension": "research_field"},
        source="trace-1",
        confidence=0.9,
    )
    candidate_store.save(candidate)

    profile = review.approve(candidate.id, "user-a")

    assert profile is not None
    assert long_term_memory.get_profile("user-a", "research_field").content == (
        "用户研究船舶水动力学"
    )
    assert candidate_store.get(candidate.id).status == "approved"
    assert candidate_store.get(candidate.id).target_memory_id == profile.id


def test_profile_review_reject(tmp_path):
    candidate_store, long_term_memory = _make_stores(tmp_path)
    review = ProfileReviewService(candidate_store, long_term_memory)
    candidate = MemoryCandidate(
        user_id="user-a",
        content="不稳定的画像",
        metadata={"dimension": "research_field"},
    )
    candidate_store.save(candidate)

    assert review.reject(candidate.id, "user-a", "不准确") is True
    assert candidate_store.get(candidate.id).status == "rejected"
    assert long_term_memory.list(user_id="user-a", kind="profile") == []


def test_profile_review_edit_and_approve(tmp_path):
    candidate_store, long_term_memory = _make_stores(tmp_path)
    review = ProfileReviewService(candidate_store, long_term_memory)
    candidate = MemoryCandidate(
        user_id="user-a",
        content="用户研究船舶",
        metadata={"dimension": "research_field"},
    )
    candidate_store.save(candidate)

    profile = review.edit_and_approve(
        candidate.id,
        "user-a",
        content="用户研究船舶水动力学",
        metadata={"confidence": 0.95},
        review_note="补充完善",
    )

    assert profile is not None
    assert long_term_memory.get_profile("user-a", "research_field").content == (
        "用户研究船舶水动力学"
    )


def test_profile_review_rejects_wrong_user(tmp_path):
    candidate_store, long_term_memory = _make_stores(tmp_path)
    review = ProfileReviewService(candidate_store, long_term_memory)
    candidate = MemoryCandidate(
        user_id="user-a",
        content="用户画像",
        metadata={"dimension": "research_field"},
    )
    candidate_store.save(candidate)

    assert review.approve(candidate.id, "user-b") is None
    assert candidate_store.get(candidate.id).status == "pending"


def test_profile_memory_service_injects_context(tmp_path):
    _, long_term_memory = _make_stores(tmp_path)
    long_term_memory.upsert_profile(
        user_id="user-a",
        content="用户研究船舶水动力学和襟翼舵",
        dimension="research_field",
        confidence=0.9,
    )
    service = ProfileMemoryService(long_term_memory)

    rewritten = service.rewrite_query("user-a", "它有什么优化方法？")
    assert "船舶水动力学" in rewritten

    original = "请详细比较不同艇型优化方法的优缺点"
    assert service.rewrite_query("user-a", original) == original

    context = service.build_context("user-a", "优化方法")
    assert "research_field" in context
    assert "船舶水动力学" in context
