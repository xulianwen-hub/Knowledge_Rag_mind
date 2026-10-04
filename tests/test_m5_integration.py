"""M5-1 / M5-2 真实 PostgreSQL 集成测试。

默认跳过，Docker Compose 服务启动后手动开启：
    $env:RUN_M5_INTEGRATION='1'
    python -m pytest tests/test_m5_integration.py -q
"""

import os
import uuid

import pytest

from knowresearch.bootstrap import (
    build_production_skill_candidate_store,
    build_production_skill_store,
    build_production_trace_store,
    build_skill_context_service,
    build_skill_extractor,
    build_skill_lifecycle_service,
    build_skill_review_service,
)
from knowresearch.core.generation.answer_generator import Answer
from knowresearch.core.ports import LLMProvider
from knowresearch.core.retrieval.citation import Citation
from knowresearch.core.schemas import Record
from knowresearch.skills import SkillReplayService


pytestmark = pytest.mark.skipif(
    os.getenv("RUN_M5_INTEGRATION") != "1",
    reason="需要本地 PostgreSQL 服务，设置 RUN_M5_INTEGRATION=1 后运行",
)


class _FakeLLM(LLMProvider):
    def generate(self, prompt, system_prompt="", temperature=0.7, max_tokens=2048):
        return (
            '{"name":"艇型对比分析","description":"比较艇型优化方法",'
            '"trigger":"用户要求比较不同艇型","steps":["统一评价维度","对比优缺点"],'
            '"retrieval_params":{"top_k":10,"use_rerank":true},'
            '"tags":["艇型","对比"],"confidence":0.86,"evidence":"多条相似轨迹"}'
        )

    def generate_with_messages(self, messages, temperature=0.7, max_tokens=2048):
        return "回答"


class _FakeReplayEngine:
    def __init__(self, skill_context=None):
        self.skill_context = skill_context

    def with_skill_context(self, skill_context):
        return _FakeReplayEngine(skill_context=skill_context)

    def answer(self, question, user_id=None):
        if self.skill_context is None:
            content = "普通回答"
        else:
            content = "艇型对比分析：统一评价维度，对比优缺点"
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


def _trace(trace_id: str, user_id: str, question: str) -> Record:
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


def test_m5_skill_extract_review_postgres_integration():
    trace_store = build_production_trace_store()
    candidate_store = build_production_skill_candidate_store()
    skill_store = build_production_skill_store()
    user_id = f"skill-user-{uuid.uuid4().hex}"
    trace_ids = [
        f"skill-trace-{uuid.uuid4().hex}",
        f"skill-trace-{uuid.uuid4().hex}",
    ]

    try:
        trace_store.save(_trace(trace_ids[0], user_id, "请比较不同艇型优化方法的优缺点"))
        trace_store.save(_trace(trace_ids[1], user_id, "帮我比较不同艇型优化方法的优缺点"))

        extractor = build_skill_extractor(
            llm_provider=_FakeLLM(),
            trace_store=trace_store,
            candidate_store=candidate_store,
        )
        candidates = extractor.extract_for_user(user_id)

        assert len(candidates) == 1
        candidate = candidates[0]
        assert candidate.status == "pending"
        assert set(candidate.source_trace_ids) == set(trace_ids)

        review = build_skill_review_service(
            candidate_store=candidate_store,
            skill_store=skill_store,
        )
        skill = review.approve(candidate.id, user_id, review_note="确认")

        assert skill is not None
        assert skill.metadata["status"] == "approved"
        assert skill_store.get(skill.id, user_id=user_id) is not None
        assert candidate_store.get(candidate.id).status == "approved"

        context_service = build_skill_context_service(skill_store)
        context = context_service.build_context(user_id, "请比较不同艇型优化方法")
        assert "艇型对比分析" in context

        replay = SkillReplayService(
            _FakeReplayEngine(),
            trace_store,
            candidate_store=candidate_store,
        )
        result = replay.replay_and_store(candidate.id, user_id, sample_size=2)
        assert result is not None
        assert result.improved is True
        assert candidate_store.get(candidate.id).metadata["replay_passed"] is True

        lifecycle = build_skill_lifecycle_service(skill_store)
        lifecycle.record_usage(skill.id, user_id, success=True)
        lifecycle.record_feedback(skill.id, user_id, positive=True)
        updated = lifecycle.update_skill(
            skill.id,
            user_id,
            description="第二版技能说明",
            steps=["统一评价维度", "对比优缺点", "给出推荐场景"],
        )
        assert updated.metadata["version"] == 2
        rolled_back = lifecycle.rollback(skill.id, user_id, target_version=1)
        assert rolled_back is not None
        assert rolled_back.metadata["version"] == 1
        final_skill = skill_store.get(skill.id, user_id=user_id)
        assert final_skill.metadata["usage_count"] == 1
        assert final_skill.metadata["feedback_up"] == 1
    finally:
        for trace_id in trace_ids:
            trace_store.delete(trace_id)
        for candidate in candidate_store.list(user_id=user_id):
            candidate_store.delete(candidate.id)
        for skill in skill_store.list_all(user_id=user_id):
            skill_store.delete(skill.id, user_id=user_id)
