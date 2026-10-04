"""演示模式测试。"""

from knowresearch.adapters.providers.embedding_cache import CachedEmbeddingProvider
from knowresearch.adapters.providers.demo_llm import DemoLLM
from knowresearch.adapters.providers.embedding import MockEmbedding
from knowresearch.adapters.providers.reranker import MockReranker
from knowresearch.adapters.memory import (
    PostgresLongTermMemory,
    PostgresMemoryCandidateStore,
)
from knowresearch.adapters.skills import (
    PostgresSkillCandidateStore,
    PostgresSkillStore,
)
from knowresearch.bootstrap import build_embedding, build_llm, build_reranker
from knowresearch.config import settings
from scripts.demo_seed import _seed_memory_candidates, _seed_skill_data


def test_demo_llm_answers_from_reference():
    llm = DemoLLM()
    answer = llm.generate(
        "【参考资料】\n[1] 《演示论文》（demo.docx 第1页）\n"
        "襟翼舵可以提升升力。\n\n【问题】\n襟翼舵有什么作用？"
    )
    assert "襟翼舵可以提升升力" in answer
    assert "[1]" in answer


def test_demo_llm_handles_json_prompts():
    llm = DemoLLM()
    assert llm.generate('只返回 JSON 数组，字段 "dimension"') == "[]"
    assert llm.generate('只返回 JSON 对象，字段 "retrieval_params"') == "{}"


def test_demo_bootstrap_providers(monkeypatch):
    monkeypatch.setattr(settings.app, "demo_mode", True)
    assert isinstance(build_llm(), DemoLLM)
    assert isinstance(build_reranker(), MockReranker)
    embedding = build_embedding()
    assert isinstance(embedding, CachedEmbeddingProvider)
    assert isinstance(embedding.provider, MockEmbedding)


def test_demo_seed_helpers(tmp_path):
    db_url = f"sqlite:///{tmp_path / 'demo.db'}"
    candidate_store = PostgresMemoryCandidateStore(db_url)
    _seed_memory_candidates(candidate_store)
    assert len(candidate_store.list(user_id="demo-user")) == 2

    skill_candidate_store = PostgresSkillCandidateStore(db_url)
    skill_store = PostgresSkillStore(db_url)
    _seed_skill_data(skill_candidate_store, skill_store)
    assert len(skill_candidate_store.list(user_id="demo-user")) == 1
    assert len(skill_store.list_all(user_id="demo-user", status="approved")) == 1
