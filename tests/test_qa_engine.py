"""M3 Step 5+6 测试：QAEngine 端到端 + 降级场景。"""

import logging

from knowresearch.core.ports import (
    DocumentStorePort,
    EmbeddingProvider,
    FullTextStorePort,
    LLMProvider,
    MemoryContextPort,
    RerankerProvider,
    SessionStorePort,
    SkillContextPort,
    SkillLifecyclePort,
    TraceStorePort,
    VectorStorePort,
)
from knowresearch.core.generation.answer_generator import (
    NO_ANSWER_TEXT,
    AnswerGenerator,
)
from knowresearch.core.qa_engine import CONTEXT_TOKEN_BUDGET, QAEngine, SERVICE_UNAVAILABLE_TEXT
from knowresearch.core.observability import metrics
from knowresearch.core.retrieval.citation import Citation, CitationBuilder
from knowresearch.core.retrieval.reranker import Reranker
from knowresearch.core.retrieval.retriever import Retriever
from knowresearch.core.schemas import (
    Chunk,
    Document,
    Record,
    SectionBlock,
    ToolResult,
)

logging.basicConfig(level=logging.CRITICAL)


# ================================================================
# Mock 实现
# ================================================================


class _MockVectorStore(VectorStorePort):
    def __init__(self, results=None, raise_exc=False):
        self._results = results or []
        self._raise = raise_exc

    def add(self, records, embeddings):
        pass

    def search(self, query_embedding, top_k=10):
        if self._raise:
            raise RuntimeError("向量库挂了")
        return self._results[:top_k]

    def delete(self, record_ids):
        pass

    def clear(self):
        pass


class _MockFullTextStore(FullTextStorePort):
    def __init__(self, results=None, raise_exc=False):
        self._results = results or []
        self._raise = raise_exc

    def add(self, records):
        pass

    def search(self, query, top_k=10):
        if self._raise:
            raise RuntimeError("全文索引挂了")
        return self._results[:top_k]

    def delete(self, record_ids):
        pass

    def clear(self):
        pass


class _MockEmbedding(EmbeddingProvider):
    def __init__(self, raise_exc=False):
        self._raise = raise_exc

    @property
    def dimension(self) -> int:
        return 384

    def embed_documents(self, texts):
        if self._raise:
            raise RuntimeError("embedding 挂了")
        return [[0.0] * 384 for _ in texts]

    def embed_query(self, text):
        if self._raise:
            raise RuntimeError("embedding 挂了")
        return [0.0] * 384


class _MockReranker(RerankerProvider):
    def __init__(self, raise_exc=False):
        self._raise = raise_exc

    def rerank(self, query, documents, top_k=5):
        if self._raise:
            raise RuntimeError("reranker 挂了")
        return [(i, 1.0 - i * 0.1) for i in range(min(top_k, len(documents)))]


class _MockLLM(LLMProvider):
    def __init__(self, response="这是答案[1]。", raise_exc=False, empty=False):
        self._response = "" if empty else response
        self._raise = raise_exc
        self.calls: list[list[dict[str, str]]] = []

    def generate(self, prompt, system_prompt="", temperature=0.7, max_tokens=2048):
        return self._response

    def generate_with_messages(self, messages, temperature=0.7, max_tokens=2048):
        if self._raise:
            raise RuntimeError("LLM 挂了")
        self.calls.append(messages)
        return self._response


class _MockDocumentStore(DocumentStorePort):
    def __init__(self, documents=None, sections=None):
        self._documents = {d.id: d for d in (documents or [])}
        self._sections = sections or []

    def save_document(self, doc):
        self._documents[doc.id] = doc

    def get_document(self, doc_id):
        return self._documents.get(doc_id)

    def get_document_by_hash(self, file_hash):
        for d in self._documents.values():
            if d.file_hash == file_hash:
                return d
        return None

    def list_documents(self):
        return list(self._documents.values())

    def save_sections(self, sections):
        self._sections = sections

    def get_sections(self, doc_id):
        return [s for s in self._sections if s.document_id == doc_id]

    def delete_document(self, doc_id):
        self._documents.pop(doc_id, None)
        self._sections = [s for s in self._sections if s.document_id != doc_id]


class _MockSessionStore(SessionStorePort):
    """内存版 SessionStore，支持模拟异常。"""

    def __init__(self, raise_on_load=False, raise_on_save=False):
        self._sessions: dict[str, list[dict[str, str]]] = {}
        self._summaries: dict[str, str] = {}
        self._raise_on_load = raise_on_load
        self._raise_on_save = raise_on_save

    def save_messages(self, session_id, messages, ttl_seconds=3600, summary=""):
        if self._raise_on_save:
            raise RuntimeError("保存失败")
        self._sessions[session_id] = list(messages)
        self._summaries[session_id] = summary

    def load_messages(self, session_id):
        if self._raise_on_load:
            raise RuntimeError("加载失败")
        return self._sessions.get(session_id, [])

    def load_summary(self, session_id):
        if self._raise_on_load:
            raise RuntimeError("加载失败")
        return self._summaries.get(session_id, "")

    def delete_session(self, session_id):
        self._sessions.pop(session_id, None)
        self._summaries.pop(session_id, None)


class _MockTraceStore(TraceStorePort):
    def __init__(self, raise_on_save=False):
        self._traces: dict[str, Record] = {}
        self._raise_on_save = raise_on_save

    def save(self, trace):
        if self._raise_on_save:
            raise RuntimeError("trace 保存失败")
        self._traces[trace.id] = trace

    def get(self, trace_id):
        return self._traces.get(trace_id)

    def list(self, filters=None):
        return list(self._traces.values())

    def delete(self, trace_id):
        self._traces.pop(trace_id, None)


class _MockMemoryContext(MemoryContextPort):
    def rewrite_query(self, user_id, query):
        return "改写后的问题"

    def build_context(self, user_id, query, max_chars=800):
        return "用户研究船舶水动力学"


class _MockSkillContext(SkillContextPort):
    def match(self, user_id, query, top_k=3):
        return [Record(id="skill-1", kind="skill", content="技能")]

    def build_context(self, user_id, query, max_chars=1200):
        return "技能：艇型对比分析"


class _MockSkillLifecycle(SkillLifecyclePort):
    def __init__(self):
        self.usage: list[dict] = []
        self.feedback: list[dict] = []

    def record_usage(self, skill_id, user_id, success):
        self.usage.append(
            {"skill_id": skill_id, "user_id": user_id, "success": success}
        )

    def record_feedback(self, skill_id, user_id, positive):
        self.feedback.append(
            {"skill_id": skill_id, "user_id": user_id, "positive": positive}
        )


class _MockToolOrchestrator:
    def run(self, question, permissions=None):
        return [
            ToolResult(
                tool_name="knowledge_base_search",
                success=True,
                output={"chunk_id": "chunk-1"},
                duration_ms=1.0,
            )
        ]

    def build_context(self, results, max_chars=2000):
        return "【工具结果】\n- knowledge_base_search: chunk-1"


def _make_chunk(cid="c1", doc_id="d1", sec_id="s1", page=3, content="内容"):
    return Chunk(
        id=cid,
        content=content,
        metadata={"document_id": doc_id, "section_id": sec_id, "page": page},
    )


def _make_document(doc_id="d1", file_name="论文.pdf"):
    return Document(id=doc_id, metadata={"file_name": file_name})


def _make_section(sec_id="s1", doc_id="d1", title="引言"):
    return SectionBlock(
        id=sec_id,
        metadata={"document_id": doc_id, "section_title": title},
    )


def _build_engine(**kwargs):
    """构建 QAEngine，允许覆盖各组件。"""
    chunks = kwargs.pop("chunks", [_make_chunk()])
    vec_results = kwargs.pop("vec_results", [(c, 0.9) for c in chunks])
    ft_results = kwargs.pop("ft_results", [(c, 5.0) for c in chunks])

    vec_store = kwargs.pop("vector_store", _MockVectorStore(results=vec_results))
    ft_store = kwargs.pop("fulltext_store", _MockFullTextStore(results=ft_results))
    embedding = kwargs.pop("embedding", _MockEmbedding())
    reranker_provider = kwargs.pop("reranker_provider", _MockReranker())
    llm = kwargs.pop("llm", _MockLLM())
    doc_store = kwargs.pop(
        "document_store",
        _MockDocumentStore(
            documents=[_make_document()], sections=[_make_section()]
        ),
    )
    session_store = kwargs.pop("session_store", None)
    trace_store = kwargs.pop("trace_store", None)
    memory_context = kwargs.pop("memory_context", None)
    skill_context = kwargs.pop("skill_context", None)
    skill_lifecycle = kwargs.pop("skill_lifecycle", None)
    tool_orchestrator = kwargs.pop("tool_orchestrator", None)

    retriever = Retriever(vec_store, ft_store, embedding)
    reranker = Reranker(reranker_provider)
    citation_builder = CitationBuilder(doc_store)
    answer_generator = AnswerGenerator(llm)

    return QAEngine(
        retriever=retriever,
        reranker=reranker,
        citation_builder=citation_builder,
        answer_generator=answer_generator,
        embedding_provider=embedding,
        session_store=session_store,
        trace_store=trace_store,
        memory_context=memory_context,
        skill_context=skill_context,
        skill_lifecycle=skill_lifecycle,
        tool_orchestrator=tool_orchestrator,
        **kwargs,
    )


# ================================================================
# 正常流程
# ================================================================


def test_qa_engine_happy_path():
    """正常链路：检索→重排→生成，返回答案。"""
    engine = _build_engine()
    answer = engine.answer("BERT 是什么？")

    assert answer.text == "这是答案[1]。"
    assert answer.trace_id  # 有 trace_id
    assert len(answer.citations) == 1


def test_qa_engine_no_results():
    """检索无结果时返回"未找到相关内容"。"""
    engine = _build_engine(chunks=[], vec_results=[], ft_results=[])
    answer = engine.answer("问题")

    assert answer.text == NO_ANSWER_TEXT
    assert answer.citations == []


# ================================================================
# 降级场景
# ================================================================


def test_qa_engine_embedding_failure_degrades_to_fulltext():
    """Embedding 失败时降级为纯全文检索。"""
    ft_chunk = _make_chunk(cid="ft1", content="全文检索命中的内容")
    engine = _build_engine(
        chunks=[],
        vec_results=[],
        ft_results=[(ft_chunk, 5.0)],
        embedding=_MockEmbedding(raise_exc=True),
        vector_store=_MockVectorStore(results=[]),
    )

    answer = engine.answer("问题")

    assert answer.text == "这是答案[1]。"
    assert len(answer.citations) == 1


def test_qa_engine_reranker_failure_skips_rerank():
    """Reranker 失败时跳过重排，直接用检索候选。"""
    chunks = [_make_chunk(cid=f"c{i}") for i in range(3)]
    engine = _build_engine(
        chunks=chunks,
        reranker_provider=_MockReranker(raise_exc=True),
    )

    answer = engine.answer("问题")

    assert answer.text == "这是答案[1]。"
    assert len(answer.citations) >= 1


def test_qa_engine_llm_failure_returns_fallback():
    """LLM 失败时返回兜底文案。"""
    engine = _build_engine(llm=_MockLLM(raise_exc=True))
    answer = engine.answer("问题")

    assert answer.text == SERVICE_UNAVAILABLE_TEXT
    assert answer.citations == []


def test_qa_engine_llm_empty_response():
    """LLM 返回空内容时返回"未找到相关内容"。"""
    engine = _build_engine(llm=_MockLLM(empty=True))
    answer = engine.answer("问题")

    assert answer.text == NO_ANSWER_TEXT


def test_qa_engine_all_search_fails():
    """向量和全文都失败时返回"未找到相关内容"。"""
    engine = _build_engine(
        chunks=[],
        vec_results=[],
        ft_results=[],
        embedding=_MockEmbedding(raise_exc=True),
        vector_store=_MockVectorStore(raise_exc=True),
        fulltext_store=_MockFullTextStore(raise_exc=True),
    )

    answer = engine.answer("问题")

    assert answer.text == NO_ANSWER_TEXT


# ================================================================
# 上下文预算
# ================================================================


def test_qa_engine_context_budget_truncation():
    """总内容超出预算时，只保留前面（相关性高）的 chunk。"""
    big_chunks = [
        _make_chunk(cid=f"c{i}", content="x" * 1000) for i in range(10)
    ]
    engine = _build_engine(chunks=big_chunks, context_token_budget=CONTEXT_TOKEN_BUDGET)

    answer = engine.answer("问题")

    total_chars = sum(len(c.content) for c in [_make_chunk(cid=f"c{i}", content="x" * 1000) for i in range(10)])
    assert total_chars > CONTEXT_TOKEN_BUDGET * 2


def test_qa_engine_trace_id_unique():
    """每次调用生成不同的 trace_id。"""
    engine = _build_engine()
    a1 = engine.answer("q1")
    a2 = engine.answer("q2")
    assert a1.trace_id != a2.trace_id


# ================================================================
# 多轮对话（M4 Step 2）
# ================================================================


def test_no_session_id_does_not_save_history():
    """不传 session_id 时，不保存历史，LLM messages 不含历史。"""
    session_store = _MockSessionStore()
    engine = _build_engine(session_store=session_store)
    engine.answer("问题")

    assert session_store._sessions == {}


def test_session_id_saves_conversation_history():
    """传入 session_id 后，本轮对话被保存到 SessionStore。"""
    session_store = _MockSessionStore()
    engine = _build_engine(session_store=session_store)

    engine.answer("第一个问题", session_id="sess-1")

    saved = session_store.load_messages("sess-1")
    assert len(saved) == 2
    assert saved[0] == {"role": "user", "content": "第一个问题"}
    assert saved[1]["role"] == "assistant"
    assert saved[1]["content"] == "这是答案[1]。"


def test_multi_turn_history_passed_to_llm():
    """第二轮对话时，LLM 收到的 messages 包含第一轮历史。"""
    session_store = _MockSessionStore()
    llm = _MockLLM()
    engine = _build_engine(session_store=session_store, llm=llm)

    engine.answer("第一轮问题", session_id="sess-1")
    engine.answer("第二轮问题", session_id="sess-1")

    second_call_messages = llm.calls[1]
    roles = [m["role"] for m in second_call_messages]
    assert roles == ["system", "user", "assistant", "user"]
    assert second_call_messages[1]["content"] == "第一轮问题"
    assert second_call_messages[2]["content"] == "这是答案[1]。"


def test_sliding_window_trims_old_history():
    """超过 max_history_messages 条时，溢出的早期对话被压缩进摘要，
    messages 只保留窗口内的最近消息。"""
    session_store = _MockSessionStore()
    llm = _MockLLM(response="摘要内容" if False else "这是答案[1]。")
    engine = _build_engine(
        session_store=session_store, llm=llm, max_history_messages=4
    )

    for i in range(5):
        engine.answer(f"第{i}轮", session_id="sess-1")

    saved = session_store.load_messages("sess-1")
    assert len(saved) == 4

    summary = session_store.load_summary("sess-1")
    assert summary != ""

    last_call = llm.calls[-1]
    history_messages = [m for m in last_call[1:-1] if m["role"] != "system"]
    assert len(history_messages) == 4


def test_summary_injected_into_prompt():
    """当会话有摘要时，摘要会作为 system 消息注入 LLM prompt。"""
    session_store = _MockSessionStore()
    llm = _MockLLM()
    engine = _build_engine(
        session_store=session_store, llm=llm, max_history_messages=4
    )

    for i in range(5):
        engine.answer(f"第{i}轮", session_id="sess-1")

    summary = session_store.load_summary("sess-1")
    assert summary != ""

    last_call = llm.calls[-1]
    assert any(
        m["role"] == "system" and "历史对话摘要" in m["content"]
        for m in last_call
    )


def test_summary_compresses_overflow_only():
    """未溢出窗口时，不生成摘要。"""
    session_store = _MockSessionStore()
    llm = _MockLLM()
    engine = _build_engine(
        session_store=session_store, llm=llm, max_history_messages=10
    )

    for i in range(3):
        engine.answer(f"第{i}轮", session_id="sess-1")

    summary = session_store.load_summary("sess-1")
    assert summary == ""


def test_session_load_failure_degrades_gracefully():
    """SessionStore 加载失败时，降级为无历史，问答正常。"""
    session_store = _MockSessionStore(raise_on_load=True)
    engine = _build_engine(session_store=session_store)

    answer = engine.answer("问题", session_id="sess-1")
    assert answer.text == "这是答案[1]。"


def test_session_save_failure_does_not_break_answer():
    """SessionStore 保存失败时，不影响答案返回。"""
    session_store = _MockSessionStore(raise_on_save=True)
    engine = _build_engine(session_store=session_store)

    answer = engine.answer("问题", session_id="sess-1")
    assert answer.text == "这是答案[1]。"


def test_no_answer_still_saves_to_session():
    """检索无结果时，兜底答案也会保存到会话历史。"""
    session_store = _MockSessionStore()
    engine = _build_engine(
        chunks=[], vec_results=[], ft_results=[], session_store=session_store
    )

    engine.answer("问题", session_id="sess-1")

    saved = session_store.load_messages("sess-1")
    assert len(saved) == 2
    assert saved[1]["content"] == NO_ANSWER_TEXT


# ================================================================
# P2-3 轨迹落库
# ================================================================


def test_trace_saved_on_success():
    trace_store = _MockTraceStore()
    engine = _build_engine(trace_store=trace_store)

    answer = engine.answer("问题", session_id="sess-1", user_id="user-1")

    assert answer.text == "这是答案[1]。"
    trace = trace_store.get(answer.trace_id)
    assert trace is not None
    assert trace.kind == "trace"
    assert trace.source == "sess-1"
    assert trace.metadata["status"] == "success"
    assert trace.metadata["user_id"] == "user-1"
    assert trace.metadata["answer"] == "这是答案[1]。"
    assert trace.metadata["citation_count"] == 1
    assert trace.metadata["candidate_count"] == 1


def test_trace_saved_on_no_results():
    trace_store = _MockTraceStore()
    engine = _build_engine(
        chunks=[], vec_results=[], ft_results=[], trace_store=trace_store
    )

    answer = engine.answer("问题", user_id="user-1")

    trace = trace_store.get(answer.trace_id)
    assert trace is not None
    assert trace.metadata["status"] == "no_results"
    assert trace.metadata["candidate_count"] == 0
    assert trace.metadata["citation_count"] == 0


def test_trace_save_failure_does_not_break_answer():
    trace_store = _MockTraceStore(raise_on_save=True)
    engine = _build_engine(trace_store=trace_store)

    answer = engine.answer("问题")

    assert answer.text == "这是答案[1]。"


# ================================================================
# P2-7 画像注入
# ================================================================


def test_memory_context_injected_into_prompt_and_trace():
    llm = _MockLLM()
    trace_store = _MockTraceStore()
    engine = _build_engine(
        llm=llm,
        trace_store=trace_store,
        memory_context=_MockMemoryContext(),
    )

    answer = engine.answer("原问题", user_id="user-a")

    assert answer.text == "这是答案[1]。"
    last_messages = llm.calls[-1]
    assert any(
        message["role"] == "system" and "用户研究船舶水动力学" in message["content"]
        for message in last_messages
    )
    trace = trace_store.get(answer.trace_id)
    assert trace.metadata["retrieval_query"] == "改写后的问题"
    assert trace.metadata["memory_context_used"] is True


# ================================================================
# M5-4 技能注入
# ================================================================


def test_skill_context_injected_into_prompt_and_trace():
    llm = _MockLLM()
    trace_store = _MockTraceStore()
    engine = _build_engine(
        llm=llm,
        trace_store=trace_store,
        skill_context=_MockSkillContext(),
    )

    answer = engine.answer("怎么比较艇型？", user_id="user-a")

    last_messages = llm.calls[-1]
    assert any(
        message["role"] == "system" and "艇型对比分析" in message["content"]
        for message in last_messages
    )
    trace = trace_store.get(answer.trace_id)
    assert trace.metadata["skill_context_used"] is True
    assert trace.metadata["matched_skill_ids"] == ["skill-1"]


def test_skill_usage_recorded_after_answer():
    lifecycle = _MockSkillLifecycle()
    engine = _build_engine(
        skill_context=_MockSkillContext(),
        skill_lifecycle=lifecycle,
    )

    answer = engine.answer("怎么比较艇型？", user_id="user-a")

    assert answer.text == "这是答案[1]。"
    assert lifecycle.usage == [
        {"skill_id": "skill-1", "user_id": "user-a", "success": True}
    ]


# ================================================================
# M6-2 工具注入
# ================================================================


def test_tool_context_injected_into_prompt_and_trace():
    llm = _MockLLM()
    trace_store = _MockTraceStore()
    engine = _build_engine(
        llm=llm,
        trace_store=trace_store,
        tool_orchestrator=_MockToolOrchestrator(),
    )

    answer = engine.answer("请检索论文")

    last_messages = llm.calls[-1]
    assert any(
        message["role"] == "system" and "【工具结果】" in message["content"]
        for message in last_messages
    )
    trace = trace_store.get(answer.trace_id)
    assert trace.metadata["tool_context_used"] is True
    assert trace.metadata["tool_results"][0]["tool_name"] == "knowledge_base_search"


def test_stage_timings_and_metrics_recorded():
    metrics.reset()
    trace_store = _MockTraceStore()
    engine = _build_engine(trace_store=trace_store)

    answer = engine.answer("问题")

    trace = trace_store.get(answer.trace_id)
    assert "retrieve_ms" in trace.metadata["stage_timings"]
    snapshot = metrics.snapshot()
    assert snapshot["counters"]["qa_requests_total"] == 1
    assert "qa_latency_ms" in snapshot["histograms"]
