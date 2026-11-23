"""ReAct Agent 测试：解析、循环护栏、路由、引擎降级。

全部使用脚本化假 LLM / 假工具，不依赖真实模型与网络，运行是毫秒级的。
"""

from __future__ import annotations

from knowresearch.agent.engine import AgentEngine
from knowresearch.agent.parser import parse_decision
from knowresearch.agent.react_agent import AgentRunResult, ReActAgent
from knowresearch.agent.router import ComplexityRouter
from knowresearch.core.generation.answer_generator import Answer, AnswerGenerator
from knowresearch.core.ports import (
    DocumentStorePort,
    LLMProvider,
    ToolPort,
    WorkingMemoryPort,
)
from knowresearch.core.retrieval.citation import CitationBuilder
from knowresearch.core.schemas import Chunk, Document, ToolDefinition
from knowresearch.tools.executor import ToolExecutor
from knowresearch.tools.registry import ToolRegistry


# ====================================================================
# 测试替身
# ====================================================================


class _ScriptedLLM(LLMProvider):
    """按脚本依次返回响应，记录调用。"""

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.calls: list[list[dict[str, str]]] = []

    def generate(self, prompt, system_prompt="", temperature=0.7, max_tokens=2048):
        return self.generate_with_messages(
            [{"role": "user", "content": prompt}], temperature, max_tokens
        )

    def generate_with_messages(self, messages, temperature=0.7, max_tokens=2048):
        self.calls.append(messages)
        if not self._responses:
            return ""
        return self._responses.pop(0)


class _FakeTool(ToolPort):
    def __init__(self, name="knowledge_base_search", chunks=None):
        self._name = name
        self._chunks = chunks or []

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self._name,
            description="检索本地知识库",
            input_schema={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
            permissions=["knowledge_base:read"],
        )

    def run(self, arguments):
        return self._chunks


class _FakeWorkingMemory(WorkingMemoryPort):
    def __init__(self):
        self.store: dict[str, dict] = {}

    def save_state(self, task_id, state):
        self.store[task_id] = dict(state)

    def load_state(self, task_id):
        return self.store.get(task_id)

    def update_state(self, task_id, **kwargs):
        current = self.store.get(task_id, {})
        current.update(kwargs)
        self.store[task_id] = current

    def delete_task(self, task_id):
        self.store.pop(task_id, None)


class _FakeDocumentStore(DocumentStorePort):
    def __init__(self, documents=None):
        self._documents = {d.id: d for d in (documents or [])}

    def save_document(self, doc):
        self._documents[doc.id] = doc

    def get_document(self, doc_id):
        return self._documents.get(doc_id)

    def get_document_by_hash(self, file_hash):
        return None

    def list_documents(self):
        return list(self._documents.values())

    def save_sections(self, sections):
        pass

    def get_sections(self, doc_id):
        return []

    def delete_document(self, doc_id):
        self._documents.pop(doc_id, None)


class _FakeSessionStore:
    def __init__(self):
        self.messages: list[dict[str, str]] = []
        self.summary = ""

    def load_messages(self, session_id):
        return list(self.messages)

    def load_summary(self, session_id):
        return self.summary

    def save_messages(self, session_id, messages, ttl_seconds=3600, summary=""):
        self.messages = list(messages)
        self.summary = summary


class _FakeQAEngine:
    """AgentEngine 测试用的最小 QAEngine 替身。"""

    def __init__(self, answer_text="快路径答案"):
        self._answer_text = answer_text
        self.answer_calls = 0
        self.answer_generator = AnswerGenerator(
            _ScriptedLLM(["ReAct 答案[1]"])
        )
        doc = Document(
            id="doc-1",
            content="",
            metadata={"file_name": "论文.pdf"},
        )
        self.citation_builder = CitationBuilder(_FakeDocumentStore([doc]))
        self.session_store = _FakeSessionStore()
        self.trace_store = None
        self.max_history_messages = 20
        self.ttl_seconds = 3600
        self.memory_context = None

    def answer(self, question, session_id=None, user_id=None):
        self.answer_calls += 1
        return Answer(text=self._answer_text, citations=[], trace_id="qa-trace")


def _make_registry(chunks):
    registry = ToolRegistry()
    registry.register(_FakeTool(chunks=chunks))
    return registry


def _chunk_dict(chunk_id="c1", content="PINN 相关内容"):
    return {
        "chunk_id": chunk_id,
        "content": content,
        "document_id": "doc-1",
        "section_id": "",
        "page": 1,
    }


# ====================================================================
# 解析
# ====================================================================


def test_parse_decision_valid_action():
    decision = parse_decision(
        '{"thought": "先查", "action": "knowledge_base_search", '
        '"action_input": {"query": "PINN"}}'
    )
    assert decision is not None
    assert decision.action == "knowledge_base_search"
    assert decision.action_input == {"query": "PINN"}
    assert decision.is_final is False


def test_parse_decision_fenced_and_final():
    raw = '```json\n{"thought": "够了", "final_answer": "答案"}\n```'
    decision = parse_decision(raw)
    assert decision is not None
    assert decision.is_final is True
    assert decision.final_answer == "答案"


def test_parse_decision_alt_keys():
    decision = parse_decision(
        '{"reasoning": "查一下", "tool": "knowledge_base_search", '
        '"arguments": {"query": "x"}}'
    )
    assert decision is not None
    assert decision.action == "knowledge_base_search"
    assert decision.action_input == {"query": "x"}


def test_parse_decision_invalid_returns_none():
    assert parse_decision("这不是 JSON") is None
    assert parse_decision("") is None
    assert parse_decision('{"foo": "bar"}') is None


# ====================================================================
# ReAct 循环
# ====================================================================


def test_react_loop_tool_then_final():
    llm = _ScriptedLLM(
        [
            '{"thought": "先检索", "action": "knowledge_base_search", '
            '"action_input": {"query": "PINN"}}',
            '{"thought": "信息够了", "final_answer": "PINN 是一种方法"}',
        ]
    )
    from knowresearch.agent.planner import JSONReActPlanner

    memory = _FakeWorkingMemory()
    agent = ReActAgent(
        planner=JSONReActPlanner(llm),
        registry=_make_registry([_chunk_dict()]),
        executor=ToolExecutor(_make_registry([_chunk_dict()])),
        working_memory=memory,
        max_steps=5,
    )

    result = agent.run("PINN 是什么", task_id="task-1")

    assert result.final_answer == "PINN 是一种方法"
    assert result.stop_reason == "final_answer"
    assert len(result.steps) == 1
    assert result.steps[0].action == "knowledge_base_search"
    assert len(result.evidence) == 1
    assert result.llm_calls == 2
    # 状态落库（可恢复、可审计）
    assert memory.load_state("task-1")["goal"] == "PINN 是什么"
    assert len(memory.load_state("task-1")["steps"]) == 1


def test_react_loop_stops_on_duplicate_action():
    llm = _ScriptedLLM(
        [
            '{"action": "knowledge_base_search", "action_input": {"query": "PINN"}}',
            '{"action": "knowledge_base_search", "action_input": {"query": "PINN"}}',
        ]
    )
    from knowresearch.agent.planner import JSONReActPlanner

    agent = ReActAgent(
        planner=JSONReActPlanner(llm),
        registry=_make_registry([_chunk_dict()]),
        executor=ToolExecutor(_make_registry([_chunk_dict()])),
    )
    result = agent.run("PINN 是什么")

    assert result.stop_reason == "duplicate_action"
    assert len(result.steps) == 1


def test_react_loop_respects_max_steps():
    llm = _ScriptedLLM(
        [
            '{"action": "knowledge_base_search", "action_input": {"query": "a"}}',
            '{"action": "knowledge_base_search", "action_input": {"query": "b"}}',
            '{"action": "knowledge_base_search", "action_input": {"query": "c"}}',
        ]
    )
    from knowresearch.agent.planner import JSONReActPlanner

    agent = ReActAgent(
        planner=JSONReActPlanner(llm),
        registry=_make_registry([_chunk_dict()]),
        executor=ToolExecutor(_make_registry([_chunk_dict()])),
        max_steps=2,
    )
    result = agent.run("复杂问题")

    assert result.stop_reason == "max_steps"
    assert len(result.steps) == 2


def test_react_loop_parse_error_stops_gracefully():
    llm = _ScriptedLLM(["模型吐了非 JSON 内容"])
    from knowresearch.agent.planner import JSONReActPlanner

    agent = ReActAgent(
        planner=JSONReActPlanner(llm),
        registry=_make_registry([_chunk_dict()]),
        executor=ToolExecutor(_make_registry([_chunk_dict()])),
    )
    result = agent.run("复杂问题")

    assert result.stop_reason == "parse_error"
    assert result.steps == []
    assert result.status == "failed"


def test_react_observation_is_truncated():
    long_content = "很长的内容" * 200
    llm = _ScriptedLLM(
        [
            '{"action": "knowledge_base_search", "action_input": {"query": "x"}}',
            '{"final_answer": "done"}',
        ]
    )
    from knowresearch.agent.planner import JSONReActPlanner

    agent = ReActAgent(
        planner=JSONReActPlanner(llm),
        registry=_make_registry([_chunk_dict(content=long_content)]),
        executor=ToolExecutor(_make_registry([_chunk_dict(content=long_content)])),
        observation_max_chars=80,
    )
    result = agent.run("问题")

    observation = result.steps[0].observation
    assert len(observation) <= 80 + 3
    assert observation.endswith("...")


def test_react_loop_marks_failed_when_no_evidence_and_no_answer():
    llm = _ScriptedLLM(
        [
            '{"action": "knowledge_base_search", "action_input": {"query": "x"}}',
        ]
    )
    from knowresearch.agent.planner import JSONReActPlanner

    agent = ReActAgent(
        planner=JSONReActPlanner(llm),
        registry=_make_registry([]),
        executor=ToolExecutor(_make_registry([])),
        max_steps=1,
    )
    result = agent.run("问题")
    assert result.status == "failed"


# ====================================================================
# 路由
# ====================================================================


def test_router_simple_question():
    router = ComplexityRouter()
    assert router.is_complex("PINN 是什么？") is False


def test_router_comparison_is_complex():
    router = ComplexityRouter()
    assert router.is_complex("比较 A 论文和 B 论文的方法差异") is True


def test_router_long_question_is_complex():
    router = ComplexityRouter()
    assert router.is_complex("这个" * 25) is True


def test_router_multi_question_is_complex():
    router = ComplexityRouter()
    assert router.is_complex("A 是什么？B 又是什么？") is True


# ====================================================================
# 引擎：路由 + 降级
# ====================================================================


class _FakeReactAgent:
    def __init__(self, result=None, raise_exc=False):
        self._result = result
        self._raise = raise_exc
        self.calls = 0

    def run(self, goal, task_id=None):
        self.calls += 1
        if self._raise:
            raise RuntimeError("ReAct 内部异常")
        return self._result or AgentRunResult(task_id=task_id or "t", goal=goal)


def test_agent_engine_disabled_uses_fast_path():
    qa = _FakeQAEngine()
    engine = AgentEngine(
        qa_engine=qa,
        react_agent=_FakeReactAgent(),
        router=ComplexityRouter(),
        enabled=False,
    )
    answer = engine.answer("比较 A 和 B")
    assert answer.text == "快路径答案"
    assert qa.answer_calls == 1


def test_agent_engine_simple_question_uses_fast_path():
    qa = _FakeQAEngine()
    react = _FakeReactAgent()
    engine = AgentEngine(
        qa_engine=qa,
        react_agent=react,
        router=ComplexityRouter(),
        enabled=True,
        router_enabled=True,
    )
    answer = engine.answer("PINN 是什么？")
    assert answer.text == "快路径答案"
    assert react.calls == 0


def test_agent_engine_complex_question_uses_react():
    qa = _FakeQAEngine()
    react = _FakeReactAgent(
        result=AgentRunResult(
            task_id="t1",
            goal="比较 A 和 B",
            evidence=[_chunk_dict()],
            stop_reason="final_answer",
        )
    )
    engine = AgentEngine(
        qa_engine=qa,
        react_agent=react,
        router=ComplexityRouter(),
        enabled=True,
        router_enabled=True,
    )
    answer = engine.answer("比较 A 论文和 B 论文的方法差异")

    assert react.calls == 1
    assert answer.text == "ReAct 答案[1]"
    assert answer.citations and answer.citations[0].document_title == "论文.pdf"


def test_agent_engine_falls_back_when_react_fails():
    qa = _FakeQAEngine()
    react = _FakeReactAgent(raise_exc=True)
    engine = AgentEngine(
        qa_engine=qa,
        react_agent=react,
        router=ComplexityRouter(),
        enabled=True,
        router_enabled=True,
    )
    answer = engine.answer("比较 A 论文和 B 论文的方法差异")
    assert answer.text == "快路径答案"
    assert qa.answer_calls == 1


# ====================================================================
# bootstrap 装配（轻量栈，不依赖 Postgres / Redis / 真实模型）
# ====================================================================


def test_build_agent_engine_assembles_and_degrades_gracefully():
    from knowresearch.adapters.providers.demo_llm import DemoLLM
    from knowresearch.adapters.providers.embedding import MockEmbedding
    from knowresearch.adapters.stores import BM25FullTextStore, InMemoryVectorStore
    from knowresearch.bootstrap import build_agent_engine
    from knowresearch.core.retrieval.retriever import Retriever

    class _StubQA:
        def __init__(self):
            self.retriever = Retriever(
                InMemoryVectorStore(dimension=384),
                BM25FullTextStore(),
                MockEmbedding(),
            )
            base = _FakeQAEngine()
            self.answer_generator = base.answer_generator
            self.citation_builder = base.citation_builder
            self.session_store = None
            self.trace_store = None
            self.memory_context = None
            self.max_history_messages = 20
            self.ttl_seconds = 3600

        def answer(self, question, session_id=None, user_id=None):
            return Answer(text="快路径答案", citations=[], trace_id="qa-trace")

    engine = build_agent_engine(
        qa_engine=_StubQA(),
        working_memory=_FakeWorkingMemory(),
        llm_provider=DemoLLM(),
    )

    assert isinstance(engine, AgentEngine)
    # DemoLLM 不会输出合法 ReAct JSON，复杂问题应解析失败并降级回快路径
    answer = engine.answer("比较 A 论文和 B 论文的方法差异")
    assert answer.text == "快路径答案"
