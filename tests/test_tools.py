"""M6-1 / M6-2 工具层单元测试。"""

import time

from knowresearch.core.ports import ToolPort
from knowresearch.core.schemas import Chunk, ToolDefinition
from knowresearch.tools import (
    LocalKnowledgeBaseTool,
    ToolExecutor,
    ToolOrchestrator,
    ToolRegistry,
)


class _EchoTool(ToolPort):
    @property
    def definition(self):
        return ToolDefinition(
            name="echo",
            description="返回输入",
            input_schema={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
            permissions=["echo:read"],
            timeout_seconds=1.0,
        )

    def run(self, arguments):
        return {"text": arguments["text"]}


class _SlowTool(ToolPort):
    @property
    def definition(self):
        return ToolDefinition(
            name="slow",
            description="慢工具",
            input_schema={"type": "object", "required": []},
            timeout_seconds=0.05,
        )

    def run(self, arguments):
        time.sleep(0.2)
        return "done"


class _FailTool(ToolPort):
    @property
    def definition(self):
        return ToolDefinition(name="fail", input_schema={"type": "object"})

    def run(self, arguments):
        raise RuntimeError("工具内部错误")


class _FakeRetriever:
    def retrieve(self, query):
        return [
            Chunk(
                id="chunk-1",
                content="测试内容",
                metadata={"document_id": "doc-1", "section_id": "sec-1", "page": 2},
            )
        ]


def _registry_with_echo():
    registry = ToolRegistry()
    registry.register(_EchoTool())
    return registry


def test_registry_register_get_list_and_permissions():
    registry = _registry_with_echo()
    assert registry.get("echo") is not None
    assert registry.names() == ["echo"]
    assert len(registry.list_tools()) == 1
    assert len(registry.list_tools({"echo:read"})) == 1
    assert registry.list_tools({"other"}) == []


def test_executor_success_and_validation():
    executor = ToolExecutor(_registry_with_echo())

    result = executor.execute("echo", {"text": "hello"})
    assert result.success is True
    assert result.output == {"text": "hello"}

    missing = executor.execute("echo", {})
    assert missing.success is False
    assert "缺少必填参数" in missing.error


def test_executor_unknown_permission_and_exception():
    registry = _registry_with_echo()
    registry.register(_FailTool())
    executor = ToolExecutor(registry)

    assert executor.execute("missing").success is False
    assert executor.execute("echo", {"text": "x"}, permissions=set()).success is False
    failed = executor.execute("fail", {})
    assert failed.success is False
    assert "工具内部错误" in failed.error


def test_executor_timeout():
    registry = ToolRegistry()
    registry.register(_SlowTool())
    executor = ToolExecutor(registry)

    result = executor.execute("slow", {})
    assert result.success is False
    assert "超时" in result.error


def test_local_knowledge_base_tool():
    tool = LocalKnowledgeBaseTool(_FakeRetriever())
    result = tool.run({"query": "测试", "top_k": 1})

    assert result[0]["chunk_id"] == "chunk-1"
    assert result[0]["page"] == 2


def test_tool_orchestrator_select_run_and_context():
    registry = ToolRegistry()
    registry.register(LocalKnowledgeBaseTool(_FakeRetriever()))
    executor = ToolExecutor(registry)
    orchestrator = ToolOrchestrator(registry, executor)

    calls = orchestrator.select("请检索这篇论文的方法")
    assert len(calls) == 1
    assert calls[0].tool_name == "knowledge_base_search"

    results = orchestrator.run("请检索这篇论文的方法")
    context = orchestrator.build_context(results)
    assert "工具结果" in context
    assert "chunk-1" in context


def test_tool_orchestrator_returns_empty_without_trigger():
    registry = ToolRegistry()
    registry.register(LocalKnowledgeBaseTool(_FakeRetriever()))
    orchestrator = ToolOrchestrator(registry, ToolExecutor(registry))

    assert orchestrator.select("你好") == []
    assert orchestrator.run("你好") == []
