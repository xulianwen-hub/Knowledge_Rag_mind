"""工具编排：规则选择工具、执行并构建上下文。"""

from __future__ import annotations

import json

from knowresearch.core.schemas import ToolCall, ToolResult
from knowresearch.tools.executor import ToolExecutor
from knowresearch.tools.registry import ToolRegistry


class ToolOrchestrator:
    """M6-2 的规则版工具编排器。"""

    def __init__(
        self,
        registry: ToolRegistry,
        executor: ToolExecutor,
        tool_rules: dict[str, list[str]] | None = None,
    ):
        self.registry = registry
        self.executor = executor
        self.tool_rules = tool_rules or {
            "knowledge_base_search": [
                "检索",
                "搜索",
                "查找",
                "论文",
                "文献",
                "资料",
                "知识库",
                "引用",
            ]
        }

    def select(
        self,
        question: str,
        permissions: set[str] | None = None,
    ) -> list[ToolCall]:
        if not question:
            return []
        selected: list[ToolCall] = []
        for tool_name, keywords in self.tool_rules.items():
            tool = self.registry.get(tool_name)
            if tool is None:
                continue
            required = set(tool.definition.permissions)
            if permissions is not None and not required.issubset(permissions):
                continue
            if any(keyword in question for keyword in keywords):
                selected.append(
                    ToolCall(
                        tool_name=tool_name,
                        arguments={"query": question, "top_k": 5},
                    )
                )
        return selected

    def run(
        self,
        question: str,
        permissions: set[str] | None = None,
    ) -> list[ToolResult]:
        calls = self.select(question, permissions=permissions)
        return [
            self.executor.execute_call(call, permissions=permissions)
            for call in calls
        ]

    def build_context(
        self,
        results: list[ToolResult],
        max_chars: int = 2000,
    ) -> str:
        successful = [result for result in results if result.success]
        if not successful:
            return ""

        lines = ["【工具结果】"]
        used = 0
        for result in successful:
            output = result.output
            if not isinstance(output, str):
                output = json.dumps(output, ensure_ascii=False)
            block = f"- {result.tool_name}: {output}"
            if used + len(block) > max_chars:
                break
            lines.append(block)
            used += len(block)
        return "\n".join(lines)
