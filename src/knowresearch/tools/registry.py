"""工具注册表。"""

from __future__ import annotations

from knowresearch.core.ports import ToolPort


class ToolRegistry:
    """注册和查找工具。"""

    def __init__(self):
        self._tools: dict[str, ToolPort] = {}

    def register(self, tool: ToolPort) -> None:
        self._tools[tool.definition.name] = tool

    def unregister(self, tool_name: str) -> None:
        self._tools.pop(tool_name, None)

    def get(self, tool_name: str) -> ToolPort | None:
        return self._tools.get(tool_name)

    def list_tools(self, permissions: set[str] | None = None) -> list[ToolPort]:
        if permissions is None:
            return list(self._tools.values())
        return [
            tool
            for tool in self._tools.values()
            if set(tool.definition.permissions).issubset(permissions)
        ]

    def names(self) -> list[str]:
        return sorted(self._tools)
