"""MCP 工具适配器与注册器。"""

from __future__ import annotations

from typing import Any

from knowresearch.core.ports import MCPClientPort, ToolPort
from knowresearch.core.schemas import ToolDefinition
from knowresearch.tools.registry import ToolRegistry


class MCPToolAdapter(ToolPort):
    """把一个 MCP tool 包装成本地 ToolPort。"""

    def __init__(
        self,
        client: MCPClientPort,
        definition: ToolDefinition,
    ):
        self.client = client
        self._definition = definition

    @property
    def definition(self) -> ToolDefinition:
        return self._definition

    def run(self, arguments: dict[str, Any]) -> Any:
        return self.client.call_tool(self._definition.name, arguments)


def register_mcp_tools(
    registry: ToolRegistry,
    client: MCPClientPort,
    permissions: list[str] | None = None,
) -> list[str]:
    """发现 MCP 工具并注册到 ToolRegistry，返回注册的工具名。"""
    registered: list[str] = []
    for definition in client.list_tools():
        if permissions is not None:
            definition.permissions = list(permissions)
        definition.source = "mcp"
        registry.register(MCPToolAdapter(client, definition))
        registered.append(definition.name)
    return registered
