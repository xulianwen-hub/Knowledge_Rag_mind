"""工具端口：可插拔工具的统一抽象。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from knowresearch.core.schemas import ToolDefinition


class ToolPort(ABC):
    """单个工具的抽象接口。"""

    @property
    @abstractmethod
    def definition(self) -> ToolDefinition:
        """工具元数据。"""

    @abstractmethod
    def run(self, arguments: dict[str, Any]) -> Any:
        """执行工具。"""


class MCPClientPort(ABC):
    """MCP 客户端端口：发现并调用外部 MCP 工具。"""

    @abstractmethod
    def list_tools(self) -> list[ToolDefinition]:
        """发现 MCP server 暴露的工具。"""

    @abstractmethod
    def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """调用 MCP 工具。"""

    @abstractmethod
    def close(self) -> None:
        """关闭 MCP 连接。"""
