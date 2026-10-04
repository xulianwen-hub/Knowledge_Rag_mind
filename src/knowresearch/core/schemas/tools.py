"""工具层 schema：工具定义、调用和结果。"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ToolDefinition(BaseModel):
    """工具定义。"""

    name: str
    description: str = ""
    input_schema: dict[str, Any] = Field(default_factory=dict)
    permissions: list[str] = Field(default_factory=list)
    timeout_seconds: float = 10.0
    source: str = "local"


class ToolCall(BaseModel):
    """一次工具调用请求。"""

    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    """一次工具调用结果。"""

    tool_name: str
    success: bool
    output: Any = None
    error: str = ""
    duration_ms: float = 0.0
