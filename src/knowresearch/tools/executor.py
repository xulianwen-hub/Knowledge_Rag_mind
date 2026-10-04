"""工具执行器：参数校验、权限检查、超时和审计结果。"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from typing import Any

from knowresearch.core.schemas import ToolCall, ToolResult
from knowresearch.tools.registry import ToolRegistry


class ToolExecutor:
    """执行工具调用，并把失败转成 ToolResult，不向主链路抛异常。"""

    def __init__(
        self,
        registry: ToolRegistry,
        default_timeout_seconds: float = 10.0,
    ):
        self.registry = registry
        self.default_timeout_seconds = default_timeout_seconds

    def execute(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
        permissions: set[str] | None = None,
    ) -> ToolResult:
        started_at = time.perf_counter()
        tool = self.registry.get(tool_name)
        if tool is None:
            return self._error(tool_name, "工具不存在", started_at)

        required_permissions = set(tool.definition.permissions)
        if permissions is not None and not required_permissions.issubset(permissions):
            return self._error(tool_name, "权限不足", started_at)

        arguments = arguments or {}
        missing = self._missing_required(tool.definition.input_schema, arguments)
        if missing:
            return self._error(
                tool_name,
                f"缺少必填参数: {', '.join(missing)}",
                started_at,
            )

        timeout = tool.definition.timeout_seconds or self.default_timeout_seconds
        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(tool.run, arguments)
        try:
            output = future.result(timeout=timeout)
            return ToolResult(
                tool_name=tool_name,
                success=True,
                output=output,
                duration_ms=self._elapsed_ms(started_at),
            )
        except TimeoutError:
            future.cancel()
            return self._error(tool_name, "工具执行超时", started_at)
        except Exception as e:
            return self._error(tool_name, str(e), started_at)
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

    def execute_call(
        self,
        call: ToolCall,
        permissions: set[str] | None = None,
    ) -> ToolResult:
        return self.execute(call.tool_name, call.arguments, permissions=permissions)

    @staticmethod
    def _missing_required(
        input_schema: dict,
        arguments: dict,
    ) -> list[str]:
        required = input_schema.get("required", [])
        return [key for key in required if key not in arguments or arguments[key] is None]

    @staticmethod
    def _elapsed_ms(started_at: float) -> float:
        return round((time.perf_counter() - started_at) * 1000, 2)

    def _error(self, tool_name: str, error: str, started_at: float) -> ToolResult:
        return ToolResult(
            tool_name=tool_name,
            success=False,
            error=error,
            duration_ms=self._elapsed_ms(started_at),
        )
