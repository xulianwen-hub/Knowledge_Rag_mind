"""MCP stdio 客户端：在后台事件循环里维持 MCP session。"""

from __future__ import annotations

import asyncio
import threading
from datetime import timedelta
from typing import Any

from knowresearch.core.ports import MCPClientPort
from knowresearch.core.schemas import ToolDefinition


class StdioMCPClient(MCPClientPort):
    """同步 facade，内部用后台 asyncio 事件循环维持 MCP stdio session。"""

    def __init__(
        self,
        command: str,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
        request_timeout_seconds: float = 30.0,
    ):
        self.command = command
        self.args = args or []
        self.env = env
        self.cwd = cwd
        self.request_timeout_seconds = request_timeout_seconds

        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._thread_main,
            name="mcp-stdio-client",
            daemon=True,
        )
        self._ready = threading.Event()
        self._error: Exception | None = None
        self._session = None
        self._stop_event: asyncio.Event | None = None

        self._thread.start()
        if not self._ready.wait(timeout=request_timeout_seconds):
            raise TimeoutError("MCP 初始化超时")
        if self._error is not None:
            raise self._error

    def _thread_main(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.create_task(self._session_main())
        self._loop.run_forever()

    async def _session_main(self) -> None:
        from mcp import ClientSession
        from mcp.client.stdio import StdioServerParameters, stdio_client

        try:
            params = StdioServerParameters(
                command=self.command,
                args=self.args,
                env=self.env,
                cwd=self.cwd,
            )
            async with stdio_client(params) as (read_stream, write_stream):
                async with ClientSession(
                    read_stream,
                    write_stream,
                    read_timeout_seconds=timedelta(
                        seconds=self.request_timeout_seconds
                    ),
                ) as session:
                    await session.initialize()
                    self._session = session
                    self._stop_event = asyncio.Event()
                    self._ready.set()
                    await self._stop_event.wait()
        except Exception as e:
            self._error = e
            self._ready.set()
        finally:
            self._session = None
            self._loop.call_soon_threadsafe(self._loop.stop)

    def list_tools(self) -> list[ToolDefinition]:
        session = self._require_session()
        result = self._run_coroutine(session.list_tools())
        definitions = []
        for tool in result.tools:
            definitions.append(
                ToolDefinition(
                    name=tool.name,
                    description=tool.description or "",
                    input_schema=tool.inputSchema or {},
                    permissions=[],
                    timeout_seconds=self.request_timeout_seconds,
                    source="mcp",
                )
            )
        return definitions

    def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        session = self._require_session()
        result = self._run_coroutine(session.call_tool(tool_name, arguments))
        return {
            "is_error": bool(getattr(result, "isError", False)),
            "content": [
                self._serialize_content(item)
                for item in getattr(result, "content", [])
            ],
        }

    def close(self) -> None:
        if self._stop_event is not None:
            self._loop.call_soon_threadsafe(self._stop_event.set)
        if self._thread.is_alive():
            self._thread.join(timeout=5)

    def _require_session(self):
        if self._session is None:
            raise RuntimeError("MCP session 不可用")
        return self._session

    def _run_coroutine(self, coroutine):
        future = asyncio.run_coroutine_threadsafe(coroutine, self._loop)
        return future.result(timeout=self.request_timeout_seconds)

    @staticmethod
    def _serialize_content(item: Any) -> dict[str, Any]:
        if hasattr(item, "text"):
            return {"type": getattr(item, "type", "text"), "text": item.text}
        if hasattr(item, "data"):
            return {"type": getattr(item, "type", "data"), "data": item.data}
        return {"type": "unknown", "value": str(item)}


class StreamableHTTPMCPClient(StdioMCPClient):
    """通过 streamable HTTP 连接 MCP server。"""

    def __init__(
        self,
        url: str,
        headers: dict[str, str] | None = None,
        request_timeout_seconds: float = 30.0,
    ):
        self.url = url
        self.headers = headers
        super().__init__(
            command="",
            request_timeout_seconds=request_timeout_seconds,
        )

    async def _session_main(self) -> None:
        from mcp import ClientSession
        from mcp.client.streamable_http import streamablehttp_client

        try:
            async with streamablehttp_client(
                self.url,
                headers=self.headers,
                timeout=self.request_timeout_seconds,
            ) as (read_stream, write_stream, _):
                async with ClientSession(
                    read_stream,
                    write_stream,
                    read_timeout_seconds=timedelta(
                        seconds=self.request_timeout_seconds
                    ),
                ) as session:
                    await session.initialize()
                    self._session = session
                    self._stop_event = asyncio.Event()
                    self._ready.set()
                    await self._stop_event.wait()
        except Exception as e:
            self._error = e
            self._ready.set()
        finally:
            self._session = None
            self._loop.call_soon_threadsafe(self._loop.stop)
