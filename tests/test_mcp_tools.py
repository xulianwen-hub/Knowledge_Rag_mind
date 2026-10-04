"""M6-3 MCP 客户端、工具适配器和注册器测试。"""

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from knowresearch.tools import (
    MCPToolAdapter,
    StdioMCPClient,
    StreamableHTTPMCPClient,
    ToolExecutor,
    ToolRegistry,
    register_mcp_tools,
)


def test_stdio_mcp_client_list_and_call_tool():
    server_script = Path(__file__).parent / "fake_mcp_server.py"
    client = StdioMCPClient(
        command=sys.executable,
        args=[str(server_script)],
        request_timeout_seconds=20,
    )
    try:
        tools = client.list_tools()
        assert any(tool.name == "echo" for tool in tools)

        result = client.call_tool("echo", {"text": "hello"})
        assert result["is_error"] is False
        assert "echo:hello" in result["content"][0]["text"]
    finally:
        client.close()


def test_register_mcp_tools_into_registry_and_executor():
    server_script = Path(__file__).parent / "fake_mcp_server.py"
    client = StdioMCPClient(
        command=sys.executable,
        args=[str(server_script)],
        request_timeout_seconds=20,
    )
    try:
        registry = ToolRegistry()
        registered = register_mcp_tools(
            registry,
            client,
            permissions=["mcp:read"],
        )
        assert "echo" in registered

        executor = ToolExecutor(registry)
        result = executor.execute(
            "echo",
            {"text": "world"},
            permissions={"mcp:read"},
        )
        assert result.success is True
        assert "echo:world" in result.output["content"][0]["text"]
    finally:
        client.close()


def test_streamable_http_mcp_client():
    server_script = Path(__file__).parent / "fake_mcp_server.py"
    port = _free_port()
    env = os.environ.copy()
    env["MCP_TRANSPORT"] = "streamable-http"
    env["MCP_PORT"] = str(port)
    process = subprocess.Popen(
        [sys.executable, str(server_script)],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    client = None
    try:
        _wait_for_port("127.0.0.1", port, timeout=20)
        deadline = time.time() + 20
        last_error = None
        while time.time() < deadline:
            try:
                client = StreamableHTTPMCPClient(
                    url=f"http://127.0.0.1:{port}/mcp",
                    request_timeout_seconds=20,
                )
                break
            except Exception as e:
                last_error = e
                time.sleep(0.5)
        if client is None:
            pytest.fail(f"MCP HTTP 客户端初始化失败: {last_error}")

        tools = client.list_tools()
        assert any(tool.name == "echo" for tool in tools)
        result = client.call_tool("echo", {"text": "http"})
        assert "echo:http" in result["content"][0]["text"]
    finally:
        if client is not None:
            client.close()
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _wait_for_port(host: str, port: int, timeout: float) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return
        except OSError:
            time.sleep(0.2)
    raise TimeoutError(f"端口未就绪: {host}:{port}")
