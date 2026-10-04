"""测试用 MCP server，支持 stdio 和 streamable HTTP。"""

import os

from mcp.server.fastmcp import FastMCP


mcp = FastMCP(
    "KnowResearchTestServer",
    port=int(os.getenv("MCP_PORT", "8765")),
)


@mcp.tool()
def echo(text: str) -> str:
    """返回 echo 前缀。"""
    return f"echo:{text}"


if __name__ == "__main__":
    mcp.run(transport=os.getenv("MCP_TRANSPORT", "stdio"))
