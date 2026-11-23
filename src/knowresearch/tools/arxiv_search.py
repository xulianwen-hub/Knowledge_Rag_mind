"""arXiv 搜索工具：让 Agent 在本地知识库不足时转向外部最新文献。

网络失败时抛出异常，由 ToolExecutor 统一转成 success=False 的 ToolResult，
不影响主循环；Agent 会根据 Observation 决定下一步。
"""

from __future__ import annotations

from typing import Any
from urllib.parse import quote
from urllib.request import urlopen
import xml.etree.ElementTree as ET

from knowresearch.core.ports import ToolPort
from knowresearch.core.schemas import ToolDefinition


class ArxivSearchTool(ToolPort):
    """调用 arXiv Atom API 搜索论文元数据。"""

    def __init__(
        self,
        base_url: str = "https://export.arxiv.org/api/query",
        timeout_seconds: float = 15.0,
        name: str = "arxiv_search",
    ) -> None:
        self.base_url = base_url
        self.timeout_seconds = timeout_seconds
        self._name = name

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self._name,
            description="搜索 arXiv 上的最新论文，返回标题、作者、摘要与链接。",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "max_results": {"type": "integer", "default": 5},
                },
                "required": ["query"],
            },
            permissions=["web:arxiv:read"],
            timeout_seconds=self.timeout_seconds,
            source="external",
        )

    def run(self, arguments: dict[str, Any]) -> list[dict[str, Any]]:
        query = str(arguments["query"])
        max_results = int(arguments.get("max_results", 5))
        max_results = max(1, min(max_results, 20))

        url = (
            f"{self.base_url}?search_query=all:{quote(query)}"
            f"&start=0&max_results={max_results}"
        )
        with urlopen(url, timeout=self.timeout_seconds) as response:
            payload = response.read()

        root = ET.fromstring(payload)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        results: list[dict[str, Any]] = []
        for entry in root.findall("atom:entry", ns):
            title = _text(entry, "atom:title", ns)
            summary = _text(entry, "atom:summary", ns)
            link = entry.findtext("atom:id", default="", namespaces=ns)
            published = entry.findtext("atom:published", default="", namespaces=ns)
            authors = [
                author.findtext("atom:name", default="", namespaces=ns)
                for author in entry.findall("atom:author", ns)
            ]
            results.append(
                {
                    "title": title,
                    "authors": authors,
                    "summary": summary,
                    "link": link,
                    "published": published,
                }
            )
        return results


def _text(element: ET.Element, tag: str, namespaces: dict) -> str:
    node = element.find(tag, namespaces)
    return "".join(node.itertext()).strip() if node is not None else ""
