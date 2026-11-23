"""长期记忆检索工具：回答「我读过什么」这类个人历史问题。"""

from __future__ import annotations

from typing import Any

from knowresearch.core.ports import LongTermMemoryPort, ToolPort
from knowresearch.core.schemas import ToolDefinition


class MemorySearchTool(ToolPort):
    """把长期记忆封装为 Agent 可调用的工具。"""

    def __init__(
        self,
        long_term_memory: LongTermMemoryPort,
        name: str = "memory_search",
    ) -> None:
        self.long_term_memory = long_term_memory
        self._name = name

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self._name,
            description="检索用户的长期记忆（画像、术语、阅读历史），回答个人历史类问题。",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer", "default": 5},
                },
                "required": ["query"],
            },
            permissions=["memory:read"],
            timeout_seconds=10.0,
            source="local",
        )

    def run(self, arguments: dict[str, Any]) -> list[dict[str, Any]]:
        query = str(arguments.get("query", ""))
        top_k = int(arguments.get("top_k", 5))
        records = self.long_term_memory.search(query, top_k=top_k)
        return [
            {
                "memory_id": record.id,
                "kind": record.kind,
                "content": record.content,
                "metadata": record.metadata,
            }
            for record in records
        ]
