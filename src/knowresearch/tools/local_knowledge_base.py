"""本地知识库检索工具。"""

from __future__ import annotations

from typing import Any

from knowresearch.core.ports import ToolPort
from knowresearch.core.schemas import ToolDefinition


class LocalKnowledgeBaseTool(ToolPort):
    """把现有 Retriever 包装成工具。"""

    def __init__(self, retriever, name: str = "knowledge_base_search"):
        self.retriever = retriever
        self._name = name

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self._name,
            description="检索本地科研知识库，返回相关论文片段。",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer", "default": 5},
                },
                "required": ["query"],
            },
            permissions=["knowledge_base:read"],
            timeout_seconds=15.0,
            source="local",
        )

    def run(self, arguments: dict[str, Any]) -> list[dict[str, Any]]:
        query = str(arguments["query"])
        top_k = int(arguments.get("top_k", 5))
        chunks = self.retriever.retrieve(query)
        return [
            {
                "chunk_id": chunk.id,
                "content": chunk.content,
                "document_id": chunk.document_id,
                "section_id": chunk.section_id,
                "page": chunk.page,
            }
            for chunk in chunks[:top_k]
        ]
