"""BibTeX 生成工具：辅助产出引用条目。"""

from __future__ import annotations

import re
from typing import Any

from knowresearch.core.ports import DocumentStorePort, ToolPort
from knowresearch.core.schemas import ToolDefinition


class BibTeXTool(ToolPort):
    """根据题录信息生成 BibTeX 条目。"""

    def __init__(
        self,
        document_store: DocumentStorePort | None = None,
        name: str = "bibtex_generator",
    ) -> None:
        self.document_store = document_store
        self._name = name

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self._name,
            description="根据标题/作者/年份/venue 生成 BibTeX 引用条目。",
            input_schema={
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "authors": {"type": "string", "default": ""},
                    "year": {"type": "string", "default": ""},
                    "venue": {"type": "string", "default": ""},
                },
                "required": ["title"],
            },
            permissions=["assist:bibtex"],
            timeout_seconds=5.0,
            source="local",
        )

    def run(self, arguments: dict[str, Any]) -> str:
        title = str(arguments.get("title", "")).strip()
        if not title:
            raise ValueError("缺少 title 参数")

        authors = _split_authors(str(arguments.get("authors", "")))
        year = str(arguments.get("year", "")).strip()
        venue = str(arguments.get("venue", "")).strip()
        key = _citation_key(authors, year, title)

        fields = [f"  title = {{{title}}}"]
        if authors:
            fields.append(f"  author = {{{' and '.join(authors)}}}")
        if year:
            fields.append(f"  year = {{{year}}}")
        if venue:
            fields.append(f"  booktitle = {{{venue}}}")
        fields.append("  source = {knowresearch}")

        entry_type = "inproceedings" if venue else "misc"
        return f"@{entry_type}{{{key},\n" + ",\n".join(fields) + "\n}"


def _split_authors(raw: str) -> list[str]:
    raw = raw.strip()
    if not raw:
        return []
    return [part.strip() for part in re.split(r"[,;]| and ", raw) if part.strip()]


def _citation_key(authors: list[str], year: str, title: str) -> str:
    first = authors[0].split()[-1] if authors else "unknown"
    head = re.sub(r"[^A-Za-z0-9]+", "", title)[:20] or "paper"
    return f"{first}{year}{head}"
