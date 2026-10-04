"""PDF 解析器：基于 PyMuPDF 提取文本与章节结构。

策略：
1. 逐页提取文本，记录页码
2. 逐行扫描，用 heading_patterns.match_heading 识别标题
3. 构建 SectionBlock 层级（每个标题及其下属内容为一个 section）
4. 参考文献章节打标 is_reference=True
5. 无标题的开头内容归入"正文"section
"""

import hashlib
import os
from datetime import datetime

import fitz  # PyMuPDF

from knowresearch.core.schemas import Document, SectionBlock
from knowresearch.ingestion.parsers.base import ParserPort
from knowresearch.ingestion.parsers.heading_patterns import match_heading


REFERENCE_KEYWORDS = ["参考文献", "References", "REFERENCES", "Bibliography", "BIBLIOGRAPHY", "引用文献"]


def _compute_file_hash(file_path: str) -> str:
    """计算文件 SHA256。"""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


class PDFParser(ParserPort):
    """PDF 解析器。"""

    def parse(self, file_path: str, doc_id: str = "") -> tuple[Document, list[SectionBlock]]:
        file_hash = _compute_file_hash(file_path)
        file_name = os.path.basename(file_path)

        doc = fitz.open(file_path)
        total_pages = doc.page_count

        raw_sections: list[dict] = []
        current: dict | None = None

        for page_idx in range(total_pages):
            page = doc[page_idx]
            page_num = page_idx + 1
            text = page.get_text("text")
            for line in text.splitlines():
                heading = match_heading(line)
                if heading is not None:
                    level, title = heading
                    if current is not None:
                        raw_sections.append(current)
                    current = {
                        "level": level,
                        "title": title,
                        "content_lines": [],
                        "page_start": page_num,
                        "page_end": page_num,
                        "is_reference": any(kw in title for kw in REFERENCE_KEYWORDS),
                    }
                else:
                    if current is None:
                        current = {
                            "level": 1,
                            "title": "正文",
                            "content_lines": [],
                            "page_start": page_num,
                            "page_end": page_num,
                            "is_reference": False,
                        }
                    current["content_lines"].append(line)
                    current["page_end"] = page_num

        if current is not None:
            raw_sections.append(current)

        doc.close()

        doc_kwargs = dict(
            content="",
            source=file_path,
            metadata={
                "file_name": file_name,
                "file_hash": file_hash,
                "file_type": "pdf",
                "total_pages": total_pages,
                "file_size": os.path.getsize(file_path),
            },
        )
        if doc_id:
            doc_kwargs["id"] = doc_id
        document = Document(**doc_kwargs)

        sections: list[SectionBlock] = []
        for idx, sec in enumerate(raw_sections):
            content = "\n".join(sec["content_lines"]).strip()
            section = SectionBlock(
                content=content,
                source=file_path,
                metadata={
                    "document_id": document.id,
                    "section_title": sec["title"],
                    "section_level": sec["level"],
                    "page_start": sec["page_start"],
                    "page_end": sec["page_end"],
                    "is_reference": sec["is_reference"],
                    "order_index": idx,
                },
            )
            sections.append(section)

        return document, sections