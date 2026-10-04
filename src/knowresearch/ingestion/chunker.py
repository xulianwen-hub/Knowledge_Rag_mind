"""文档切块器：将 SectionBlock 切分为检索用的 Chunk。

切块策略：
1. 按段落切分
2. 段落过长按句子切分（中文按 。？！，英文按 .?!）
3. 相邻 chunk 重叠 overlap 字，保留上下文连续性
4. 每个 chunk 记录精确的定位元数据（页码、字符偏移）
"""

import re
from typing import Any

from knowresearch.core.schemas import Chunk, SectionBlock


def _estimate_tokens(text: str) -> int:
    """粗略估算 token 数：中文字符数 + 英文单词数。"""
    chinese = len(re.findall(r"[\u4e00-\u9fff]", text))
    english = len(re.findall(r"[a-zA-Z]+", text))
    return chinese + english


def _split_sentences(text: str) -> list[str]:
    """按句子切分文本，保留分隔符。"""
    parts = re.split(r"(?<=[。？！.?!])\s*", text)
    return [p for p in parts if p.strip()]


def _split_paragraphs(text: str) -> list[str]:
    """按段落切分文本。"""
    parts = re.split(r"\n\s*\n", text)
    return [p.strip() for p in parts if p.strip()]


class Chunker:
    """文档切块器。"""

    def __init__(self, target_size: int = 500, overlap: int = 100) -> None:
        """
        Args:
            target_size: 每个 chunk 的目标字符数（中文约等于 token 数）
            overlap: 相邻 chunk 的重叠字符数
        """
        self.target_size = target_size
        self.overlap = overlap

    def chunk_section(self, section: SectionBlock) -> list[Chunk]:
        """将一个 SectionBlock 切分为多个 Chunk。"""
        text = section.content
        if not text.strip():
            return []

        paragraphs = _split_paragraphs(text)
        if not paragraphs:
            paragraphs = [text]

        chunks_text: list[str] = []
        current = ""

        for para in paragraphs:
            if len(current) + len(para) <= self.target_size:
                current = (current + "\n" + para).strip() if current else para
            else:
                if current:
                    chunks_text.append(current)
                    tail = current[-self.overlap:] if self.overlap > 0 else ""
                    current = tail + para
                else:
                    sentences = _split_sentences(para)
                    buf = ""
                    for sent in sentences:
                        if len(buf) + len(sent) <= self.target_size:
                            buf += sent
                        else:
                            if buf:
                                chunks_text.append(buf)
                                tail = buf[-self.overlap:] if self.overlap > 0 else ""
                                buf = tail + sent
                            else:
                                chunks_text.append(sent)
                                buf = ""
                    current = buf

        if current.strip():
            chunks_text.append(current)

        chunks: list[Chunk] = []
        char_offset = 0
        for i, ct in enumerate(chunks_text):
            start = text.find(ct[:20], char_offset)
            if start == -1:
                start = char_offset
            end = start + len(ct)
            char_offset = end

            chunk = Chunk(
                content=ct,
                source=section.source,
                metadata={
                    "document_id": section.document_id,
                    "section_id": section.id,
                    "page": section.page_start,
                    "paragraph_index": i,
                    "char_start": start,
                    "char_end": end,
                    "token_count": _estimate_tokens(ct),
                },
            )
            chunks.append(chunk)

        return chunks

    def chunk_sections(self, sections: list[SectionBlock]) -> list[Chunk]:
        """批量切分多个 section。"""
        all_chunks: list[Chunk] = []
        for sec in sections:
            all_chunks.extend(self.chunk_section(sec))
        return all_chunks