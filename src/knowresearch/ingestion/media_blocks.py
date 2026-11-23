"""PDF 图表提取：定位图/表，表格转 Markdown，图渲染成图片。

这是「方案一：图像描述化」的地基。这里只做确定性的轻量提取：
- 表：用 PyMuPDF 的 find_tables 定位 + 还原行列 → Markdown 文本。
- 图：用 get_images / get_image_rects 定位 bbox 和所在页，抽图题 caption，
  并能把该区域渲染成 PNG 供 VLM 理解。

不在这里做语义理解（VLM 描述由 FigureVerbalizer 负责），也不改动
PDFParser 现有行为，完全作为增量模块挂在摄取流水线里。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import fitz  # PyMuPDF


_CAPTION_HINT = re.compile(r"^(图\s*\d*|Fig\.?\s*\d*|Figure\s*\d*|表\s*\d*|Table\s*\d*)", re.IGNORECASE)


@dataclass
class TableBlock:
    """一张表格的结构化结果。"""

    page: int
    bbox: tuple[float, float, float, float]
    markdown: str
    caption: str = ""


@dataclass
class FigureBlock:
    """一张图的位置信息。"""

    page: int
    bbox: tuple[float, float, float, float]
    xref: int
    caption: str = ""


def table_to_markdown(rows: list[list[str]]) -> str:
    """把行列数据转成 Markdown 表格文本。"""
    if not rows:
        return ""
    clean: list[list[str]] = [
        [str(cell).strip().replace("\n", " ") if cell is not None else "" for cell in row]
        for row in rows
    ]
    clean = [row for row in clean if any(row)]
    if not clean:
        return ""

    ncols = max(len(row) for row in clean)
    clean = [row + [""] * (ncols - len(row)) for row in clean]

    lines = [
        "| " + " | ".join(clean[0]) + " |",
        "| " + " | ".join(["---"] * ncols) + " |",
    ]
    for row in clean[1:]:
        cells = [cell.replace("|", "\\|") for cell in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


class PDFMediaExtractor:
    """从 PDF 里提取图和表。"""

    def __init__(self, min_image_side: float = 30.0, caption_search_span: float = 40.0):
        self.min_image_side = min_image_side
        self.caption_search_span = caption_search_span

    def extract_tables(self, file_path: str) -> list[TableBlock]:
        tables: list[TableBlock] = []
        doc = fitz.open(file_path)
        try:
            for page_index in range(doc.page_count):
                page = doc[page_index]
                for table in page.find_tables():
                    try:
                        rows = table.extract()
                        markdown = table_to_markdown(rows)
                    except Exception:
                        markdown = ""
                    bbox = tuple(float(v) for v in table.bbox)
                    tables.append(
                        TableBlock(
                            page=page_index + 1,
                            bbox=bbox,
                            markdown=markdown,
                            caption=self._find_caption(page, bbox),
                        )
                    )
        finally:
            doc.close()
        return tables

    def extract_figures(self, file_path: str) -> list[FigureBlock]:
        figures: list[FigureBlock] = []
        doc = fitz.open(file_path)
        try:
            for page_index in range(doc.page_count):
                page = doc[page_index]
                for image in page.get_images(full=True):
                    xref = image[0]
                    for rect in page.get_image_rects(xref):
                        if min(rect.width, rect.height) < self.min_image_side:
                            continue
                        bbox = (rect.x0, rect.y0, rect.x1, rect.y1)
                        figures.append(
                            FigureBlock(
                                page=page_index + 1,
                                bbox=bbox,
                                xref=xref,
                                caption=self._find_caption(page, bbox),
                            )
                        )
        finally:
            doc.close()
        return figures

    def render_figure(self, file_path: str, figure: FigureBlock, dpi: int = 144) -> bytes:
        """把图所在的页面区域渲染成 PNG 字节。"""
        return self.render_region(file_path, figure.page, figure.bbox, dpi=dpi)

    def render_region(
        self,
        file_path: str,
        page: int,
        bbox: tuple[float, float, float, float],
        dpi: int = 144,
    ) -> bytes:
        """把任意页面区域渲染成 PNG 字节。"""
        doc = fitz.open(file_path)
        try:
            target_page = doc[page - 1]
            clip = fitz.Rect(*bbox)
            zoom = dpi / 72.0
            matrix = fitz.Matrix(zoom, zoom)
            pixmap = target_page.get_pixmap(clip=clip, matrix=matrix)
            return pixmap.tobytes("png")
        finally:
            doc.close()

    def _find_caption(
        self, page: fitz.Page, bbox: tuple[float, float, float, float]
    ) -> str:
        """启发式抽取图/表下方最近的一句题注，失败返回空字符串。"""
        x0, y0, x1, y1 = bbox
        candidates: list[tuple[float, str]] = []
        for block in page.get_text("blocks"):
            if len(block) < 7 or block[6] != 0:  # 只取文本块
                continue
            bx0, by0, bx1, by1, text = block[0], block[1], block[2], block[3], block[4]
            text = text.strip()
            if not text:
                continue
            # 位于图/表下方、水平上有重叠的文本块
            if by0 < y1 - 2 or by0 > y1 + self.caption_search_span:
                continue
            if bx1 < x0 - 5 or bx0 > x1 + 5:
                continue
            candidates.append((by0, text))
        if not candidates:
            return ""
        candidates.sort(key=lambda item: item[0])
        text = candidates[0][1]
        if _CAPTION_HINT.match(text) or len(text) <= 120:
            return text
        return ""
