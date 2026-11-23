"""文档摄取：多源解析（PDF / Word / 图片）、切块、去重入库。"""

from knowresearch.ingestion.chunker import Chunker
from knowresearch.ingestion.media_blocks import (
    FigureBlock,
    PDFMediaExtractor,
    TableBlock,
    table_to_markdown,
)
from knowresearch.ingestion.pipeline import IngestPipeline, IngestResult, IngestSummary
from knowresearch.ingestion.verbalizer import FigureVerbalizer, TableTranscriber

__all__ = [
    "Chunker",
    "IngestPipeline",
    "IngestResult",
    "IngestSummary",
    "PDFMediaExtractor",
    "TableBlock",
    "FigureBlock",
    "table_to_markdown",
    "FigureVerbalizer",
    "TableTranscriber",
]
