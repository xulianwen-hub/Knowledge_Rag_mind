"""文档解析器模块。"""

from knowresearch.ingestion.parsers.base import ParserPort
from knowresearch.ingestion.parsers.pdf_parser import PDFParser
from knowresearch.ingestion.parsers.word_parser import WordParser

__all__ = ["ParserPort", "PDFParser", "WordParser"]