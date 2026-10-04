"""
验证 M1 数据模型：Record / Document / SectionBlock / Chunk。

验证内容：
1. 模块能正常 import
2. 四个类能实例化，字段默认值正确
3. 必填字段（kind）缺失时 pydantic 报错
4. 序列化（model_dump）/ 反序列化（model_validate）能跑通
5. 属性访问（@property）能正确读取 metadata
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from knowresearch.core.schemas import Chunk, Document, Record, SectionBlock
from pydantic import ValidationError


def test_import():
    print("[1/5] import 检查 ...", end=" ")
    assert Record is not None
    assert Document is not None
    assert SectionBlock is not None
    assert Chunk is not None
    print("OK")


def test_default_fields():
    print("[2/5] 默认字段检查 ...", end=" ")
    doc = Document(
        source="/data/raw/test.pdf",
        metadata={"file_name": "test.pdf", "file_hash": "abc123", "total_pages": 10},
    )
    assert doc.id != ""
    assert doc.kind == "document"
    assert doc.content == ""
    assert doc.source == "/data/raw/test.pdf"
    assert doc.metadata == {"file_name": "test.pdf", "file_hash": "abc123", "total_pages": 10}
    assert doc.relations == []
    assert doc.file_name == "test.pdf"
    assert doc.file_hash == "abc123"
    assert doc.total_pages == 10
    assert doc.created_at <= doc.updated_at
    print("OK")


def test_required_kind():
    print("[3/5] 必填字段校验（kind 缺失应报错）...", end=" ")
    try:
        Record()
        assert False, "应该抛出 ValidationError"
    except ValidationError:
        print("OK")


def test_serialization():
    print("[4/5] 序列化 / 反序列化 ...", end=" ")
    chunk = Chunk(
        content="这是一个测试子块",
        source="/data/raw/test.pdf",
        metadata={
            "document_id": "doc-1",
            "section_id": "sec-1",
            "page": 3,
            "paragraph_index": 2,
            "char_start": 100,
            "char_end": 200,
            "token_count": 15,
        },
    )
    dumped = chunk.model_dump(mode="json")
    assert dumped["kind"] == "chunk"
    assert dumped["content"] == "这是一个测试子块"
    assert dumped["metadata"]["page"] == 3

    restored = Chunk.model_validate(dumped)
    assert restored.id == chunk.id
    assert restored.content == chunk.content
    assert restored.page == 3
    assert restored.char_start == 100
    assert restored.char_end == 200
    assert restored.token_count == 15
    print("OK")


def test_hierarchy():
    print("[5/5] Document → Section → Chunk 层级关系 ...", end=" ")
    doc = Document(source="test.pdf", metadata={"file_name": "test.pdf"})
    section = SectionBlock(
        content="第一章 引言",
        source="test.pdf",
        metadata={
            "document_id": doc.id,
            "section_title": "引言",
            "section_level": 1,
            "page_start": 1,
            "page_end": 3,
        },
    )
    chunk = Chunk(
        content="本文研究了 RAG 系统的设计。",
        source="test.pdf",
        metadata={
            "document_id": doc.id,
            "section_id": section.id,
            "page": 1,
            "paragraph_index": 0,
            "char_start": 0,
            "char_end": 16,
            "token_count": 8,
        },
    )
    assert section.document_id == doc.id
    assert chunk.document_id == doc.id
    assert chunk.section_id == section.id
    assert chunk.page == 1
    print("OK")


if __name__ == "__main__":
    test_import()
    test_default_fields()
    test_required_kind()
    test_serialization()
    test_hierarchy()
    print("\n✅ 全部 5 项验证通过！")