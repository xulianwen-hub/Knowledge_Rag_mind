"""PostgresDocumentStore 在 SQLite 测试环境下的接口测试。

这里借 SQLite 验证通用 Record 表逻辑和 DocumentStorePort 行为；
生产环境使用 PostgreSQL 时，JSONType 会编译为 JSONB。
"""

from knowresearch.adapters.documents import PostgresDocumentStore
from knowresearch.core.schemas import Document, SectionBlock


def _make_store(tmp_path):
    db_path = tmp_path / "documents.db"
    store = PostgresDocumentStore(f"sqlite:///{db_path}")
    return store, db_path


def test_save_and_get_document(tmp_path):
    store, _ = _make_store(tmp_path)
    doc = Document(
        id="doc-1",
        metadata={"file_name": "论文.pdf", "file_hash": "abc"},
    )
    store.save_document(doc)

    got = store.get_document("doc-1")
    assert got is not None
    assert got.file_name == "论文.pdf"
    assert got.file_hash == "abc"


def test_get_document_by_hash(tmp_path):
    store, _ = _make_store(tmp_path)
    doc = Document(
        id="doc-1",
        metadata={"file_name": "论文.pdf", "file_hash": "abc"},
    )
    store.save_document(doc)
    assert store.get_document_by_hash("abc").id == "doc-1"
    assert store.get_document_by_hash("missing") is None


def test_save_and_get_sections(tmp_path):
    store, _ = _make_store(tmp_path)
    section = SectionBlock(
        id="sec-1",
        metadata={
            "document_id": "doc-1",
            "section_title": "引言",
            "section_level": 1,
            "order_index": 0,
        },
    )
    store.save_sections([section])

    sections = store.get_sections("doc-1")
    assert len(sections) == 1
    assert sections[0].section_title == "引言"


def test_delete_document_cascades_sections(tmp_path):
    store, _ = _make_store(tmp_path)
    doc = Document(id="doc-1", metadata={"file_name": "论文.pdf"})
    section = SectionBlock(
        id="sec-1",
        metadata={"document_id": "doc-1", "section_title": "引言"},
    )
    store.save_document(doc)
    store.save_sections([section])

    store.delete_document("doc-1")
    assert store.get_document("doc-1") is None
    assert store.get_sections("doc-1") == []
