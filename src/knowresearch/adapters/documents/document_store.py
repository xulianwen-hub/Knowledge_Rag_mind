"""DocumentStorePort 的 SQLite 实现。

Document 和 SectionBlock 共用 documents 表，通过 kind 区分。
复用 sqlite_base 的通用 Record 函数，row_to_record 返回基类 Record，
此处将其转换回 Document / SectionBlock 子类。
"""

from knowresearch.adapters.sqlite_base import (
    delete_record,
    ensure_table,
    get_connection,
    get_record,
    insert_record,
    list_records,
    record_to_row,
    transaction,
)
from knowresearch.core.schemas import Document, SectionBlock

TABLE = "documents"
KIND_DOCUMENT = "document"
KIND_SECTION = "section"


class SQLiteDocumentStore:
    """基于 SQLite 的文档/章节持久化实现。"""

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        ensure_table(db_path, TABLE)

    def save_document(self, doc: Document) -> None:
        insert_record(self._db_path, TABLE, doc)

    def get_document(self, doc_id: str) -> Document | None:
        record = get_record(self._db_path, TABLE, doc_id)
        if record is None or record.kind != KIND_DOCUMENT:
            return None
        return Document(**record.model_dump())

    def get_document_by_hash(self, file_hash: str) -> Document | None:
        for doc in self.list_documents():
            if doc.file_hash == file_hash:
                return doc
        return None

    def list_documents(self) -> list[Document]:
        records = list_records(self._db_path, TABLE, filters={"kind": KIND_DOCUMENT})
        return [Document(**r.model_dump()) for r in records]

    def save_sections(self, sections: list[SectionBlock]) -> None:
        if not sections:
            return
        with transaction(self._db_path) as conn:
            conn.executemany(
                f"INSERT OR REPLACE INTO {TABLE} "
                f"(id, kind, source, created_at, updated_at, data) "
                f"VALUES (?, ?, ?, ?, ?, ?)",
                [record_to_row(s) for s in sections],
            )

    def get_sections(self, doc_id: str) -> list[SectionBlock]:
        records = list_records(self._db_path, TABLE, filters={"kind": KIND_SECTION})
        sections = [SectionBlock(**r.model_dump()) for r in records]
        doc_sections = [s for s in sections if s.document_id == doc_id]
        doc_sections.sort(
            key=lambda s: (s.section_level, s.metadata.get("order_index", 0))
        )
        return doc_sections

    def delete_document(self, doc_id: str) -> None:
        ensure_table(self._db_path, TABLE)
        with transaction(self._db_path) as conn:
            section_ids = [
                s.id for s in self.get_sections(doc_id)
            ]
            for sid in section_ids:
                delete_record(self._db_path, TABLE, sid)
            delete_record(self._db_path, TABLE, doc_id)