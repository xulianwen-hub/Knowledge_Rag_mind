"""DocumentStorePort 的 PostgreSQL 实现。"""

from __future__ import annotations

from sqlalchemy import MetaData

from knowresearch.adapters.postgres_base import (
    build_record_table,
    create_postgres_engine,
    delete_record,
    delete_records_by_ids,
    get_record,
    list_records,
    upsert_record,
)
from knowresearch.core.ports import DocumentStorePort
from knowresearch.core.schemas import Document, SectionBlock


class PostgresDocumentStore(DocumentStorePort):
    """基于 PostgreSQL 的文档/章节持久化实现。"""

    TABLE_NAME = "documents"
    KIND_DOCUMENT = "document"
    KIND_SECTION = "section"

    def __init__(self, database_url: str):
        self.engine = create_postgres_engine(database_url)
        self.metadata = MetaData()
        self.table = build_record_table(self.TABLE_NAME, self.metadata)
        self.metadata.create_all(self.engine)

    def save_document(self, doc: Document) -> None:
        upsert_record(self.engine, self.table, doc)

    def get_document(self, doc_id: str) -> Document | None:
        record = get_record(self.engine, self.table, doc_id)
        if record is None or record.kind != self.KIND_DOCUMENT:
            return None
        return Document(**record.model_dump())

    def get_document_by_hash(self, file_hash: str) -> Document | None:
        for document in self.list_documents():
            if document.file_hash == file_hash:
                return document
        return None

    def list_documents(self) -> list[Document]:
        records = list_records(self.engine, self.table, kind=self.KIND_DOCUMENT)
        return [Document(**record.model_dump()) for record in records]

    def save_sections(self, sections: list[SectionBlock]) -> None:
        if not sections:
            return
        for section in sections:
            upsert_record(self.engine, self.table, section)

    def get_sections(self, doc_id: str) -> list[SectionBlock]:
        records = list_records(self.engine, self.table, kind=self.KIND_SECTION)
        sections = [SectionBlock(**record.model_dump()) for record in records]
        matched = [section for section in sections if section.document_id == doc_id]
        matched.sort(
            key=lambda section: (
                section.section_level,
                section.metadata.get("order_index", 0),
            )
        )
        return matched

    def delete_document(self, doc_id: str) -> None:
        section_ids = [section.id for section in self.get_sections(doc_id)]
        delete_records_by_ids(self.engine, self.table, section_ids)
        delete_record(self.engine, self.table, doc_id)
