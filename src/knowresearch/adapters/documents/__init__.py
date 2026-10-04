"""文档存储适配器。"""

from knowresearch.adapters.documents.document_store import SQLiteDocumentStore
from knowresearch.adapters.documents.postgres_document_store import PostgresDocumentStore

__all__ = ["SQLiteDocumentStore", "PostgresDocumentStore"]
