"""P1 真实 PostgreSQL/Redis 集成测试。

默认跳过，需要本地 Docker Compose 已启动后手动开启：
    $env:RUN_P1_INTEGRATION='1'
    python -m pytest tests/test_p1_integration.py -q
"""

import os
import uuid

import pytest

from knowresearch.adapters.documents import PostgresDocumentStore
from knowresearch.adapters.memory import RedisSessionStore
from knowresearch.adapters.stores import PostgresVectorStore
from knowresearch.adapters.traces import PostgresTraceStore
from knowresearch.core.schemas import Document, Record


pytestmark = pytest.mark.skipif(
    os.getenv("RUN_P1_INTEGRATION") != "1",
    reason="需要本地 PostgreSQL/Redis 服务，设置 RUN_P1_INTEGRATION=1 后运行",
)


POSTGRES_URL = os.getenv(
    "P1_POSTGRES_URL",
    "postgresql+psycopg2://knowresearch:knowresearch@localhost:5432/knowresearch",
)
REDIS_URL = os.getenv("P1_REDIS_URL", "redis://localhost:6379/10")


def test_postgres_document_store_integration():
    store = PostgresDocumentStore(POSTGRES_URL)
    doc_id = f"doc-{uuid.uuid4().hex}"
    doc = Document(
        id=doc_id,
        metadata={"file_name": "p1-integration.pdf", "file_hash": uuid.uuid4().hex},
    )

    try:
        store.save_document(doc)
        got = store.get_document(doc_id)
        assert got is not None
        assert got.file_name == "p1-integration.pdf"
        assert store.get_document_by_hash(doc.file_hash).id == doc_id
    finally:
        store.delete_document(doc_id)


def test_postgres_trace_store_integration():
    store = PostgresTraceStore(POSTGRES_URL)
    trace_id = f"trace-{uuid.uuid4().hex}"
    trace = Record(
        id=trace_id,
        kind="trace",
        content="P1 integration trace",
        metadata={"session_id": "p1-session"},
    )

    try:
        store.save(trace)
        got = store.get(trace_id)
        assert got is not None
        assert got.content == "P1 integration trace"
        assert any(item.id == trace_id for item in store.list({"session_id": "p1-session"}))
    finally:
        store.delete(trace_id)


def test_postgres_vector_store_integration():
    store = PostgresVectorStore(POSTGRES_URL, dimension=3)
    record_id = f"chunk-{uuid.uuid4().hex}"
    record = Record(id=record_id, kind="chunk", content="P1 vector integration")

    try:
        store.add([record], [[0.1, 0.2, 0.3]])
        results = store.search([0.1, 0.2, 0.3], top_k=1)
        assert results
        assert results[0][0].id == record_id
        assert results[0][1] > 0.99
    finally:
        store.delete([record_id])


def test_redis_session_store_integration():
    store = RedisSessionStore(REDIS_URL)
    session_id = f"session-{uuid.uuid4().hex}"
    messages = [{"role": "user", "content": "P1 Redis integration"}]

    try:
        store.save_messages(session_id, messages, ttl_seconds=60, summary="P1")
        assert store.load_messages(session_id) == messages
        assert store.load_summary(session_id) == "P1"
    finally:
        store.delete_session(session_id)
