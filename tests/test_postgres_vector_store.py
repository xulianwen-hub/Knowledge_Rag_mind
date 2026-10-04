"""PostgresVectorStore 的接口级单元测试。

pgvector 不能直接跑在 SQLite 上，所以这里用一个最小 FakeEngine
验证 SQL 参数、返回结果解析和接口约定，不依赖真实 PostgreSQL。
"""

import json

from knowresearch.adapters.stores.postgres_vector_store import PostgresVectorStore
from knowresearch.core.schemas import Record


class _FakeConnection:
    def __init__(self, rows=None):
        self._rows = rows or []
        self.calls = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, stmt, params=None):
        self.calls.append((stmt, params))
        return _FakeResult(self._rows)


class _FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def mappings(self):
        return self

    def all(self):
        return self._rows


class _FakeEngine:
    def __init__(self, rows=None):
        self._rows = rows or []
        self.connection = _FakeConnection(self._rows)

    def connect(self):
        return self.connection

    def begin(self):
        return self.connection


def _make_store(rows=None):
    store = PostgresVectorStore.__new__(PostgresVectorStore)
    store._dimension = 3
    store._table_name = "chunk_vectors"
    store.engine = _FakeEngine(rows)
    return store


def test_search_parses_records_and_scores():
    record = Record(id="c1", kind="chunk", content="内容")
    rows = [
        {
            "data": json.dumps(record.model_dump(mode="json"), ensure_ascii=False),
            "score": 0.93,
        }
    ]
    store = _make_store(rows)

    results = store.search([0.1, 0.2, 0.3], top_k=5)

    assert len(results) == 1
    assert results[0][0].id == "c1"
    assert results[0][1] == 0.93


def test_search_empty_when_top_k_zero():
    store = _make_store()
    assert store.search([0.1, 0.2, 0.3], top_k=0) == []


def test_add_writes_record_parameters():
    store = _make_store()
    record = Record(id="c1", kind="chunk", content="内容")

    store.add([record], [[0.1, 0.2, 0.3]])

    stmt, params = store.engine.connection.calls[0]
    assert "INSERT INTO chunk_vectors" in str(stmt)
    assert params[0]["id"] == "c1"
    assert params[0]["embedding"] == [0.1, 0.2, 0.3]
