"""PostgresTraceStore 在 SQLite 测试环境下的接口测试。"""

from knowresearch.adapters.traces import PostgresTraceStore
from knowresearch.core.schemas import Record


def _make_store(tmp_path):
    db_path = tmp_path / "traces.db"
    return PostgresTraceStore(f"sqlite:///{db_path}")


def test_save_and_get(tmp_path):
    store = _make_store(tmp_path)
    trace = Record(
        id="trace-1",
        kind="trace",
        content="一次问答轨迹",
        metadata={"session_id": "s1"},
    )
    store.save(trace)

    got = store.get("trace-1")
    assert got is not None
    assert got.content == "一次问答轨迹"
    assert got.metadata["session_id"] == "s1"


def test_list_all_and_filter_by_kind(tmp_path):
    store = _make_store(tmp_path)
    store.save(Record(id="t1", kind="trace", content="轨迹一"))
    store.save(Record(id="t2", kind="trace", content="轨迹二"))
    store.save(Record(id="t3", kind="other", content="其他"))

    assert len(store.list()) == 3
    traces = store.list({"kind": "trace"})
    assert {t.id for t in traces} == {"t1", "t2"}


def test_list_filter_by_metadata(tmp_path):
    store = _make_store(tmp_path)
    store.save(
        Record(
            id="t1",
            kind="trace",
            content="轨迹",
            metadata={"session_id": "s1"},
        )
    )
    store.save(
        Record(
            id="t2",
            kind="trace",
            content="轨迹",
            metadata={"session_id": "s2"},
        )
    )

    result = store.list({"session_id": "s2"})
    assert len(result) == 1
    assert result[0].id == "t2"


def test_delete(tmp_path):
    store = _make_store(tmp_path)
    store.save(Record(id="t1", kind="trace", content="轨迹"))
    store.delete("t1")
    assert store.get("t1") is None
