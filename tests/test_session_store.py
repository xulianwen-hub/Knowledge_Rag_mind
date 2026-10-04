"""M4 Step 1 测试：SQLiteSessionStore（短期记忆 + TTL）。"""

import time

from knowresearch.adapters.memory import SQLiteSessionStore
from knowresearch.adapters.sqlite_base import close_connection


def _make_store(tmp_path):
    return SQLiteSessionStore(str(tmp_path / "test.db"))


def test_save_and_load_messages():
    """保存后能正确加载。"""
    import tempfile, pathlib
    with tempfile.TemporaryDirectory() as d:
        store = _make_store(pathlib.Path(d))
        messages = [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好，有什么可以帮你？"},
        ]
        store.save_messages("sess1", messages)
        loaded = store.load_messages("sess1")
        assert loaded == messages
        close_connection(str(pathlib.Path(d) / "test.db"))


def test_load_nonexistent_session_returns_empty():
    """加载不存在的会话返回空列表。"""
    import tempfile, pathlib
    with tempfile.TemporaryDirectory() as d:
        store = _make_store(pathlib.Path(d))
        assert store.load_messages("nonexistent") == []
        close_connection(str(pathlib.Path(d) / "test.db"))


def test_session_isolation():
    """不同 session_id 互不影响。"""
    import tempfile, pathlib
    with tempfile.TemporaryDirectory() as d:
        store = _make_store(pathlib.Path(d))
        store.save_messages("s1", [{"role": "user", "content": "会话1"}])
        store.save_messages("s2", [{"role": "user", "content": "会话2"}])
        assert store.load_messages("s1") == [{"role": "user", "content": "会话1"}]
        assert store.load_messages("s2") == [{"role": "user", "content": "会话2"}]
        close_connection(str(pathlib.Path(d) / "test.db"))


def test_overwrite_existing_session():
    """重复保存同一 session_id 会覆盖。"""
    import tempfile, pathlib
    with tempfile.TemporaryDirectory() as d:
        store = _make_store(pathlib.Path(d))
        store.save_messages("s1", [{"role": "user", "content": "旧消息"}])
        store.save_messages("s1", [{"role": "user", "content": "新消息"}])
        assert store.load_messages("s1") == [{"role": "user", "content": "新消息"}]
        close_connection(str(pathlib.Path(d) / "test.db"))


def test_delete_session():
    """删除会话后加载返回空。"""
    import tempfile, pathlib
    with tempfile.TemporaryDirectory() as d:
        store = _make_store(pathlib.Path(d))
        store.save_messages("s1", [{"role": "user", "content": "消息"}])
        store.delete_session("s1")
        assert store.load_messages("s1") == []
        close_connection(str(pathlib.Path(d) / "test.db"))


def test_ttl_expiry():
    """TTL 过期后加载返回空（惰性清理）。"""
    import tempfile, pathlib
    with tempfile.TemporaryDirectory() as d:
        store = _make_store(pathlib.Path(d))
        store.save_messages("s1", [{"role": "user", "content": "消息"}], ttl_seconds=1)
        assert store.load_messages("s1") == [{"role": "user", "content": "消息"}]
        time.sleep(1.1)
        assert store.load_messages("s1") == []
        close_connection(str(pathlib.Path(d) / "test.db"))


def test_expired_session_does_not_affect_others():
    """一个会话过期不影响其他未过期会话。"""
    import tempfile, pathlib
    with tempfile.TemporaryDirectory() as d:
        store = _make_store(pathlib.Path(d))
        store.save_messages("expired", [{"role": "user", "content": "会过期"}], ttl_seconds=1)
        store.save_messages("alive", [{"role": "user", "content": "不会过期"}], ttl_seconds=3600)
        time.sleep(1.1)
        assert store.load_messages("expired") == []
        assert store.load_messages("alive") == [{"role": "user", "content": "不会过期"}]
        close_connection(str(pathlib.Path(d) / "test.db"))