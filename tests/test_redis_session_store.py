"""RedisSessionStore 使用 fakeredis 的单元测试。"""

import fakeredis

from knowresearch.adapters.memory.redis_session_store import RedisSessionStore


def _make_store():
    server = fakeredis.FakeServer()
    store = RedisSessionStore.__new__(RedisSessionStore)
    store._client = fakeredis.FakeStrictRedis(server=server, decode_responses=True)
    store._prefix = "session"
    store._default_ttl_seconds = 3600
    return store


def test_save_and_load_messages():
    store = _make_store()
    messages = [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好，有什么可以帮你？"},
    ]
    store.save_messages("sess-1", messages, summary="摘要")

    assert store.load_messages("sess-1") == messages
    assert store.load_summary("sess-1") == "摘要"


def test_load_nonexistent_returns_empty():
    store = _make_store()
    assert store.load_messages("nonexistent") == []
    assert store.load_summary("nonexistent") == ""


def test_session_isolation():
    store = _make_store()
    store.save_messages("s1", [{"role": "user", "content": "会话1"}])
    store.save_messages("s2", [{"role": "user", "content": "会话2"}])

    assert store.load_messages("s1") == [{"role": "user", "content": "会话1"}]
    assert store.load_messages("s2") == [{"role": "user", "content": "会话2"}]


def test_delete_session():
    store = _make_store()
    store.save_messages("s1", [{"role": "user", "content": "消息"}])
    store.delete_session("s1")
    assert store.load_messages("s1") == []
