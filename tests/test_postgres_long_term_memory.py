"""PostgresLongTermMemory 在 SQLite 兼容环境下的接口测试。"""

from knowresearch.adapters.memory import PostgresLongTermMemory
from knowresearch.core.schemas import Record


def _make_store(tmp_path) -> PostgresLongTermMemory:
    return PostgresLongTermMemory(f"sqlite:///{tmp_path / 'ltm.db'}")


def test_add_and_list_by_user(tmp_path):
    store = _make_store(tmp_path)
    memory = Record(kind="profile", content="用户研究船舶水动力学")
    store.add(memory, user_id="user-a")

    records = store.list(user_id="user-a", kind="profile")
    assert len(records) == 1
    assert records[0].content == "用户研究船舶水动力学"
    assert records[0].metadata["user_id"] == "user-a"


def test_user_isolation(tmp_path):
    store = _make_store(tmp_path)
    store.add(Record(kind="profile", content="A 的画像"), user_id="user-a")
    store.add(Record(kind="profile", content="B 的画像"), user_id="user-b")

    assert len(store.list(user_id="user-a")) == 1
    assert len(store.list(user_id="user-b")) == 1
    assert store.list(user_id="user-a")[0].content == "A 的画像"


def test_search_profile(tmp_path):
    store = _make_store(tmp_path)
    store.add(
        Record(
            kind="profile",
            content="用户关注物理信息神经网络",
            metadata={"dimension": "research_field"},
        ),
        user_id="user-a",
    )

    assert len(store.search("神经网络", user_id="user-a")) == 1
    assert store.search("船舶", user_id="user-a") == []


def test_update_and_delete_with_user_isolation(tmp_path):
    store = _make_store(tmp_path)
    memory = Record(kind="profile", content="旧画像")
    store.add(memory, user_id="user-a")

    store.update(memory.id, user_id="user-b", verified=True)
    assert store.get(memory.id, user_id="user-a").metadata.get("verified") is None

    store.update(memory.id, user_id="user-a", verified=True)
    assert store.get(memory.id, user_id="user-a").metadata["verified"] is True

    store.delete(memory.id, user_id="user-b")
    assert store.get(memory.id, user_id="user-a") is not None

    store.delete(memory.id, user_id="user-a")
    assert store.get(memory.id, user_id="user-a") is None


def test_upsert_profile_insert_and_update(tmp_path):
    store = _make_store(tmp_path)

    first = store.upsert_profile(
        user_id="user-a",
        content="用户研究船舶水动力学",
        dimension="research_field",
        confidence=0.8,
    )
    assert first.metadata["dimension"] == "research_field"

    second = store.upsert_profile(
        user_id="user-a",
        content="用户研究船舶水动力学和襟翼舵",
        dimension="research_field",
        confidence=0.9,
    )
    assert second.id == first.id
    assert "襟翼舵" in store.get_profile("user-a", "research_field").content
    assert len(store.list(user_id="user-a", kind="profile")) == 1


def test_upsert_profile_does_not_overwrite_explicit(tmp_path):
    store = _make_store(tmp_path)
    explicit = store.upsert_profile(
        user_id="user-a",
        content="用户明确要求研究船舶水动力学",
        dimension="research_field",
        explicit=True,
    )

    store.upsert_profile(
        user_id="user-a",
        content="模型推断用户研究方向变化",
        dimension="research_field",
        explicit=False,
    )

    assert store.get_profile("user-a", "research_field").content == explicit.content
