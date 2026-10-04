"""验证首版适配器：13 个端口的具体实现是否正确。

测试范围：
1. 所有适配器能实例化
2. 存储端口（ObjectStorage / VectorStore / FullTextStore）的 CRUD
3. 记忆端口（Session / Working / LongTerm）的存取
4. 技能 / 轨迹端口的存取
5. 模型提供商端口的基本调用（LLM/VLM 不实际调用 API，只测实例化）
"""

import os
import sys
import tempfile
from datetime import datetime

from knowresearch.adapters import (
    BM25FullTextStore,
    DeepSeekLLM,
    InMemoryVectorStore,
    LocalObjectStorage,
    MockEmbedding,
    MockOCR,
    MockReranker,
    QwenVLM,
    SQLLongTermMemory,
    SQLiteSessionStore,
    SQLiteSkillStore,
    SQLiteTraceStore,
    SQLiteWorkingMemory,
)
from knowresearch.core.schemas import Record

DB_PATH = os.path.join(tempfile.gettempdir(), "knowresearch_test_adapters.db")
FILE_DIR = os.path.join(tempfile.gettempdir(), "knowresearch_test_files")


def _cleanup():
    """清理测试数据库和文件目录。"""
    from knowresearch.adapters.sqlite_base import close_connection

    close_connection(DB_PATH)
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    import shutil

    if os.path.exists(FILE_DIR):
        shutil.rmtree(FILE_DIR, ignore_errors=True)


def test_all_adapters_instantiable():
    """13 个适配器都能实例化。"""
    emb = MockEmbedding()
    rerank = MockReranker()
    llm = DeepSeekLLM()
    vlm = QwenVLM()
    ocr = MockOCR()
    obj = LocalObjectStorage(FILE_DIR)
    vec = InMemoryVectorStore(dimension=384)
    ft = BM25FullTextStore()
    sess = SQLiteSessionStore(DB_PATH)
    work = SQLiteWorkingMemory(DB_PATH)
    ltm = SQLLongTermMemory(DB_PATH)
    skill = SQLiteSkillStore(DB_PATH)
    trace = SQLiteTraceStore(DB_PATH)

    assert emb.dimension == 384
    assert vec.dimension == 384
    print("[PASS] 13 个适配器全部实例化成功")


def test_object_storage():
    """ObjectStoragePort：put / get / exists / delete / list_keys。"""
    storage = LocalObjectStorage(FILE_DIR)

    storage.put("test/hello.txt", b"hello world")
    assert storage.exists("test/hello.txt")
    assert storage.get("test/hello.txt") == b"hello world"

    keys = storage.list_keys("test/")
    assert "test/hello.txt" in keys

    storage.delete("test/hello.txt")
    assert not storage.exists("test/hello.txt")
    print("[PASS] ObjectStoragePort CRUD 正常")


def test_vector_store():
    """VectorStorePort：add / search / delete / clear。"""
    store = InMemoryVectorStore(dimension=384)
    emb = MockEmbedding(dimension=384)

    records = [
        Record(kind="chunk", content="机器学习是人工智能的一个分支"),
        Record(kind="chunk", content="深度学习使用多层神经网络"),
        Record(kind="chunk", content="自然语言处理关注文本理解"),
    ]
    embeddings = emb.embed_documents([r.content for r in records])
    store.add(records, embeddings)

    query_vec = emb.embed_query("神经网络")
    results = store.search(query_vec, top_k=2)
    assert len(results) <= 2
    assert len(results) > 0

    store.delete([records[0].id])
    results_after = store.search(query_vec, top_k=10)
    assert all(r[0].id != records[0].id for r in results_after)

    store.clear()
    assert store.search(query_vec) == []
    print("[PASS] VectorStorePort CRUD 正常")


def test_fulltext_store():
    """FullTextStorePort：add / search / delete / clear。"""
    store = BM25FullTextStore()

    records = [
        Record(kind="chunk", content="机器学习是人工智能的一个分支"),
        Record(kind="chunk", content="深度学习使用多层神经网络"),
        Record(kind="chunk", content="自然语言处理关注文本理解"),
    ]
    store.add(records)

    results = store.search("神经网络", top_k=5)
    assert len(results) > 0

    store.delete([records[0].id])
    store.clear()
    assert store.search("神经网络") == []
    print("[PASS] FullTextStorePort CRUD 正常")


def test_session_store():
    """SessionStorePort：save / load / delete。"""
    store = SQLiteSessionStore(DB_PATH)

    messages = [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好，有什么可以帮你的？"},
    ]
    store.save_messages("sess_1", messages, ttl_seconds=3600)

    loaded = store.load_messages("sess_1")
    assert loaded == messages

    store.delete_session("sess_1")
    assert store.load_messages("sess_1") == []
    print("[PASS] SessionStorePort CRUD 正常")


def test_working_memory():
    """WorkingMemoryPort：save / load / update / delete。"""
    store = SQLiteWorkingMemory(DB_PATH)

    state = {"step": 1, "thought": "开始分析问题"}
    store.save_state("task_1", state)

    loaded = store.load_state("task_1")
    assert loaded == state

    store.update_state("task_1", step=2, thought="正在检索")
    updated = store.load_state("task_1")
    assert updated["step"] == 2
    assert updated["thought"] == "正在检索"

    store.delete_task("task_1")
    assert store.load_state("task_1") is None
    print("[PASS] WorkingMemoryPort CRUD 正常")


def test_long_term_memory():
    """LongTermMemoryPort：add / search / update / delete。"""
    store = SQLLongTermMemory(DB_PATH)

    memory = Record(kind="glossary", content="RAG 是检索增强生成", metadata={"term": "RAG"})
    store.add(memory)

    results = store.search("RAG", top_k=5)
    assert any(r.id == memory.id for r in results)

    store.update(memory.id, verified=True)
    updated = store.search("RAG")[0]
    assert updated.metadata.get("verified") is True

    store.delete(memory.id)
    assert store.search("RAG") == []
    print("[PASS] LongTermMemoryPort CRUD 正常")


def test_skill_store():
    """SkillStorePort：save / get / search / list / update / delete。"""
    store = SQLiteSkillStore(DB_PATH)

    skill = Record(kind="skill", content="文献综述检索技能", metadata={"trigger": "综述"})
    store.save(skill)

    got = store.get(skill.id)
    assert got is not None
    assert got.content == "文献综述检索技能"

    results = store.search("综述")
    assert any(r.id == skill.id for r in results)

    all_skills = store.list_all()
    assert any(s.id == skill.id for s in all_skills)

    store.update(skill.id, usage_count=1)
    updated = store.get(skill.id)
    assert updated.metadata.get("usage_count") == 1

    store.delete(skill.id)
    assert store.get(skill.id) is None
    print("[PASS] SkillStorePort CRUD 正常")


def test_trace_store():
    """TraceStorePort：save / get / list / delete。"""
    store = SQLiteTraceStore(DB_PATH)

    trace = Record(kind="trace", content="一次 RAG 问答轨迹", metadata={"session_id": "s1"})
    store.save(trace)

    got = store.get(trace.id)
    assert got is not None

    all_traces = store.list()
    assert any(t.id == trace.id for t in all_traces)

    filtered = store.list({"kind": "trace"})
    assert any(t.id == trace.id for t in filtered)

    meta_filtered = store.list({"session_id": "s1"})
    assert any(t.id == trace.id for t in meta_filtered)

    store.delete(trace.id)
    assert store.get(trace.id) is None
    print("[PASS] TraceStorePort CRUD 正常")


def test_embedding_provider():
    """EmbeddingProvider：embed_documents / embed_query 维度一致。"""
    emb = MockEmbedding(dimension=384)

    docs = emb.embed_documents(["文本一", "文本二"])
    assert len(docs) == 2
    assert all(len(v) == 384 for v in docs)

    query = emb.embed_query("查询")
    assert len(query) == 384
    print("[PASS] EmbeddingProvider 维度正确")


def test_reranker_provider():
    """RerankerProvider：rerank 返回 (索引, 分数) 列表。"""
    rerank = MockReranker()

    results = rerank.rerank("查询", ["doc1", "doc2", "doc3"], top_k=2)
    assert len(results) == 2
    assert all(isinstance(r, tuple) and len(r) == 2 for r in results)
    print("[PASS] RerankerProvider 返回格式正确")


def main():
    import io
    import traceback

    output = io.StringIO()

    def log(msg):
        print(msg)
        output.write(msg + "\n")

    try:
        log("=" * 60)
        log("首版适配器验证（13 个端口）")
        log("=" * 60)

        _cleanup()

        test_all_adapters_instantiable()
        test_object_storage()
        test_vector_store()
        test_fulltext_store()
        test_session_store()
        test_working_memory()
        test_long_term_memory()
        test_skill_store()
        test_trace_store()
        test_embedding_provider()
        test_reranker_provider()

        _cleanup()

        log("=" * 60)
        log("全部验证通过（11/11）")
        log("=" * 60)
    except Exception:
        log("ERROR:")
        log(traceback.format_exc())
    finally:
        with open(os.path.join(os.path.dirname(__file__), "_test_adapters_result.txt"), "w", encoding="utf-8") as f:
            f.write(output.getvalue())


if __name__ == "__main__":
    main()
    sys.exit(0)