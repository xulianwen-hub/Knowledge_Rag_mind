"""验证端口接口：导入、抽象性、继承关系。"""

import sys
from abc import ABC

from knowresearch.core.ports import (
    EmbeddingProvider,
    FullTextStorePort,
    LLMProvider,
    LongTermMemoryPort,
    ObjectStoragePort,
    OCRProvider,
    RerankerProvider,
    SessionStorePort,
    SkillStorePort,
    TraceStorePort,
    VectorStorePort,
    VLMProvider,
    WorkingMemoryPort,
)

ALL_PORTS = [
    EmbeddingProvider,
    RerankerProvider,
    LLMProvider,
    VLMProvider,
    OCRProvider,
    VectorStorePort,
    FullTextStorePort,
    ObjectStoragePort,
    SessionStorePort,
    WorkingMemoryPort,
    LongTermMemoryPort,
    SkillStorePort,
    TraceStorePort,
]


def test_import_all_ports():
    """13 个端口全部能导入。"""
    assert len(ALL_PORTS) == 13, f"预期 13 个端口，实际 {len(ALL_PORTS)}"
    for port in ALL_PORTS:
        assert isinstance(port, type), f"{port.__name__} 不是类"
    print(f"[PASS] 13 个端口全部导入成功")


def test_all_are_abstract():
    """所有端口都是抽象基类，不能直接实例化。"""
    for port in ALL_PORTS:
        assert issubclass(port, ABC), f"{port.__name__} 没有继承 ABC"
        assert len(port.__abstractmethods__) > 0, (
            f"{port.__name__} 没有抽象方法，不是真正的抽象类"
        )
        try:
            port()
            assert False, f"{port.__name__} 不应该能被实例化"
        except TypeError:
            pass
    print(f"[PASS] 13 个端口都是抽象基类，无法直接实例化")


def test_port_categories():
    """端口分 5 类，检查继承关系。"""
    providers = [EmbeddingProvider, RerankerProvider, LLMProvider, VLMProvider, OCRProvider]
    stores = [VectorStorePort, FullTextStorePort, ObjectStoragePort]
    memory = [SessionStorePort, WorkingMemoryPort, LongTermMemoryPort]
    skills = [SkillStorePort]
    traces = [TraceStorePort]

    assert len(providers) == 5
    assert len(stores) == 3
    assert len(memory) == 3
    assert len(skills) == 1
    assert len(traces) == 1
    print("[PASS] 端口分类正确：5 provider + 3 store + 3 memory + 1 skill + 1 trace")


def test_abstract_methods_exist():
    """检查关键端口的抽象方法是否存在。"""
    assert "embed_documents" in EmbeddingProvider.__abstractmethods__
    assert "embed_query" in EmbeddingProvider.__abstractmethods__
    assert "dimension" in EmbeddingProvider.__abstractmethods__

    assert "rerank" in RerankerProvider.__abstractmethods__

    assert "generate" in LLMProvider.__abstractmethods__
    assert "generate_with_messages" in LLMProvider.__abstractmethods__

    assert "add" in VectorStorePort.__abstractmethods__
    assert "search" in VectorStorePort.__abstractmethods__

    assert "put" in ObjectStoragePort.__abstractmethods__
    assert "get" in ObjectStoragePort.__abstractmethods__
    print("[PASS] 关键端口的抽象方法定义正确")


def test_can_create_concrete_subclass():
    """验证可以通过实现所有抽象方法创建具体子类。"""

    class FakeEmbedding(EmbeddingProvider):
        @property
        def dimension(self):
            return 384

        def embed_documents(self, texts):
            return [[0.1] * 384 for _ in texts]

        def embed_query(self, text):
            return [0.1] * 384

    emb = FakeEmbedding()
    assert emb.dimension == 384
    assert len(emb.embed_query("test")) == 384
    assert len(emb.embed_documents(["a", "b"])) == 2
    print("[PASS] 可以创建具体子类并实现抽象方法")


def main():
    print("=" * 60)
    print("端口接口验证")
    print("=" * 60)

    test_import_all_ports()
    test_all_are_abstract()
    test_port_categories()
    test_abstract_methods_exist()
    test_can_create_concrete_subclass()

    print("=" * 60)
    print("全部验证通过")
    print("=" * 60)


if __name__ == "__main__":
    main()
    sys.exit(0)