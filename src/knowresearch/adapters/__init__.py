"""适配器层：端口的具体实现（SQLite / 本地文件 / API 等）。

14 个端口的首版适配器：
- 模型提供商：MockEmbedding, MockReranker, DeepSeekLLM, QwenVLM, MockOCR
- 存储：LocalObjectStorage, InMemoryVectorStore, BM25FullTextStore
- 文档：SQLiteDocumentStore
- 记忆：SQLiteSessionStore, SQLiteWorkingMemory, SQLLongTermMemory
- 技能：SQLiteSkillStore
- 轨迹：SQLiteTraceStore
"""

from knowresearch.adapters.documents import (
    PostgresDocumentStore,
    SQLiteDocumentStore,
)
from knowresearch.adapters.feedback import PostgresFeedbackStore
from knowresearch.adapters.memory import (
    SQLLongTermMemory,
    PostgresLongTermMemory,
    PostgresMemoryCandidateStore,
    RedisSessionStore,
    SQLiteSessionStore,
    SQLiteWorkingMemory,
)
from knowresearch.adapters.providers import (
    CachedEmbeddingProvider,
    DeepSeekLLM,
    DemoLLM,
    LocalBGEZhEmbedding,
    LocalBGEReranker,
    MockEmbedding,
    MockOCR,
    MockReranker,
    QwenVLM,
)
from knowresearch.adapters.skills import (
    PostgresSkillCandidateStore,
    PostgresSkillStore,
    SQLiteSkillStore,
)
from knowresearch.adapters.stores import (
    BM25FullTextStore,
    InMemoryVectorStore,
    LocalObjectStorage,
    PostgresVectorStore,
)
from knowresearch.adapters.traces import PostgresTraceStore, SQLiteTraceStore

__all__ = [
    # 模型提供商
    "MockEmbedding",
    "CachedEmbeddingProvider",
    "MockReranker",
    "DeepSeekLLM",
    "DemoLLM",
    "LocalBGEZhEmbedding",
    "LocalBGEReranker",
    "QwenVLM",
    "MockOCR",
    # 存储
    "LocalObjectStorage",
    "InMemoryVectorStore",
    "BM25FullTextStore",
    # 文档
    "SQLiteDocumentStore",
    "PostgresDocumentStore",
    # 记忆
    "SQLiteSessionStore",
    "RedisSessionStore",
    "SQLiteWorkingMemory",
    "SQLLongTermMemory",
    "PostgresLongTermMemory",
    "PostgresMemoryCandidateStore",
    # 技能
    "SQLiteSkillStore",
    "PostgresSkillStore",
    "PostgresSkillCandidateStore",
    # 轨迹
    "SQLiteTraceStore",
    "PostgresTraceStore",
    # 生产向量存储
    "PostgresVectorStore",
    "PostgresFeedbackStore",
]
