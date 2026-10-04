"""端口定义：存储 / 模型 / 服务的抽象接口（依赖倒置，便于替换实现）。

14 个端口分 6 类：
- 模型提供商（providers.py）：EmbeddingProvider, RerankerProvider, LLMProvider, VLMProvider, OCRProvider
- 存储（stores.py）：VectorStorePort, FullTextStorePort, ObjectStoragePort
- 文档（documents.py）：DocumentStorePort
- 记忆（memory.py）：SessionStorePort, WorkingMemoryPort, LongTermMemoryPort
- 技能（skills.py）：SkillStorePort
- 轨迹（traces.py）：TraceStorePort
"""

from knowresearch.core.ports.documents import DocumentStorePort
from knowresearch.core.ports.memory import (
    LongTermMemoryPort,
    MemoryCandidateStorePort,
    MemoryContextPort,
    SessionStorePort,
    WorkingMemoryPort,
)
from knowresearch.core.ports.providers import (
    EmbeddingProvider,
    LLMProvider,
    OCRProvider,
    RerankerProvider,
    VLMProvider,
)
from knowresearch.core.ports.skills import (
    SkillCandidateStorePort,
    SkillContextPort,
    SkillLifecyclePort,
    SkillStorePort,
)
from knowresearch.core.ports.stores import (
    FullTextStorePort,
    ObjectStoragePort,
    VectorStorePort,
)
from knowresearch.core.ports.traces import TraceStorePort
from knowresearch.core.ports.tools import MCPClientPort, ToolPort
from knowresearch.core.ports.feedback import FeedbackStorePort

__all__ = [
    # 模型提供商
    "EmbeddingProvider",
    "RerankerProvider",
    "LLMProvider",
    "VLMProvider",
    "OCRProvider",
    # 存储
    "VectorStorePort",
    "FullTextStorePort",
    "ObjectStoragePort",
    # 文档
    "DocumentStorePort",
    # 记忆
    "SessionStorePort",
    "WorkingMemoryPort",
    "LongTermMemoryPort",
    "MemoryCandidateStorePort",
    "MemoryContextPort",
    # 技能
    "SkillStorePort",
    "SkillCandidateStorePort",
    "SkillContextPort",
    "SkillLifecyclePort",
    # 轨迹
    "TraceStorePort",
    # 工具
    "ToolPort",
    "MCPClientPort",
    "FeedbackStorePort",
]
