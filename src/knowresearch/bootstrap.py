"""应用组装入口：把适配器和核心编排对象装配到一起。

业务代码仍然只依赖端口；这里只负责“选择哪个实现”。
默认生产装配使用：
- 本地 BGE Embedding / Reranker
- DeepSeek LLM
- PostgreSQL 文档 / 轨迹 / 向量存储
- Redis 会话记忆
- 内存 BM25 全文索引（启动时从 pgvector 表重建）
- 本地文件对象存储

P0 阶段仍保留 SQLite + 内存索引的轻量装配函数。
"""

from __future__ import annotations

from pathlib import Path

from knowresearch.adapters import (
    BM25FullTextStore,
    CachedEmbeddingProvider,
    DeepSeekLLM,
    DemoLLM,
    InMemoryVectorStore,
    LocalBGEZhEmbedding,
    LocalBGEReranker,
    LocalObjectStorage,
    MockEmbedding,
    MockReranker,
    PostgresDocumentStore,
    PostgresFeedbackStore,
    PostgresLongTermMemory,
    PostgresMemoryCandidateStore,
    PostgresSkillCandidateStore,
    PostgresSkillStore,
    PostgresTraceStore,
    PostgresVectorStore,
    RedisSessionStore,
    SQLiteDocumentStore,
)
from knowresearch.config import settings
from knowresearch.core.generation.answer_generator import AnswerGenerator
from knowresearch.core.ports import (
    DocumentStorePort,
    EmbeddingProvider,
    FullTextStorePort,
    LLMProvider,
    MemoryContextPort,
    ObjectStoragePort,
    RerankerProvider,
    SessionStorePort,
    SkillContextPort,
    SkillLifecyclePort,
    TraceStorePort,
    VectorStorePort,
)
from knowresearch.core.qa_engine import QAEngine
from knowresearch.core.retrieval.citation import CitationBuilder
from knowresearch.core.retrieval.reranker import Reranker
from knowresearch.core.retrieval.retriever import Retriever
from knowresearch.ingestion.pipeline import IngestPipeline
from knowresearch.memory import (
    ProfileExtractor,
    ProfileMemoryService,
    ProfileReviewService,
)
from knowresearch.skills import (
    SkillContextService,
    SkillExtractor,
    SkillLifecycleService,
    SkillReplayService,
    SkillReviewService,
)
from knowresearch.feedback import FeedbackService
from knowresearch.tools import (
    LocalKnowledgeBaseTool,
    StdioMCPClient,
    StreamableHTTPMCPClient,
    ToolExecutor,
    ToolOrchestrator,
    ToolRegistry,
    register_mcp_tools,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_project_path(path_value: str | Path) -> Path:
    """把配置里的相对路径解析为基于项目根目录的绝对路径。"""
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def sqlite_db_path(database_url: str | None = None) -> Path:
    """从 sqlite:// URL 中解析出文件路径，供 SQLite 适配器使用。"""
    url = database_url or settings.storage.database_url
    prefix = "sqlite:///"
    if url.startswith(prefix):
        return resolve_project_path(url[len(prefix) :])
    return Path(url)


def build_embedding() -> EmbeddingProvider:
    if settings.app.demo_mode:
        return CachedEmbeddingProvider(MockEmbedding())
    return CachedEmbeddingProvider(
        LocalBGEZhEmbedding(
            model_name=settings.embedding.model_name,
            model_dir=resolve_project_path(settings.embedding.model_dir),
        )
    )


def build_reranker() -> RerankerProvider:
    if settings.app.demo_mode:
        return MockReranker()
    return LocalBGEReranker(
        model_name=settings.reranker.model_name,
        model_dir=resolve_project_path(settings.reranker.model_dir),
    )


def build_llm() -> LLMProvider:
    if settings.app.demo_mode:
        return DemoLLM()
    return DeepSeekLLM()


def build_document_store(db_path: str | None = None) -> DocumentStorePort:
    path = Path(db_path) if db_path else sqlite_db_path()
    return SQLiteDocumentStore(str(path))


def build_object_storage(base_dir: str | None = None) -> ObjectStoragePort:
    directory = resolve_project_path(base_dir or settings.storage.file_storage_dir)
    return LocalObjectStorage(str(directory))


def build_production_document_store() -> PostgresDocumentStore:
    return PostgresDocumentStore(settings.storage.postgres_url)


def build_production_trace_store() -> PostgresTraceStore:
    return PostgresTraceStore(settings.storage.postgres_url)


def build_production_long_term_memory() -> PostgresLongTermMemory:
    return PostgresLongTermMemory(settings.storage.postgres_url)


def build_production_memory_candidate_store() -> PostgresMemoryCandidateStore:
    return PostgresMemoryCandidateStore(settings.storage.postgres_url)


def build_production_skill_store() -> PostgresSkillStore:
    return PostgresSkillStore(settings.storage.postgres_url)


def build_production_feedback_store() -> PostgresFeedbackStore:
    return PostgresFeedbackStore(settings.storage.postgres_url)


def build_production_feedback_service() -> FeedbackService:
    return FeedbackService(
        store=build_production_feedback_store(),
        skill_lifecycle=build_skill_lifecycle_service(),
    )


def build_production_skill_candidate_store() -> PostgresSkillCandidateStore:
    return PostgresSkillCandidateStore(settings.storage.postgres_url)


def build_profile_extractor(
    llm_provider: LLMProvider | None = None,
    trace_store: TraceStorePort | None = None,
    candidate_store: PostgresMemoryCandidateStore | None = None,
) -> ProfileExtractor:
    return ProfileExtractor(
        llm=llm_provider or build_llm(),
        trace_store=trace_store or build_production_trace_store(),
        candidate_store=candidate_store or build_production_memory_candidate_store(),
    )


def build_profile_review_service(
    candidate_store: PostgresMemoryCandidateStore | None = None,
    long_term_memory: PostgresLongTermMemory | None = None,
) -> ProfileReviewService:
    return ProfileReviewService(
        candidate_store=candidate_store or build_production_memory_candidate_store(),
        long_term_memory=long_term_memory or build_production_long_term_memory(),
    )


def build_profile_memory_service(
    long_term_memory: PostgresLongTermMemory | None = None,
) -> ProfileMemoryService:
    return ProfileMemoryService(
        long_term_memory=long_term_memory or build_production_long_term_memory()
    )


def build_skill_extractor(
    llm_provider: LLMProvider | None = None,
    trace_store: TraceStorePort | None = None,
    candidate_store: PostgresSkillCandidateStore | None = None,
) -> SkillExtractor:
    return SkillExtractor(
        llm=llm_provider or build_llm(),
        trace_store=trace_store or build_production_trace_store(),
        candidate_store=candidate_store or build_production_skill_candidate_store(),
    )


def build_skill_review_service(
    candidate_store: PostgresSkillCandidateStore | None = None,
    skill_store: PostgresSkillStore | None = None,
) -> SkillReviewService:
    return SkillReviewService(
        candidate_store=candidate_store or build_production_skill_candidate_store(),
        skill_store=skill_store or build_production_skill_store(),
    )


def build_skill_context_service(
    skill_store: PostgresSkillStore | None = None,
) -> SkillContextService:
    return SkillContextService(
        skill_store=skill_store or build_production_skill_store(),
    )


def build_skill_lifecycle_service(
    skill_store: PostgresSkillStore | None = None,
) -> SkillLifecycleService:
    return SkillLifecycleService(
        skill_store=skill_store or build_production_skill_store(),
    )


def build_skill_replay_service(
    qa_engine: QAEngine | None = None,
    trace_store: TraceStorePort | None = None,
    candidate_store: PostgresSkillCandidateStore | None = None,
) -> SkillReplayService:
    return SkillReplayService(
        qa_engine=qa_engine or build_production_qa_engine(),
        trace_store=trace_store or build_production_trace_store(),
        candidate_store=candidate_store or build_production_skill_candidate_store(),
    )


def build_stdio_mcp_client(
    command: str,
    args: list[str] | None = None,
    env: dict[str, str] | None = None,
    cwd: str | None = None,
    request_timeout_seconds: float = 30.0,
) -> StdioMCPClient:
    return StdioMCPClient(
        command=command,
        args=args,
        env=env,
        cwd=cwd,
        request_timeout_seconds=request_timeout_seconds,
    )


def build_mcp_tool_registry(
    client: StdioMCPClient,
    permissions: list[str] | None = None,
) -> ToolRegistry:
    registry = ToolRegistry()
    register_mcp_tools(registry, client, permissions=permissions)
    return registry


def build_streamable_http_mcp_client(
    url: str,
    headers: dict[str, str] | None = None,
    request_timeout_seconds: float = 30.0,
) -> StreamableHTTPMCPClient:
    return StreamableHTTPMCPClient(
        url=url,
        headers=headers,
        request_timeout_seconds=request_timeout_seconds,
    )


def build_production_vector_store(
    dimension: int | None = None,
) -> PostgresVectorStore:
    if dimension is None:
        dimension = 384 if settings.app.demo_mode else settings.embedding.dimension
    table_name = "chunk_vectors_demo" if settings.app.demo_mode else "chunk_vectors"
    return PostgresVectorStore(
        settings.storage.postgres_url,
        dimension=dimension,
        table_name=table_name,
    )


def build_redis_session_store() -> RedisSessionStore:
    return RedisSessionStore(settings.storage.redis_url)


def build_production_fulltext_store(
    vector_store: VectorStorePort | None = None,
) -> FullTextStorePort:
    """BM25 是内存索引；启动时从 pgvector 表里的 chunk 记录重建。"""
    store = BM25FullTextStore()
    list_all = getattr(vector_store, "list_all", None)
    if callable(list_all):
        records = list_all()
        if records:
            store.add(records)
    return store


def build_ingest_pipeline(
    document_store: DocumentStorePort,
    vector_store: VectorStorePort,
    fulltext_store: FullTextStorePort,
    object_storage: ObjectStoragePort,
    embedding_provider: EmbeddingProvider | None = None,
) -> IngestPipeline:
    embedding = embedding_provider or build_embedding()
    return IngestPipeline(
        document_store=document_store,
        vector_store=vector_store,
        fulltext_store=fulltext_store,
        object_storage=object_storage,
        embedding_provider=embedding,
    )


def build_qa_engine(
    vector_store: VectorStorePort,
    fulltext_store: FullTextStorePort,
    document_store: DocumentStorePort,
    embedding_provider: EmbeddingProvider | None = None,
    reranker_provider: RerankerProvider | None = None,
    llm_provider: LLMProvider | None = None,
    session_store: SessionStorePort | None = None,
    trace_store: TraceStorePort | None = None,
    memory_context: MemoryContextPort | None = None,
    skill_context: SkillContextPort | None = None,
    skill_lifecycle: SkillLifecyclePort | None = None,
    tool_orchestrator: ToolOrchestrator | None = None,
    tool_permissions: set[str] | None = None,
    enable_local_tools: bool = False,
) -> QAEngine:
    embedding = embedding_provider or build_embedding()
    reranker = reranker_provider or build_reranker()
    llm = llm_provider or build_llm()

    retriever = Retriever(vector_store, fulltext_store, embedding)
    tool_registry = None
    if tool_orchestrator is None and enable_local_tools:
        tool_registry = ToolRegistry()
        tool_registry.register(LocalKnowledgeBaseTool(retriever))
        tool_orchestrator = ToolOrchestrator(
            tool_registry,
            ToolExecutor(tool_registry),
        )
    elif tool_orchestrator is not None:
        tool_registry = getattr(tool_orchestrator, "registry", None)
    citation_builder = CitationBuilder(document_store)
    answer_generator = AnswerGenerator(llm)

    engine = QAEngine(
        retriever=retriever,
        reranker=Reranker(reranker),
        citation_builder=citation_builder,
        answer_generator=answer_generator,
        embedding_provider=embedding,
        session_store=session_store,
        trace_store=trace_store,
        memory_context=memory_context,
        skill_context=skill_context,
        skill_lifecycle=skill_lifecycle,
        tool_orchestrator=tool_orchestrator,
        tool_permissions=tool_permissions,
    )
    engine.tool_registry = tool_registry
    return engine


def build_production_ingest_pipeline(
    document_store: DocumentStorePort | None = None,
    vector_store: VectorStorePort | None = None,
    fulltext_store: FullTextStorePort | None = None,
    object_storage: ObjectStoragePort | None = None,
    embedding_provider: EmbeddingProvider | None = None,
) -> IngestPipeline:
    doc_store = document_store or build_production_document_store()
    vec_store = vector_store or build_production_vector_store()
    ft_store = fulltext_store or build_production_fulltext_store(vec_store)
    obj_storage = object_storage or build_object_storage()
    return build_ingest_pipeline(
        document_store=doc_store,
        vector_store=vec_store,
        fulltext_store=ft_store,
        object_storage=obj_storage,
        embedding_provider=embedding_provider,
    )


def build_production_qa_engine(
    document_store: DocumentStorePort | None = None,
    vector_store: VectorStorePort | None = None,
    fulltext_store: FullTextStorePort | None = None,
    embedding_provider: EmbeddingProvider | None = None,
    reranker_provider: RerankerProvider | None = None,
    llm_provider: LLMProvider | None = None,
    session_store: SessionStorePort | None = None,
    trace_store: TraceStorePort | None = None,
    memory_context: MemoryContextPort | None = None,
    skill_context: SkillContextPort | None = None,
    skill_lifecycle: SkillLifecyclePort | None = None,
    tool_orchestrator: ToolOrchestrator | None = None,
    tool_permissions: set[str] | None = None,
    enable_local_tools: bool = True,
) -> QAEngine:
    doc_store = document_store or build_production_document_store()
    vec_store = vector_store or build_production_vector_store()
    ft_store = fulltext_store or build_production_fulltext_store(vec_store)
    sess_store = session_store if session_store is not None else build_redis_session_store()
    trace_store = trace_store if trace_store is not None else build_production_trace_store()
    if memory_context is None:
        long_term_memory = build_production_long_term_memory()
        memory_context = ProfileMemoryService(long_term_memory)
    if skill_context is None:
        skill_context = build_skill_context_service()
    if skill_lifecycle is None:
        skill_lifecycle = build_skill_lifecycle_service()
    return build_qa_engine(
        vector_store=vec_store,
        fulltext_store=ft_store,
        document_store=doc_store,
        embedding_provider=embedding_provider,
        reranker_provider=reranker_provider,
        llm_provider=llm_provider,
        session_store=sess_store,
        trace_store=trace_store,
        memory_context=memory_context,
        skill_context=skill_context,
        skill_lifecycle=skill_lifecycle,
        tool_orchestrator=tool_orchestrator,
        tool_permissions=tool_permissions,
        enable_local_tools=enable_local_tools,
    )
