"""QA 引擎：把检索 → 重排 → 引用溯源 → 生成串成完整闭环。

这是问答链路的唯一入口，集中处理异常降级、trace_id、上下文预算。
横切关注点（见开发计划书 §3.1）在此统一落地。

支持多轮对话：传入 session_id 时自动加载历史、滑动窗口压缩、保存新消息。
当对话超过滑动窗口上限时，溢出的早期对话会被 LLM 压缩成摘要，
与最近窗口内的对话一起注入 prompt，既控制 token 又保留上下文。
"""

import logging
import time
import uuid
import copy

from knowresearch.core.generation.answer_generator import (
    NO_ANSWER_TEXT,
    Answer,
    AnswerGenerator,
)
from knowresearch.core.observability import metrics
from knowresearch.core.ports import (
    EmbeddingProvider,
    FullTextStorePort,
    MemoryContextPort,
    SessionStorePort,
    SkillContextPort,
    SkillLifecyclePort,
    TraceStorePort,
    VectorStorePort,
)
from knowresearch.core.retrieval.citation import CitationBuilder
from knowresearch.core.retrieval.reranker import Reranker
from knowresearch.core.retrieval.retriever import Retriever
from knowresearch.core.schemas import Chunk, Record

logger = logging.getLogger(__name__)

SERVICE_UNAVAILABLE_TEXT = "服务暂时不可用，请稍后重试。"

CONTEXT_TOKEN_BUDGET = 4000

MAX_HISTORY_MESSAGES = 20

DEFAULT_TTL_SECONDS = 3600

SUMMARY_PROMPT = (
    "请将以下对话历史压缩成一段简洁的摘要，保留关键信息（用户的问题、核心结论、重要约束），"
    "不要逐句复述，控制在 200 字以内。\n"
    "【已有摘要】\n{old_summary}\n\n"
    "【新增对话】\n{overflow}\n\n"
    "请输出合并后的摘要："
)


class QAEngine:
    """问答引擎：RAG 闭环的统一入口。"""

    def __init__(
        self,
        retriever: Retriever,
        reranker: Reranker,
        citation_builder: CitationBuilder,
        answer_generator: AnswerGenerator,
        embedding_provider: EmbeddingProvider,
        context_token_budget: int = CONTEXT_TOKEN_BUDGET,
        session_store: SessionStorePort | None = None,
        trace_store: TraceStorePort | None = None,
        memory_context: MemoryContextPort | None = None,
        skill_context: SkillContextPort | None = None,
        skill_lifecycle: SkillLifecyclePort | None = None,
        tool_orchestrator=None,
        tool_permissions: set[str] | None = None,
        max_history_messages: int = MAX_HISTORY_MESSAGES,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
    ) -> None:
        self.retriever = retriever
        self.reranker = reranker
        self.citation_builder = citation_builder
        self.answer_generator = answer_generator
        self.embedding_provider = embedding_provider
        self.context_token_budget = context_token_budget
        self.session_store = session_store
        self.trace_store = trace_store
        self.memory_context = memory_context
        self.skill_context = skill_context
        self.skill_lifecycle = skill_lifecycle
        self.tool_orchestrator = tool_orchestrator
        self.tool_permissions = tool_permissions
        self.max_history_messages = max_history_messages
        self.ttl_seconds = ttl_seconds

    def answer(
        self,
        question: str,
        session_id: str | None = None,
        user_id: str | None = None,
    ) -> Answer:
        """执行完整问答链路。

        Args:
            question: 用户问题
            session_id: 会话 ID。传入时启用多轮对话（加载历史 + 保存新消息）。
            user_id: 用户 ID，用于后续长期记忆和轨迹隔离。

        任何环节失败都走降级策略，不会把异常抛给调用方。
        """
        trace_id = uuid.uuid4().hex
        started_at = time.perf_counter()
        logger.info("[trace_id=%s] 收到问题: %s", trace_id, question[:100])

        history: list[dict[str, str]] = []
        summary: str = ""
        retrieval_query = question
        memory_context_text = ""
        skill_context_text = ""
        matched_skill_ids: list[str] = []
        tool_context_text = ""
        tool_results: list = []
        candidates: list[Chunk] = []
        top_chunks: list[Chunk] = []
        citations: list = []
        stage_timings: dict[str, float] = {}
        if session_id and self.session_store:
            try:
                history, summary = self._load_history(session_id)
            except Exception as e:
                logger.warning("加载会话历史失败，忽略历史: %s", e)

        if user_id and self.memory_context:
            memory_started = time.perf_counter()
            try:
                retrieval_query = self.memory_context.rewrite_query(user_id, question)
                memory_context_text = self.memory_context.build_context(
                    user_id, question
                )
            except Exception as e:
                logger.warning("加载用户画像失败，忽略画像: %s", e)
            stage_timings["memory_ms"] = self._elapsed_ms(memory_started)

        if user_id and self.skill_context:
            skill_started = time.perf_counter()
            try:
                matched_skills = self.skill_context.match(user_id, question)
                matched_skill_ids = [skill.id for skill in matched_skills]
                skill_context_text = self.skill_context.build_context(
                    user_id, question
                )
            except Exception as e:
                logger.warning("加载技能失败，忽略技能: %s", e)
            stage_timings["skill_ms"] = self._elapsed_ms(skill_started)

        if self.tool_orchestrator:
            tool_started = time.perf_counter()
            try:
                tool_results = self.tool_orchestrator.run(
                    question,
                    permissions=self.tool_permissions,
                )
                tool_context_text = self.tool_orchestrator.build_context(tool_results)
            except Exception as e:
                logger.warning("工具编排失败，忽略工具: %s", e)
            stage_timings["tool_ms"] = self._elapsed_ms(tool_started)

        try:
            retrieve_started = time.perf_counter()
            candidates = self._retrieve(retrieval_query)
            stage_timings["retrieve_ms"] = self._elapsed_ms(retrieve_started)
            if not candidates:
                answer = Answer(text=NO_ANSWER_TEXT, citations=[], trace_id=trace_id)
                self._save_if_session(session_id, question, answer.text, history, summary)
                self._save_trace(
                    trace_id=trace_id,
                    question=question,
                    answer_text=answer.text,
                    status="no_results",
                    session_id=session_id,
                    user_id=user_id,
                    candidates=candidates,
                    top_chunks=top_chunks,
                    citations=citations,
                    retrieval_query=retrieval_query,
                    memory_context_text=memory_context_text,
                    skill_context_text=skill_context_text,
                    matched_skill_ids=matched_skill_ids,
                    tool_context_text=tool_context_text,
                    tool_results=tool_results,
                    stage_timings=stage_timings,
                    started_at=started_at,
                )
                self._record_skill_usage(
                    user_id=user_id,
                    skill_ids=matched_skill_ids,
                    success=False,
                )
                return answer

            rerank_started = time.perf_counter()
            top_chunks = self._rerank(question, candidates)
            top_chunks = self._apply_context_budget(top_chunks)
            stage_timings["rerank_ms"] = self._elapsed_ms(rerank_started)

            citations = self.citation_builder.build(top_chunks)

            generate_started = time.perf_counter()
            answer = self._generate(
                question,
                top_chunks,
                citations,
                history,
                summary,
                trace_id,
                memory_context_text,
                skill_context_text,
                tool_context_text,
            )
            stage_timings["generate_ms"] = self._elapsed_ms(generate_started)
            self._save_if_session(session_id, question, answer.text, history, summary)
            status = "success"
            if answer.text == SERVICE_UNAVAILABLE_TEXT:
                status = "llm_fallback"
            elif answer.text == NO_ANSWER_TEXT:
                status = "no_answer"
            self._save_trace(
                trace_id=trace_id,
                question=question,
                answer_text=answer.text,
                status=status,
                session_id=session_id,
                user_id=user_id,
                candidates=candidates,
                top_chunks=top_chunks,
                citations=citations,
                retrieval_query=retrieval_query,
                memory_context_text=memory_context_text,
                skill_context_text=skill_context_text,
                matched_skill_ids=matched_skill_ids,
                tool_context_text=tool_context_text,
                tool_results=tool_results,
                stage_timings=stage_timings,
                llm_usage=answer.usage,
                started_at=started_at,
            )
            self._record_skill_usage(
                user_id=user_id,
                skill_ids=matched_skill_ids,
                success=status == "success",
            )
            return answer
        except Exception as e:
            logger.exception("[trace_id=%s] 问答链路未捕获异常: %s", trace_id, e)
            answer = Answer(text=SERVICE_UNAVAILABLE_TEXT, citations=[], trace_id=trace_id)
            self._save_if_session(session_id, question, answer.text, history, summary)
            self._save_trace(
                trace_id=trace_id,
                question=question,
                answer_text=answer.text,
                status="error",
                session_id=session_id,
                user_id=user_id,
                candidates=candidates,
                top_chunks=top_chunks,
                citations=citations,
                retrieval_query=retrieval_query,
                memory_context_text=memory_context_text,
                skill_context_text=skill_context_text,
                matched_skill_ids=matched_skill_ids,
                tool_context_text=tool_context_text,
                tool_results=tool_results,
                stage_timings=stage_timings,
                error=str(e),
                started_at=started_at,
            )
            self._record_skill_usage(
                user_id=user_id,
                skill_ids=matched_skill_ids,
                success=False,
            )
            return answer

    def with_skill_context(self, skill_context: SkillContextPort | None) -> "QAEngine":
        """返回一个共享组件、但替换技能上下文的 QAEngine，用于技能回放。"""
        cloned = copy.copy(self)
        cloned.skill_context = skill_context
        return cloned

    def _record_skill_usage(
        self,
        user_id: str | None,
        skill_ids: list[str],
        success: bool,
    ) -> None:
        if not user_id or not self.skill_lifecycle or not skill_ids:
            return
        try:
            for skill_id in skill_ids:
                self.skill_lifecycle.record_usage(
                    skill_id=skill_id,
                    user_id=user_id,
                    success=success,
                )
        except Exception as e:
            logger.warning("记录技能使用失败，忽略: %s", e)

    @staticmethod
    def _elapsed_ms(started_at: float) -> float:
        return round((time.perf_counter() - started_at) * 1000, 2)

    # ================================================================
    # 会话历史
    # ================================================================

    def _load_history(self, session_id: str) -> tuple[list[dict[str, str]], str]:
        """加载会话历史（最近 N 条消息）和摘要。

        摘要存储在 SessionStore 的 summary 字段，是溢出滑动窗口的早期对话压缩结果。
        """
        messages = self.session_store.load_messages(session_id)
        summary = self.session_store.load_summary(session_id)
        return messages, summary

    def _summarize_overflow(
        self, old_summary: str, overflow: list[dict[str, str]]
    ) -> str:
        """用 LLM 把旧摘要 + 溢出的对话压缩成新摘要。

        失败时降级返回旧摘要，不影响主流程。
        """
        overflow_text = "\n".join(
            f"{m['role']}: {m['content']}" for m in overflow
        )
        prompt = SUMMARY_PROMPT.format(
            old_summary=old_summary or "（无）", overflow=overflow_text
        )
        try:
            return self.answer_generator.llm.generate(prompt).strip()
        except Exception as e:
            logger.warning("摘要生成失败，保留旧摘要: %s", e)
            return old_summary

    def _save_if_session(
        self,
        session_id: str | None,
        question: str,
        answer_text: str,
        history: list[dict[str, str]],
        summary: str,
    ) -> None:
        """如果传入了 session_id，保存本轮对话到会话存储。

        当消息数超过 max_history_messages 时，溢出的早期对话被压缩进摘要，
        messages 只保留窗口内的最近消息。
        """
        if not session_id or not self.session_store:
            return
        try:
            new_messages = list(history)
            new_messages.append({"role": "user", "content": question})
            new_messages.append({"role": "assistant", "content": answer_text})

            if len(new_messages) > self.max_history_messages:
                overflow = new_messages[: -self.max_history_messages]
                new_messages = new_messages[-self.max_history_messages :]
                summary = self._summarize_overflow(summary, overflow)

            self.session_store.save_messages(
                session_id,
                new_messages,
                ttl_seconds=self.ttl_seconds,
                summary=summary,
            )
        except Exception as e:
            logger.warning("保存会话历史失败: %s", e)

    def _save_trace(
        self,
        trace_id: str,
        question: str,
        answer_text: str,
        status: str,
        session_id: str | None,
        user_id: str | None,
        candidates: list[Chunk],
        top_chunks: list[Chunk],
        citations: list,
        retrieval_query: str,
        memory_context_text: str,
        skill_context_text: str,
        matched_skill_ids: list[str],
        tool_context_text: str,
        tool_results: list,
        started_at: float,
        stage_timings: dict | None = None,
        llm_usage: dict | None = None,
        error: str = "",
    ) -> None:
        """保存问答轨迹。失败不影响主链路。"""
        duration_ms = round((time.perf_counter() - started_at) * 1000, 2)
        metrics.increment("qa_requests_total")
        metrics.increment(f"qa_status_{status}_total")
        metrics.observe("qa_latency_ms", duration_ms)
        for stage_name, stage_ms in (stage_timings or {}).items():
            metrics.observe(f"qa_{stage_name}", stage_ms)
        total_tokens = (llm_usage or {}).get("total_tokens")
        if total_tokens is not None:
            metrics.observe("llm_tokens", float(total_tokens))

        if not self.trace_store:
            return
        try:
            trace = Record(
                id=trace_id,
                kind="trace",
                content=question,
                source=session_id or "",
                metadata={
                    "trace_id": trace_id,
                    "session_id": session_id or "",
                    "user_id": user_id or "",
                    "status": status,
                    "question": question,
                    "retrieval_query": retrieval_query,
                    "memory_context_used": bool(memory_context_text),
                    "memory_context": memory_context_text,
                    "skill_context_used": bool(skill_context_text),
                    "skill_context": skill_context_text,
                    "matched_skill_ids": matched_skill_ids,
                    "tool_context_used": bool(tool_context_text),
                    "tool_context": tool_context_text,
                    "tool_results": [
                        {
                            "tool_name": result.tool_name,
                            "success": result.success,
                            "error": result.error,
                            "duration_ms": result.duration_ms,
                        }
                        for result in tool_results
                    ],
                    "stage_timings": stage_timings or {},
                    "llm_usage": llm_usage or {},
                    "answer": answer_text,
                    "answer_length": len(answer_text),
                    "candidate_count": len(candidates),
                    "candidate_chunk_ids": [chunk.id for chunk in candidates],
                    "top_chunk_count": len(top_chunks),
                    "top_chunk_ids": [chunk.id for chunk in top_chunks],
                    "citation_count": len(citations),
                    "citations": [
                        citation.model_dump(mode="json")
                        for citation in citations
                    ],
                    "duration_ms": duration_ms,
                    "error": error,
                },
            )
            self.trace_store.save(trace)
        except Exception as e:
            logger.warning("保存 trace 失败，忽略: %s", e)

    # ================================================================
    # 各环节（带降级）
    # ================================================================

    def _retrieve(self, question: str) -> list[Chunk]:
        """混合检索。Embedding 失败时降级为纯全文检索。"""
        try:
            return self.retriever.retrieve(question)
        except Exception as e:
            logger.warning("混合检索失败，降级纯全文: %s", e)
            try:
                ft_results = self.retriever.fulltext_store.search(
                    question, top_k=self.retriever.final_top_k
                )
                return [
                    self.retriever._to_chunk(r)
                    for r, _ in ft_results
                    if self.retriever._to_chunk(r)
                ]
            except Exception as e2:
                logger.warning("纯全文检索也失败: %s", e2)
                return []

    def _rerank(self, question: str, candidates: list[Chunk]) -> list[Chunk]:
        """重排。Reranker 失败时跳过重排，直接返回候选。"""
        try:
            return self.reranker.rerank(question, candidates, top_k=5)
        except Exception as e:
            logger.warning("重排失败，跳过: %s", e)
            return candidates[:5]

    def _apply_context_budget(self, chunks: list[Chunk]) -> list[Chunk]:
        """上下文预算：按字符数估算，超出 budget 时从尾部（相关性最低）丢弃。

        chunks 已按相关性降序排列，所以从后往前丢。
        字符数 / 2 作为 token 数的粗略估算（中英文混合）。
        """
        total = sum(len(c.content) for c in chunks)
        if total <= self.context_token_budget * 2:
            return chunks

        kept: list[Chunk] = []
        used = 0
        for chunk in chunks:
            size = len(chunk.content)
            if used + size > self.context_token_budget * 2:
                break
            kept.append(chunk)
            used += size
        return kept

    def _generate(
        self,
        question: str,
        chunks: list[Chunk],
        citations: list,
        history: list[dict[str, str]],
        summary: str,
        trace_id: str,
        memory_context_text: str = "",
        skill_context_text: str = "",
        tool_context_text: str = "",
    ) -> Answer:
        """生成答案。LLM 失败时返回兜底文案。"""
        try:
            answer = self.answer_generator.generate(
                question,
                chunks,
                citations,
                history=history,
                summary=summary,
                memory_context=memory_context_text,
                skill_context=skill_context_text,
                tool_context=tool_context_text,
            )
            answer.trace_id = trace_id
            if not answer.text.strip():
                logger.warning("LLM 返回空内容，返回兜底文案")
                return Answer(text=NO_ANSWER_TEXT, citations=[], trace_id=trace_id)
            return answer
        except Exception as e:
            logger.warning("LLM 生成失败，返回兜底文案: %s", e)
            return Answer(text=SERVICE_UNAVAILABLE_TEXT, citations=[], trace_id=trace_id)
