"""AgentEngine：在现有 QAEngine 之上加一层「复杂度路由 + ReAct」编排。

设计要点：
- 对外返回与 QAEngine.answer 相同的 Answer 契约，API/前端/评测无需改动。
- 简单问题走原 QAEngine（快路径），只有复杂问题才进 ReAct。
- ReAct 任何环节失败都降级回 QAEngine，慢路径永远不打崩请求。
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any

from knowresearch.agent.react_agent import AgentRunResult, ReActAgent
from knowresearch.agent.router import Intent, IntentRouter
from knowresearch.core.generation.answer_generator import Answer
from knowresearch.core.schemas import Chunk, Record

logger = logging.getLogger(__name__)


class AgentEngine:
    """带 ReAct 慢路径的问答引擎。"""

    def __init__(
        self,
        qa_engine: Any,
        react_agent: ReActAgent,
        router: Any | None = None,
        answer_generator: Any | None = None,
        citation_builder: Any | None = None,
        enabled: bool = True,
        router_enabled: bool = True,
    ) -> None:
        self.qa_engine = qa_engine
        self.react_agent = react_agent
        self.router = router or IntentRouter()
        self.answer_generator = answer_generator or qa_engine.answer_generator
        self.citation_builder = citation_builder or qa_engine.citation_builder
        self.enabled = enabled
        self.router_enabled = router_enabled

    def answer(
        self,
        question: str,
        session_id: str | None = None,
        user_id: str | None = None,
    ) -> Answer:
        if not self.enabled:
            return self.qa_engine.answer(question, session_id, user_id)

        try:
            decision = self.router.route(question)
            intent = getattr(decision, "intent", None)
            sub_queries = getattr(decision, "sub_queries", [])

            if intent == Intent.CHITCHAT:
                return self._direct_answer(question, session_id, user_id)

            if not self.router_enabled:
                return (
                    self.qa_engine.answer(question, session_id, user_id)
                    if not decision.complex
                    else self._react_answer(question, session_id, user_id)
                )

            if intent == Intent.ASSIST:
                if self._has_tool("bibtex_generator"):
                    return self._react_answer(
                        question,
                        session_id,
                        user_id,
                        plan=["生成所需的 BibTeX 引用条目"],
                    )
                return self.qa_engine.answer(question, session_id, user_id)

            if intent in {Intent.COMPARISON, Intent.SURVEY, Intent.MEMORY}:
                return self._react_answer(
                    question,
                    session_id,
                    user_id,
                    plan=sub_queries,
                )

            if not decision.complex:
                return self.qa_engine.answer(question, session_id, user_id)

            return self._react_answer(
                question, session_id, user_id, plan=sub_queries
            )
        except Exception as e:
            logger.warning("ReAct 慢路径失败，降级快路径: %s", e)
            return self.qa_engine.answer(question, session_id, user_id)

    # ------------------------------------------------------------------
    # ReAct 慢路径
    # ------------------------------------------------------------------

    def _react_answer(
        self,
        question: str,
        session_id: str | None,
        user_id: str | None,
        plan: list[str] | None = None,
    ) -> Answer:
        trace_id = uuid.uuid4().hex
        started_at = time.perf_counter()

        try:
            if plan:
                result = self.react_agent.run(
                    goal=question, task_id=trace_id, plan=plan
                )
            else:
                result = self.react_agent.run(goal=question, task_id=trace_id)
        except TypeError:
            result = self.react_agent.run(goal=question, task_id=trace_id)
        chunks = self._chunks_from_evidence(result.evidence)

        if not chunks:
            if result.final_answer:
                answer = Answer(text=result.final_answer, citations=[], trace_id=trace_id)
                self._save_trace(question, answer, result, session_id, user_id, started_at)
                return answer
            # 既没有证据也没有最终答案：降级回快路径
            return self.qa_engine.answer(question, session_id, user_id)

        citations = self.citation_builder.build(chunks)
        history, summary = self._load_history(session_id)
        memory_context = self._build_memory_context(user_id, question)
        figure_context = self._build_figure_context(citations, question)

        try:
            answer = self.answer_generator.generate(
                question,
                chunks,
                citations,
                history=history,
                summary=summary,
                memory_context=memory_context,
                figure_context=figure_context,
            )
        except Exception as e:
            logger.warning("ReAct 最终生成失败，降级快路径: %s", e)
            return self.qa_engine.answer(question, session_id, user_id)

        answer.trace_id = trace_id
        self._save_session(session_id, question, answer.text, history, summary)
        self._save_trace(question, answer, result, session_id, user_id, started_at)
        return answer

    # ------------------------------------------------------------------
    # 闲聊直答：不检索，直接进入对话
    # ------------------------------------------------------------------

    def _direct_answer(
        self,
        question: str,
        session_id: str | None,
        user_id: str | None,
    ) -> Answer:
        trace_id = uuid.uuid4().hex
        history, summary = self._load_history(session_id)
        memory_context = self._build_memory_context(user_id, question)

        try:
            prompt = (
                "你是知研 KnowResearch，面向研究生的科研知识库助手。"
                "请用简洁、自然的中文回应用户的这句话，不要调用检索。\n"
                f"用户：{question}"
            )
            text = self.answer_generator.llm.generate(prompt).strip()
        except Exception:
            text = "你好，我是知研，可以帮你检索论文、做文献调研、管理阅读记忆。"

        answer = Answer(text=text, citations=[], trace_id=trace_id)
        self._save_session(session_id, question, answer.text, history, summary)
        self._save_trace(
            question,
            answer,
            AgentRunResult(
                task_id=trace_id,
                goal=question,
                final_answer=answer.text,
                status="done",
                stop_reason="chitchat",
            ),
            session_id,
            user_id,
            time.perf_counter(),
        )
        return answer

    @staticmethod
    def _chunks_from_evidence(evidence: list[dict[str, Any]]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for index, item in enumerate(evidence, start=1):
            chunk_id = str(item.get("chunk_id") or f"agent-evidence-{index}")
            chunks.append(
                Chunk(
                    id=chunk_id,
                    content=str(item.get("content") or ""),
                    metadata={
                        "document_id": item.get("document_id", ""),
                        "section_id": item.get("section_id", ""),
                        "page": item.get("page", 0),
                    },
                )
            )
        return chunks

    def _load_history(
        self, session_id: str | None
    ) -> tuple[list[dict[str, str]], str]:
        store = getattr(self.qa_engine, "session_store", None)
        if not session_id or store is None:
            return [], ""
        try:
            return store.load_messages(session_id), store.load_summary(session_id)
        except Exception as e:
            logger.warning("加载会话历史失败，忽略: %s", e)
            return [], ""

    def _has_tool(self, tool_name: str) -> bool:
        registry = getattr(self.react_agent, "registry", None)
        return registry is not None and registry.get(tool_name) is not None

    def _build_memory_context(self, user_id: str | None, question: str) -> str:
        memory = getattr(self.qa_engine, "memory_context", None)
        if not user_id or memory is None:
            return ""
        try:
            return memory.build_context(user_id, question)
        except Exception:
            return ""

    def _build_figure_context(self, citations, question: str) -> str:
        reader = getattr(self.qa_engine, "figure_reader", None)
        if reader is None:
            return ""
        try:
            return reader.read_citations(citations, question)
        except Exception:
            return ""

    def _save_session(
        self,
        session_id: str | None,
        question: str,
        answer_text: str,
        history: list[dict[str, str]],
        summary: str,
    ) -> None:
        store = getattr(self.qa_engine, "session_store", None)
        if not session_id or store is None:
            return
        try:
            messages = list(history)
            messages.append({"role": "user", "content": question})
            messages.append({"role": "assistant", "content": answer_text})
            window = int(getattr(self.qa_engine, "max_history_messages", 20))
            if len(messages) > window:
                messages = messages[-window:]
            store.save_messages(
                session_id,
                messages,
                ttl_seconds=int(getattr(self.qa_engine, "ttl_seconds", 3600)),
                summary=summary,
            )
        except Exception as e:
            logger.warning("保存会话历史失败: %s", e)

    def _save_trace(
        self,
        question: str,
        answer: Answer,
        result: AgentRunResult,
        session_id: str | None,
        user_id: str | None,
        started_at: float,
    ) -> None:
        store = getattr(self.qa_engine, "trace_store", None)
        if store is None:
            return
        try:
            duration_ms = round((time.perf_counter() - started_at) * 1000, 2)
            trace = Record(
                id=answer.trace_id,
                kind="trace",
                content=question,
                source=session_id or "",
                metadata={
                    "trace_id": answer.trace_id,
                    "session_id": session_id or "",
                    "user_id": user_id or "",
                    "status": (
                        "chitchat"
                        if result.stop_reason == "chitchat"
                        else ("agent_success" if result.evidence else "agent_no_evidence")
                    ),
                    "mode": "chitchat" if result.stop_reason == "chitchat" else "react",
                    "question": question,
                    "answer": answer.text,
                    "stop_reason": result.stop_reason,
                    "agent_plan": result.plan,
                    "agent_llm_calls": result.llm_calls,
                    "agent_step_count": len(result.steps),
                    "agent_reflections": result.reflections,
                    "agent_steps": [
                        {
                            "index": step.index,
                            "thought": step.thought,
                            "action": step.action,
                            "action_input": step.action_input,
                            "observation": step.observation,
                            "success": step.success,
                        }
                        for step in result.steps
                    ],
                    "evidence_chunk_ids": [c.id for c in self._chunks_from_evidence(result.evidence)],
                    "citation_count": len(answer.citations),
                    "duration_ms": duration_ms,
                },
            )
            store.save(trace)
        except Exception as e:
            logger.warning("保存 ReAct trace 失败，忽略: %s", e)
