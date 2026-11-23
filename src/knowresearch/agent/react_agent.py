"""ReAct 主循环：Thought → Action → Observation，有界执行。

职责边界：
- 本类只负责「循环控制 + 状态维护 + 观察压缩」，不负责生成最终答案。
- Action 通过 ToolExecutor 执行（复用权限/参数校验/超时/异常兜底）。
- 状态通过 WorkingMemoryPort 落库（复用 SQLiteWorkingMemory）。
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any

from knowresearch.agent.reflector import ReflectorPort
from knowresearch.core.ports import ActionPlannerPort, WorkingMemoryPort
from knowresearch.core.schemas import AgentState, AgentStep
from knowresearch.tools.executor import ToolExecutor
from knowresearch.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


@dataclass
class AgentRunResult:
    """一次 ReAct 运行的产出。"""

    task_id: str
    goal: str
    plan: list[str] = field(default_factory=list)
    steps: list[AgentStep] = field(default_factory=list)
    reflections: list[str] = field(default_factory=list)
    evidence: list[dict[str, Any]] = field(default_factory=list)
    final_answer: str = ""
    status: str = "done"
    stop_reason: str = ""
    llm_calls: int = 0


class ReActAgent:
    """有界 ReAct 循环。"""

    def __init__(
        self,
        planner: ActionPlannerPort,
        registry: ToolRegistry,
        executor: ToolExecutor,
        working_memory: WorkingMemoryPort | None = None,
        max_steps: int = 5,
        max_llm_calls: int = 8,
        observation_max_chars: int = 300,
        permissions: set[str] | None = None,
        reflector: ReflectorPort | None = None,
    ) -> None:
        self.planner = planner
        self.registry = registry
        self.executor = executor
        self.working_memory = working_memory
        self.max_steps = max(1, max_steps)
        self.max_llm_calls = max(1, max_llm_calls)
        self.observation_max_chars = max(40, observation_max_chars)
        self.permissions = permissions
        self.reflector = reflector

    def run(
        self,
        goal: str,
        task_id: str | None = None,
        plan: list[str] | None = None,
    ) -> AgentRunResult:
        state = AgentState(
            task_id=task_id or uuid.uuid4().hex,
            goal=goal,
            plan=list(plan or []),
        )
        state.status = "running"
        self._persist(state)

        tools = self.registry.list_tools(self.permissions)
        tool_defs = [tool.definition for tool in tools]
        seen_actions: set[str] = set()
        seen_evidence: set[str] = set()
        llm_calls = 0
        stop_reason = ""
        final_answer = ""

        for index in range(1, self.max_steps + 1):
            if llm_calls >= self.max_llm_calls:
                stop_reason = "max_llm_calls"
                break

            scratchpad = self._render_scratchpad(
                state.steps, state.reflections
            )
            decision = self.planner.propose(
                goal, tool_defs, scratchpad, plan=state.plan
            )
            llm_calls += 1
            if decision is None:
                stop_reason = "parse_error"
                break
            if decision.is_final:
                final_answer = decision.final_answer
                stop_reason = "final_answer"
                break

            action_key = self._action_key(decision.action, decision.action_input)
            if action_key in seen_actions:
                stop_reason = "duplicate_action"
                break
            seen_actions.add(action_key)

            result = self.executor.execute(
                decision.action,
                decision.action_input,
                permissions=self.permissions,
            )
            observation = self._format_observation(result.output, result.success)
            reflection_text = ""
            if self.reflector is not None:
                reflection = self.reflector.evaluate(
                    goal,
                    observation,
                    decision.action,
                    decision.action_input,
                )
                if not reflection.sufficient:
                    reflection_text = reflection.feedback
                    state.reflections.append(reflection.feedback)
            step = AgentStep(
                index=index,
                thought=decision.thought,
                action=decision.action,
                action_input=decision.action_input,
                observation=(
                    observation
                    if not reflection_text
                    else f"{observation}\n[反思] {reflection_text}"
                ),
                success=result.success,
            )
            state.steps.append(step)
            self._collect_evidence(result.output, state.evidence, seen_evidence)
            self._persist(state)
        else:
            stop_reason = "max_steps"

        state.final_answer = final_answer
        state.status = "done" if (final_answer or state.evidence) else "failed"
        state.stop_reason = stop_reason
        self._persist(state)

        return AgentRunResult(
            task_id=state.task_id,
            goal=goal,
            plan=list(state.plan),
            steps=state.steps,
            reflections=list(state.reflections),
            evidence=state.evidence,
            final_answer=final_answer,
            status=state.status,
            stop_reason=stop_reason,
            llm_calls=llm_calls,
        )

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------

    def _persist(self, state: AgentState) -> None:
        if self.working_memory is None:
            return
        try:
            self.working_memory.save_state(
                state.task_id, state.model_dump(mode="json")
            )
        except Exception as e:  # 工作记忆失败不影响循环
            logger.warning("保存工作记忆失败，忽略: %s", e)

    @staticmethod
    def _render_scratchpad(
        steps: list[AgentStep],
        reflections: list[str] | None = None,
    ) -> str:
        from knowresearch.agent.prompt import render_scratchpad

        return render_scratchpad(steps, reflections)

    @staticmethod
    def _action_key(action: str, action_input: dict[str, Any]) -> str:
        return f"{action}:{json.dumps(action_input, ensure_ascii=False, sort_keys=True)}"

    def _format_observation(self, output: Any, success: bool) -> str:
        if not success:
            return "（该步骤执行失败）"
        if output is None:
            return "（无返回）"
        if isinstance(output, str):
            text = output
        elif isinstance(output, list):
            lines: list[str] = []
            for item in output:
                if isinstance(item, dict):
                    if item.get("chunk_id"):
                        chunk_id = item.get("chunk_id", "")
                        document_id = item.get("document_id", "")
                        page = item.get("page", "")
                        content = str(item.get("content", "")).replace("\n", " ")
                        lines.append(
                            f"[{chunk_id}] doc={document_id} p={page}: {content[:200]}"
                        )
                    elif item.get("title"):
                        title = str(item.get("title", "")).replace("\n", " ")
                        summary = str(item.get("summary") or item.get("content") or "")
                        lines.append(f"[{title}]: {summary[:200]}")
                    elif item.get("content"):
                        lines.append(f"[{item.get('kind', 'memory')}]: {item.get('content')[:200]}")
                    else:
                        lines.append(json.dumps(item, ensure_ascii=False))
                else:
                    lines.append(str(item))
            text = "\n".join(lines)
        else:
            text = json.dumps(output, ensure_ascii=False)

        if len(text) > self.observation_max_chars:
            text = text[: self.observation_max_chars] + "..."
        return text

    @staticmethod
    def _collect_evidence(
        output: Any,
        evidence: list[dict[str, Any]],
        seen: set[str],
    ) -> None:
        if not isinstance(output, list):
            return
        for item in output:
            if not isinstance(item, dict):
                continue
            chunk_id = item.get("chunk_id")
            if not chunk_id or chunk_id in seen:
                continue
            seen.add(chunk_id)
            evidence.append(item)
