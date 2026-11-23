"""ReAct 提示词：把工具定义、目标和历史步骤拼成 LLM messages。

采用「结构化 JSON 输出」而不是让模型自由写 Thought/Action 文本，
这样解析更稳、也不依赖模型的 function calling 能力。
"""

from __future__ import annotations

import json

from knowresearch.core.schemas import AgentStep, ToolDefinition


REACT_SYSTEM_PROMPT = """你是知研 KnowResearch 的 ReAct 智能体，负责通过多步工具调用回答复杂问题。

你可以使用以下工具（每次只能调用一个）：
{tools}

工作方式（ReAct）：
1. Thought：分析当前已知信息、还缺什么。
2. Action：选择一个工具并给出参数。
3. 看到 Observation 后继续下一轮，直到信息足够。
4. 信息足够时给出 Final Answer。

输出格式（严格遵守，只输出一个 JSON 对象，不要输出 JSON 以外的任何文字）：
- 需要调用工具时：
{{"thought": "你的推理", "action": "工具名", "action_input": {{"参数名": "参数值"}}}}
- 信息足够、可以作答时：
{{"thought": "你的推理", "final_answer": "最终答案"}}

规则：
- 每次只输出一个动作，不要一次调用多个工具。
- 不要重复调用完全相同的工具和参数；上一轮结果不够时请改写查询词。
- 证据不足时也要基于已有观察给出最终答案，不要编造。
"""


def render_tools(tools: list[ToolDefinition]) -> str:
    """把工具定义渲染成提示词里的工具清单。"""
    if not tools:
        return "（无可用工具）"
    lines: list[str] = []
    for tool in tools:
        schema = json.dumps(tool.input_schema, ensure_ascii=False)
        lines.append(f"- {tool.name}：{tool.description}\n  参数 schema：{schema}")
    return "\n".join(lines)


def render_scratchpad(
    steps: list[AgentStep],
    reflections: list[str] | None = None,
) -> str:
    """把已执行的步骤和反思渲染成 ReAct 的 scratchpad。"""
    blocks: list[str] = []

    for index, feedback in enumerate(reflections or [], start=1):
        blocks.append(f"【反思 {index}】\n{feedback}")

    if not steps and not (reflections or []):
        return "（还没有执行任何步骤）"

    for step in steps:
        lines = [f"第 {step.index} 步"]
        if step.thought:
            lines.append(f"Thought: {step.thought}")
        if step.action:
            lines.append(f"Action: {step.action}")
        if step.action_input:
            lines.append(
                f"Action Input: {json.dumps(step.action_input, ensure_ascii=False)}"
            )
        lines.append(f"Observation: {step.observation}")
        blocks.append("\n".join(lines))

    if not blocks:
        return "（还没有执行任何步骤）"
    return "\n\n".join(blocks)


def build_react_messages(
    goal: str,
    tools: list[ToolDefinition],
    scratchpad: str,
    plan: list[str] | None = None,
) -> list[dict[str, str]]:
    """构建一次 ReAct 决策所需的 messages。"""
    system_prompt = REACT_SYSTEM_PROMPT.format(tools=render_tools(tools))
    plan_text = ""
    if plan:
        plan_lines = "\n".join(f"{i}. {item}" for i, item in enumerate(plan, 1))
        plan_text = f"\n\n【执行计划（按顺序覆盖）】\n{plan_lines}"
    user_prompt = (
        f"【目标问题】\n{goal}{plan_text}\n\n"
        f"【已执行步骤】\n{scratchpad}\n\n"
        "请给出下一步的 JSON。"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
