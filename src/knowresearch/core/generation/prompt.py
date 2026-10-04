"""RAG Prompt 模板。

设计原则：
- system prompt 明确角色 + 引用规则
- user prompt 把上下文 chunks 编号列出，引导模型用 [1] [2] 标注
- 支持多轮对话：历史消息插在 system 和当前 user 之间
- 上下文过长时截断（默认每个 chunk 最多 500 字，总长度受 max_tokens 约束）
"""

from knowresearch.core.retrieval.citation import Citation

SYSTEM_PROMPT = (
    "你是知研 KnowResearch，一个专注学术论文的知识助手。\n"
    "你只能基于用户提供的参考资料回答问题，严禁编造资料中没有的内容。\n"
    "回答时，每个关键论点必须标注引用来源，格式为 [n]，其中 n 是参考资料的编号。\n"
    "如果参考资料中没有相关信息，请直接回答「未找到相关内容」，不要猜测。\n"
    "回答用中文，简洁准确，保留专业术语。"
)


def build_user_prompt(
    question: str,
    citations: list[Citation],
    max_chunk_chars: int = 500,
) -> str:
    """构建用户消息：参考资料 + 问题。

    Args:
        question: 用户问题
        citations: 检索到的带溯源的引用列表
        max_chunk_chars: 每条参考资料最长字符数，超出截断

    Returns:
        格式化的 user prompt 字符串
    """
    parts = ["【参考资料】"]

    for cite in citations:
        title = cite.section_title or cite.document_title
        location = f"{cite.document_title}"
        if cite.page:
            location += f" 第{cite.page}页"
        snippet = cite.content[:max_chunk_chars]
        if len(cite.content) > max_chunk_chars:
            snippet += "..."

        parts.append(f"[{cite.index}] 《{title}》（{location}）\n{snippet}")

    parts.append("")
    parts.append(f"【问题】\n{question}")
    parts.append("")
    parts.append("请基于上述参考资料回答问题，并在每个关键论点后标注引用编号，例如「...[1]」。")

    return "\n\n".join(parts)


def build_messages(
    question: str,
    citations: list[Citation],
    max_chunk_chars: int = 500,
    history: list[dict[str, str]] | None = None,
    summary: str = "",
    memory_context: str = "",
    skill_context: str = "",
    tool_context: str = "",
) -> list[dict[str, str]]:
    """构建 LLM 调用的 messages。

    Args:
        question: 当前问题
        citations: 检索到的引用列表
        max_chunk_chars: 每条参考资料最长字符数
        history: 历史对话消息（滑动窗口内的最近对话）
        summary: 早期对话的压缩摘要，注入为 system 消息辅助理解上下文
        memory_context: 长期记忆注入内容（当前主要是用户画像）
        skill_context: 匹配到的技能上下文
        tool_context: 工具执行结果上下文

    Returns:
        messages 列表
    """
    messages: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]
    if summary:
        messages.append(
            {"role": "system", "content": f"【历史对话摘要】\n{summary}"}
        )
    if memory_context:
        messages.append(
            {"role": "system", "content": f"【用户画像】\n{memory_context}"}
        )
    if skill_context:
        messages.append(
            {"role": "system", "content": f"【可用技能】\n{skill_context}"}
        )
    if tool_context:
        messages.append(
            {"role": "system", "content": tool_context}
        )
    if history:
        messages.extend(history)
    messages.append(
        {"role": "user", "content": build_user_prompt(question, citations, max_chunk_chars)}
    )
    return messages
