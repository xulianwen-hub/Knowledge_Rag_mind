"""KnowResearch Streamlit 前端。"""

from __future__ import annotations

import os
import uuid

import streamlit as st

from frontend.api_client import ApiError, KnowResearchApiClient


st.set_page_config(page_title="知研 KnowResearch", page_icon="📚", layout="wide")


def _client() -> KnowResearchApiClient:
    base_url = st.session_state.get(
        "api_base_url",
        os.getenv("KNOWRESEARCH_API_URL", "http://localhost:8000"),
    )
    return KnowResearchApiClient(base_url=base_url)


def _init_state() -> None:
    st.session_state.setdefault("session_id", uuid.uuid4().hex)
    st.session_state.setdefault("user_id", "default")
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault(
        "api_base_url",
        os.getenv("KNOWRESEARCH_API_URL", "http://localhost:8000"),
    )


def _render_sidebar() -> str:
    st.sidebar.title("知研 KnowResearch")
    st.session_state["api_base_url"] = st.sidebar.text_input(
        "API 地址",
        value=st.session_state["api_base_url"],
    )
    st.session_state["user_id"] = st.sidebar.text_input(
        "用户 ID",
        value=st.session_state["user_id"],
    )
    try:
        health = _client().health()
        st.sidebar.success(f"API 正常：{health.get('status', 'unknown')}")
    except ApiError as e:
        st.sidebar.error(f"API 不可用：{e}")
    pages = ["问答", "知识库", "记忆管理", "技能管理"]
    return st.sidebar.radio("功能", pages, index=pages.index(_query_page()))


def _query_page() -> str:
    page = st.query_params.get("page", "chat")
    mapping = {
        "chat": "问答",
        "knowledge": "知识库",
        "memory": "记忆管理",
        "skills": "技能管理",
    }
    return mapping.get(page, "问答")


def _maybe_seed_demo_question() -> None:
    if st.query_params.get("demo") != "1":
        return
    if st.session_state["messages"]:
        return
    question = "襟翼舵优化为什么要在升力和阻力之间权衡？"
    st.session_state["messages"].append({"role": "user", "content": question})
    try:
        result = _client().chat(
            question,
            session_id=st.session_state["session_id"],
            user_id=st.session_state["user_id"],
        )
        st.session_state["messages"].append(
            {
                "role": "assistant",
                "content": result.get("text", ""),
                "citations": result.get("citations", []),
            }
        )
    except ApiError as e:
        st.session_state["messages"].append(
            {"role": "assistant", "content": f"Demo 请求失败：{e}"}
        )


def _render_chat() -> None:
    st.header("问答")
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            for citation in message.get("citations", []):
                st.caption(
                    f"[{citation.get('index', '')}] "
                    f"{citation.get('document_title', '')} "
                    f"第{citation.get('page', 0)}页"
                )

    question = st.chat_input("输入问题")
    if question:
        st.session_state["messages"].append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        try:
            result = _client().chat(
                question,
                session_id=st.session_state["session_id"],
                user_id=st.session_state["user_id"],
            )
            answer = result.get("text", "")
            citations = result.get("citations", [])
            st.session_state["messages"].append(
                {"role": "assistant", "content": answer, "citations": citations}
            )
            with st.chat_message("assistant"):
                st.markdown(answer)
                for citation in citations:
                    st.caption(
                        f"[{citation.get('index', '')}] "
                        f"{citation.get('document_title', '')} "
                        f"第{citation.get('page', 0)}页"
                    )
        except ApiError as e:
            st.error(f"问答失败：{e}")


def _render_knowledge() -> None:
    st.header("知识库")
    uploaded = st.file_uploader("上传 PDF / Word", type=["pdf", "docx"])
    if uploaded is not None and st.button("开始摄取"):
        try:
            result = _client().upload_document(
                uploaded.name,
                uploaded.getvalue(),
            )
            if result.get("error"):
                st.error(result["error"])
            else:
                st.success(
                    f"摄取完成：{result.get('chunk_count', 0)} 个 chunk"
                )
        except ApiError as e:
            st.error(f"上传失败：{e}")


def _render_memory() -> None:
    st.header("记忆管理")
    try:
        data = _client().list_memory_candidates(
            st.session_state["user_id"],
            status="pending",
        )
    except ApiError as e:
        st.error(f"加载失败：{e}")
        return

    candidates = data.get("candidates", [])
    if not candidates:
        st.info("暂无待审核画像")
        return

    for candidate in candidates:
        with st.expander(
            f"{candidate.get('metadata', {}).get('dimension', 'profile')} "
            f"| 置信度 {candidate.get('confidence', 0):.2f}"
        ):
            st.write(candidate.get("content", ""))
            st.caption(f"依据：{candidate.get('evidence', '')}")
            col1, col2 = st.columns(2)
            if col1.button("通过", key=f"approve-memory-{candidate['id']}"):
                _review_memory(candidate["id"], "approve")
            if col2.button("拒绝", key=f"reject-memory-{candidate['id']}"):
                _review_memory(candidate["id"], "reject")


def _review_memory(candidate_id: str, action: str) -> None:
    try:
        _client().review_memory_candidate(
            candidate_id,
            st.session_state["user_id"],
            action,
        )
        st.success("操作完成")
        st.rerun()
    except ApiError as e:
        st.error(f"操作失败：{e}")


def _render_skills() -> None:
    st.header("技能管理")
    tab1, tab2 = st.tabs(["待审核候选", "已启用技能"])
    with tab1:
        _render_skill_candidates()
    with tab2:
        _render_active_skills()


def _render_skill_candidates() -> None:
    try:
        data = _client().list_skill_candidates(
            st.session_state["user_id"],
            status="pending",
        )
    except ApiError as e:
        st.error(f"加载失败：{e}")
        return

    for candidate in data.get("candidates", []):
        with st.expander(
            f"{candidate.get('name', '技能')} "
            f"| 置信度 {candidate.get('confidence', 0):.2f}"
        ):
            st.write(candidate.get("description", ""))
            st.caption(f"触发：{candidate.get('trigger', '')}")
            if candidate.get("metadata", {}).get("replay"):
                st.json(candidate["metadata"]["replay"])
            col1, col2 = st.columns(2)
            if col1.button("通过", key=f"approve-skill-{candidate['id']}"):
                _review_skill(candidate["id"], "approve")
            if col2.button("拒绝", key=f"reject-skill-{candidate['id']}"):
                _review_skill(candidate["id"], "reject")


def _review_skill(candidate_id: str, action: str) -> None:
    try:
        _client().review_skill_candidate(
            candidate_id,
            st.session_state["user_id"],
            action,
        )
        st.success("操作完成")
        st.rerun()
    except ApiError as e:
        st.error(f"操作失败：{e}")


def _render_active_skills() -> None:
    try:
        data = _client().list_skills(st.session_state["user_id"])
    except ApiError as e:
        st.error(f"加载失败：{e}")
        return

    for skill in data.get("skills", []):
        metadata = skill.get("metadata", {})
        with st.expander(metadata.get("name", "技能")):
            st.write(skill.get("content", ""))
            st.caption(
                f"使用 {metadata.get('usage_count', 0)} 次 | "
                f"成功 {metadata.get('success_count', 0)} 次 | "
                f"版本 {metadata.get('version', 1)}"
            )
            col1, col2, col3 = st.columns(3)
            if col1.button("禁用", key=f"disable-skill-{skill['id']}"):
                _set_skill_status(skill["id"], "disable")
            if col2.button("👍", key=f"up-skill-{skill['id']}"):
                _send_skill_feedback(skill["id"], True)
            if col3.button("👎", key=f"down-skill-{skill['id']}"):
                _send_skill_feedback(skill["id"], False)


def _set_skill_status(skill_id: str, action: str) -> None:
    try:
        _client().set_skill_status(
            skill_id,
            st.session_state["user_id"],
            action,
        )
        st.success("操作完成")
        st.rerun()
    except ApiError as e:
        st.error(f"操作失败：{e}")


def _send_skill_feedback(skill_id: str, positive: bool) -> None:
    try:
        _client().send_skill_feedback(
            skill_id,
            st.session_state["user_id"],
            positive,
        )
        st.success("反馈已记录")
    except ApiError as e:
        st.error(f"反馈失败：{e}")


def main() -> None:
    _init_state()
    _maybe_seed_demo_question()
    page = _render_sidebar()
    if page == "问答":
        _render_chat()
    elif page == "知识库":
        _render_knowledge()
    elif page == "记忆管理":
        _render_memory()
    else:
        _render_skills()


main()
