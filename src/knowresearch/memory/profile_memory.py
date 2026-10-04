"""用户画像注入服务：给 RAG 主链路提供 query 改写和上下文。"""

from __future__ import annotations

from knowresearch.core.ports import LongTermMemoryPort, MemoryContextPort


AMBIGUOUS_MARKERS = (
    "它",
    "这个",
    "那个",
    "该方法",
    "上面",
    "刚才",
    "继续",
    "还有呢",
)


class ProfileMemoryService(MemoryContextPort):
    """只服务 profile 的记忆上下文实现。"""

    def __init__(self, long_term_memory: LongTermMemoryPort):
        self.long_term_memory = long_term_memory

    def rewrite_query(self, user_id: str, query: str) -> str:
        if not user_id or not query:
            return query
        try:
            profiles = self._relevant_profiles(user_id, query, top_k=2)
        except Exception:
            return query
        if not profiles:
            return query
        if not self._is_ambiguous(query):
            return query
        hints = " ".join(profile.content[:40] for profile in profiles)
        return f"{query} {hints}".strip()

    def build_context(
        self,
        user_id: str,
        query: str,
        max_chars: int = 800,
    ) -> str:
        if not user_id:
            return ""
        try:
            profiles = self._relevant_profiles(user_id, query, top_k=5)
        except Exception:
            return ""
        if not profiles:
            return ""

        lines: list[str] = []
        used = 0
        for profile in profiles:
            dimension = profile.metadata.get("dimension", "profile")
            line = f"- [{dimension}] {profile.content}"
            if used + len(line) > max_chars:
                break
            lines.append(line)
            used += len(line)
        return "\n".join(lines)

    def _relevant_profiles(self, user_id: str, query: str, top_k: int):
        if query:
            matches = self.long_term_memory.search(
                query,
                top_k=top_k,
                user_id=user_id,
            )
            if matches:
                return matches
        return self.long_term_memory.list(
            user_id=user_id,
            kind="profile",
        )[:top_k]

    @staticmethod
    def _is_ambiguous(query: str) -> bool:
        if len(query) < 12:
            return True
        return any(marker in query for marker in AMBIGUOUS_MARKERS)
