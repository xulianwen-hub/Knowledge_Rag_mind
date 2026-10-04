"""SessionStorePort 的 Redis 实现。"""

from __future__ import annotations

import json

import redis

from knowresearch.core.ports import SessionStorePort


class RedisSessionStore(SessionStorePort):
    """基于 Redis 的短期记忆，使用原生 TTL 自动过期。"""

    def __init__(
        self,
        redis_url: str,
        prefix: str = "session",
        default_ttl_seconds: int = 3600,
    ):
        self._client = redis.Redis.from_url(redis_url, decode_responses=True)
        self._prefix = prefix
        self._default_ttl_seconds = default_ttl_seconds

    def _key(self, session_id: str) -> str:
        return f"{self._prefix}:{session_id}"

    def save_messages(
        self,
        session_id: str,
        messages: list[dict[str, str]],
        ttl_seconds: int = 3600,
        summary: str = "",
    ) -> None:
        payload = json.dumps(
            {"messages": messages, "summary": summary},
            ensure_ascii=False,
        )
        self._client.setex(
            self._key(session_id),
            ttl_seconds,
            payload,
        )

    def _load_payload(self, session_id: str) -> dict | None:
        raw = self._client.get(self._key(session_id))
        if raw is None:
            return None
        return json.loads(raw)

    def load_messages(self, session_id: str) -> list[dict[str, str]]:
        payload = self._load_payload(session_id)
        if payload is None:
            return []
        messages = payload.get("messages", [])
        return messages if isinstance(messages, list) else []

    def load_summary(self, session_id: str) -> str:
        payload = self._load_payload(session_id)
        if payload is None:
            return ""
        return payload.get("summary", "") or ""

    def delete_session(self, session_id: str) -> None:
        self._client.delete(self._key(session_id))
