"""前端 API 客户端：所有请求都通过 FastAPI。"""

from __future__ import annotations

import json
from typing import Iterator

import httpx


class ApiError(RuntimeError):
    """API 调用失败。"""


class KnowResearchApiClient:
    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        timeout_seconds: float = 60.0,
        transport: httpx.BaseTransport | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=timeout_seconds,
            transport=transport,
        )

    def close(self) -> None:
        self._client.close()

    def health(self) -> dict:
        return self._request("GET", "/health")

    def chat(
        self,
        question: str,
        session_id: str | None = None,
        user_id: str | None = None,
    ) -> dict:
        return self._request(
            "POST",
            "/chat",
            json={
                "question": question,
                "session_id": session_id,
                "user_id": user_id,
            },
        )

    def chat_stream(
        self,
        question: str,
        session_id: str | None = None,
        user_id: str | None = None,
    ) -> Iterator[dict]:
        """流式问答，返回解析后的 SSE 事件迭代器。

        每个元素是一个 dict，含 "type" 字段（meta/start/delta/done/error）。
        必须消费完或显式关闭生成器，以释放底层连接。
        """
        with self._client.stream(
            "POST",
            "/chat/stream",
            json={
                "question": question,
                "session_id": session_id,
                "user_id": user_id,
            },
        ) as response:
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as e:
                detail = ""
                try:
                    detail = e.response.json().get("detail", "")
                except Exception:
                    detail = e.response.text
                raise ApiError(f"HTTP {e.response.status_code}: {detail}") from e
            for line in response.iter_lines():
                if not line:
                    continue
                if line.startswith("data: "):
                    yield json.loads(line[6:])

    def list_tools(self) -> dict:
        return self._request("GET", "/tools")

    def upload_document(self, filename: str, content: bytes) -> dict:
        return self._request(
            "POST",
            "/documents/upload",
            files={"file": (filename, content)},
        )

    def list_memory_candidates(
        self,
        user_id: str,
        status: str | None = None,
    ) -> dict:
        params = {"user_id": user_id}
        if status:
            params["status"] = status
        return self._request("GET", "/memory/candidates", params=params)

    def review_memory_candidate(
        self,
        candidate_id: str,
        user_id: str,
        action: str,
        review_note: str = "",
    ) -> dict:
        return self._request(
            "POST",
            f"/memory/candidates/{candidate_id}/review",
            json={
                "action": action,
                "user_id": user_id,
                "review_note": review_note,
            },
        )

    def list_skill_candidates(
        self,
        user_id: str,
        status: str | None = None,
    ) -> dict:
        params = {"user_id": user_id}
        if status:
            params["status"] = status
        return self._request("GET", "/skills/candidates", params=params)

    def review_skill_candidate(
        self,
        candidate_id: str,
        user_id: str,
        action: str,
        review_note: str = "",
    ) -> dict:
        return self._request(
            "POST",
            f"/skills/candidates/{candidate_id}/review",
            json={
                "action": action,
                "user_id": user_id,
                "review_note": review_note,
            },
        )

    def list_skills(self, user_id: str, status: str | None = None) -> dict:
        params = {"user_id": user_id}
        if status:
            params["status"] = status
        return self._request("GET", "/skills", params=params)

    def set_skill_status(self, skill_id: str, user_id: str, action: str) -> dict:
        return self._request(
            "POST",
            f"/skills/{skill_id}/status",
            json={"action": action, "user_id": user_id},
        )

    def send_skill_feedback(
        self,
        skill_id: str,
        user_id: str,
        positive: bool,
    ) -> dict:
        return self._request(
            "POST",
            f"/skills/{skill_id}/feedback",
            json={"user_id": user_id, "positive": positive},
        )

    def _request(self, method: str, path: str, **kwargs) -> dict:
        try:
            response = self._client.request(method, path, **kwargs)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            detail = ""
            try:
                detail = e.response.json().get("detail", "")
            except Exception:
                detail = e.response.text
            raise ApiError(f"HTTP {e.response.status_code}: {detail}") from e
        except httpx.HTTPError as e:
            raise ApiError(f"请求失败: {e}") from e