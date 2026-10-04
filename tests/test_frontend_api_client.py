"""前端 API 客户端测试。"""

import httpx

from frontend.api_client import KnowResearchApiClient


def _handler(request: httpx.Request) -> httpx.Response:
    if request.url.path == "/health":
        return httpx.Response(200, json={"status": "ok"})
    if request.url.path == "/chat":
        return httpx.Response(200, json={"text": "答案", "trace_id": "t1"})
    if request.url.path == "/documents/upload":
        return httpx.Response(200, json={"document_id": "doc-1"})
    if request.url.path == "/memory/candidates":
        return httpx.Response(200, json={"candidates": []})
    if request.url.path == "/skills":
        return httpx.Response(200, json={"skills": []})
    return httpx.Response(404, json={"detail": "not found"})


def _client() -> KnowResearchApiClient:
    return KnowResearchApiClient(
        base_url="http://testserver",
        transport=httpx.MockTransport(_handler),
    )


def test_api_client_health_and_chat():
    client = _client()
    assert client.health()["status"] == "ok"
    assert client.chat("你好")["text"] == "答案"


def test_api_client_upload_and_lists():
    client = _client()
    assert client.upload_document("a.pdf", b"pdf")["document_id"] == "doc-1"
    assert client.list_memory_candidates("user-a")["candidates"] == []
    assert client.list_skills("user-a")["skills"] == []
