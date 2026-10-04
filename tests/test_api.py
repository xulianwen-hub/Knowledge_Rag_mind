"""M6-4 FastAPI 服务层测试。"""

from fastapi.testclient import TestClient

from knowresearch.api import create_app
from knowresearch.core.schemas import (
    MemoryCandidate,
    Record,
    SkillCandidate,
    ToolDefinition,
)
from knowresearch.tools import ToolRegistry


class _FakeAnswer:
    def __init__(self, text="答案", trace_id="trace-1"):
        self.text = text
        self.trace_id = trace_id
        self.citations = []


class _FakeQAEngine:
    def __init__(self, raise_exc=False):
        self.raise_exc = raise_exc

    def answer(self, question, session_id=None, user_id=None):
        if self.raise_exc:
            raise RuntimeError("问答失败")
        return _FakeAnswer(text=f"回答：{question}")


class _FakeTool:
    @property
    def definition(self):
        return ToolDefinition(name="echo", description="测试工具")

    def run(self, arguments):
        return arguments


class _FakeCandidateStore:
    def __init__(self, items):
        self.items = items

    def list(self, user_id=None, status=None, limit=100):
        return [item for item in self.items if item.user_id == user_id][:limit]


class _FakeReviewService:
    def __init__(self, result=True):
        self.result = result

    def approve(self, candidate_id, user_id, review_note=""):
        return Record(id=f"approved-{candidate_id}", kind="profile")

    def reject(self, candidate_id, user_id, review_note=""):
        return self.result


class _FakeSkillStore:
    def __init__(self, skills):
        self.skills = skills

    def list_all(self, user_id=None, status=None, limit=100):
        return [skill for skill in self.skills if skill.metadata.get("user_id") == user_id]


class _FakeLifecycle:
    def __init__(self):
        self.feedback = []
        self.statuses = []

    def enable(self, skill_id, user_id):
        self.statuses.append((skill_id, "enable"))

    def disable(self, skill_id, user_id):
        self.statuses.append((skill_id, "disable"))

    def record_feedback(self, skill_id, user_id, positive):
        self.feedback.append((skill_id, user_id, positive))


class _FakeIngestResult:
    file_path = "test.pdf"
    document_id = "doc-1"
    section_count = 1
    chunk_count = 2
    skipped = False
    error = ""


class _FakePipeline:
    def ingest_file(self, path):
        return _FakeIngestResult()


class _FakeFeedbackService:
    def __init__(self):
        self.items = []

    def submit(self, feedback):
        self.items.append(feedback)


def test_health_and_missing_services():
    client = TestClient(create_app(allow_lazy_production=False))
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

    chat = client.post("/chat", json={"question": "你好"})
    assert chat.status_code == 503


def test_chat_and_chat_error():
    client = TestClient(create_app(qa_engine=_FakeQAEngine()))
    response = client.post(
        "/chat",
        json={"question": "你好", "user_id": "user-a"},
    )
    assert response.status_code == 200
    assert response.json()["text"] == "回答：你好"

    failing = TestClient(create_app(qa_engine=_FakeQAEngine(raise_exc=True)))
    response = failing.post("/chat", json={"question": "你好"})
    assert response.status_code == 500


def test_tools_memory_skills_endpoints():
    registry = ToolRegistry()
    registry.register(_FakeTool())
    memory_candidate = MemoryCandidate(user_id="user-a", content="画像")
    skill_candidate = SkillCandidate(user_id="user-a", name="技能", description="说明")
    skill = Record(
        kind="skill",
        content="说明",
        metadata={"user_id": "user-a", "name": "技能"},
    )
    lifecycle = _FakeLifecycle()
    app = create_app(
        qa_engine=_FakeQAEngine(),
        tool_registry=registry,
        memory_candidate_store=_FakeCandidateStore([memory_candidate]),
        profile_review_service=_FakeReviewService(),
        skill_candidate_store=_FakeCandidateStore([skill_candidate]),
        skill_review_service=_FakeReviewService(),
        skill_store=_FakeSkillStore([skill]),
        skill_lifecycle_service=lifecycle,
        ingestion_pipeline=_FakePipeline(),
    )
    client = TestClient(app)

    tools = client.get("/tools").json()["tools"]
    assert tools[0]["name"] == "echo"

    memory = client.get("/memory/candidates", params={"user_id": "user-a"}).json()
    assert len(memory["candidates"]) == 1
    review = client.post(
        f"/memory/candidates/{memory_candidate.id}/review",
        json={"action": "approve", "user_id": "user-a"},
    )
    assert review.status_code == 200

    skills = client.get("/skills/candidates", params={"user_id": "user-a"}).json()
    assert len(skills["candidates"]) == 1
    review = client.post(
        f"/skills/candidates/{skill_candidate.id}/review",
        json={"action": "reject", "user_id": "user-a"},
    )
    assert review.status_code == 200

    assert len(client.get("/skills", params={"user_id": "user-a"}).json()["skills"]) == 1
    assert client.post(
        f"/skills/{skill.id}/status",
        json={"action": "disable", "user_id": "user-a"},
    ).status_code == 200
    assert client.post(
        f"/skills/{skill.id}/feedback",
        json={"user_id": "user-a", "positive": True},
    ).status_code == 200


def test_upload_document():
    client = TestClient(create_app(ingestion_pipeline=_FakePipeline()))
    response = client.post(
        "/documents/upload",
        files={"file": ("test.pdf", b"%PDF-1.4", "application/pdf")},
    )
    assert response.status_code == 200
    assert response.json()["document_id"] == "doc-1"


def test_m6_end_to_end_flow():
    registry = ToolRegistry()
    registry.register(_FakeTool())
    memory_candidate = MemoryCandidate(user_id="user-a", content="画像")
    skill_candidate = SkillCandidate(user_id="user-a", name="技能", description="说明")
    skill = Record(
        kind="skill",
        content="说明",
        metadata={"user_id": "user-a", "name": "技能"},
    )
    lifecycle = _FakeLifecycle()
    client = TestClient(
        create_app(
            qa_engine=_FakeQAEngine(),
            tool_registry=registry,
            memory_candidate_store=_FakeCandidateStore([memory_candidate]),
            profile_review_service=_FakeReviewService(),
            skill_candidate_store=_FakeCandidateStore([skill_candidate]),
            skill_review_service=_FakeReviewService(),
            skill_store=_FakeSkillStore([skill]),
            skill_lifecycle_service=lifecycle,
            ingestion_pipeline=_FakePipeline(),
        )
    )

    assert client.get("/health").status_code == 200
    assert client.post(
        "/documents/upload",
        files={"file": ("test.pdf", b"%PDF-1.4", "application/pdf")},
    ).status_code == 200
    assert client.post("/chat", json={"question": "问题"}).status_code == 200
    assert client.get(
        "/memory/candidates", params={"user_id": "user-a"}
    ).status_code == 200
    assert client.post(
        f"/memory/candidates/{memory_candidate.id}/review",
        json={"action": "approve", "user_id": "user-a"},
    ).status_code == 200
    assert client.get(
        "/skills/candidates", params={"user_id": "user-a"}
    ).status_code == 200
    assert client.post(
        f"/skills/candidates/{skill_candidate.id}/review",
        json={"action": "approve", "user_id": "user-a"},
    ).status_code == 200
    assert client.post(
        f"/skills/{skill.id}/status",
        json={"action": "disable", "user_id": "user-a"},
    ).status_code == 200
    assert client.post(
        f"/skills/{skill.id}/feedback",
        json={"user_id": "user-a", "positive": True},
    ).status_code == 200
    assert lifecycle.feedback == [(skill.id, "user-a", True)]


def test_readiness_metrics_and_feedback_endpoints():
    feedback_service = _FakeFeedbackService()
    client = TestClient(
        create_app(
            feedback_service=feedback_service,
            health_checks={
                "ok": lambda: True,
                "bad": lambda: (_ for _ in ()).throw(RuntimeError("down")),
            },
        )
    )

    ready = client.get("/health/ready")
    assert ready.status_code == 200
    assert ready.json()["status"] == "degraded"

    metrics = client.get("/metrics")
    assert metrics.status_code == 200
    assert "counters" in metrics.json()

    feedback = client.post(
        "/feedback",
        json={
            "user_id": "user-a",
            "target_type": "skill",
            "target_id": "skill-1",
            "rating": "up",
        },
    )
    assert feedback.status_code == 200
    assert feedback_service.items[0].rating == "up"
