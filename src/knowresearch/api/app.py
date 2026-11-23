"""FastAPI 服务层。

API 只负责参数校验、调用领域服务、返回结构化响应；
业务逻辑仍然在 QAEngine / 记忆服务 / 技能服务 / 工具层中。
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel

from knowresearch.core.observability import metrics
from knowresearch.core.schemas import UserFeedback


class ChatRequest(BaseModel):
    question: str
    session_id: str | None = None
    user_id: str | None = None


class ReviewRequest(BaseModel):
    action: Literal["approve", "reject"]
    user_id: str
    review_note: str = ""


class SkillStatusRequest(BaseModel):
    user_id: str
    action: Literal["enable", "disable"]


class SkillFeedbackRequest(BaseModel):
    user_id: str
    positive: bool


class FeedbackRequest(BaseModel):
    user_id: str
    trace_id: str = ""
    target_type: Literal["answer", "memory", "skill"] = "answer"
    target_id: str = ""
    rating: Literal["up", "down"]
    correction: str = ""


def create_app(
    qa_engine: Any | None = None,
    tool_registry: Any | None = None,
    memory_candidate_store: Any | None = None,
    profile_review_service: Any | None = None,
    skill_candidate_store: Any | None = None,
    skill_review_service: Any | None = None,
    skill_store: Any | None = None,
    skill_lifecycle_service: Any | None = None,
    feedback_service: Any | None = None,
    ingestion_pipeline: Any | None = None,
    object_storage: Any | None = None,
    allow_lazy_production: bool = True,
    health_checks: dict | None = None,
) -> FastAPI:
    app = FastAPI(title="KnowResearch API", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.qa_engine = qa_engine
    app.state.tool_registry = tool_registry
    app.state.memory_candidate_store = memory_candidate_store
    app.state.profile_review_service = profile_review_service
    app.state.skill_candidate_store = skill_candidate_store
    app.state.skill_review_service = skill_review_service
    app.state.skill_store = skill_store
    app.state.skill_lifecycle_service = skill_lifecycle_service
    app.state.feedback_service = feedback_service
    app.state.ingestion_pipeline = ingestion_pipeline
    app.state.object_storage = object_storage
    app.state.allow_lazy_production = allow_lazy_production
    app.state.health_checks = health_checks or {}

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        return JSONResponse(
            status_code=500,
            content={"error": "internal_error", "detail": str(exc)},
        )

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "services": {
                "qa_engine": app.state.qa_engine is not None,
                "tool_registry": app.state.tool_registry is not None,
                "memory_candidate_store": app.state.memory_candidate_store
                is not None,
                "skill_store": app.state.skill_store is not None,
                "ingestion_pipeline": app.state.ingestion_pipeline is not None,
            },
        }

    @app.get("/health/ready")
    def readiness():
        checks = {}
        for name, check in app.state.health_checks.items():
            try:
                checks[name] = {"ok": bool(check()), "error": ""}
            except Exception as e:
                checks[name] = {"ok": False, "error": str(e)}
        return {
            "status": "ok"
            if all(item["ok"] for item in checks.values())
            else "degraded",
            "checks": checks,
        }

    @app.get("/metrics")
    def get_metrics():
        return metrics.snapshot()

    @app.post("/feedback")
    def submit_feedback(request: FeedbackRequest):
        service = _require(app.state.feedback_service, "反馈服务未配置")
        feedback = UserFeedback(
            user_id=request.user_id,
            trace_id=request.trace_id,
            target_type=request.target_type,
            target_id=request.target_id,
            rating=request.rating,
            correction=request.correction,
        )
        service.submit(feedback)
        return {"success": True, "feedback_id": feedback.id}

    @app.post("/chat")
    def chat(request: ChatRequest):
        engine = _get_qa_engine(app)
        try:
            answer = engine.answer(
                request.question,
                session_id=request.session_id,
                user_id=request.user_id,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e
        return {
            "text": answer.text,
            "trace_id": answer.trace_id,
            "citations": [
                citation.model_dump(mode="json")
                for citation in answer.citations
            ],
        }

    @app.post("/chat/stream")
    def chat_stream(request: ChatRequest):
        """SSE 流式问答。

        响应 Content-Type 为 text/event-stream，每条事件格式为：
            data: {"type": "<event>", ...}\\n\\n
        事件类型见 QAEngine.answer_stream：meta / start / delta / done / error。
        用同步生成器 + 同步路由，FastAPI 会在线程池中执行，避免阻塞事件循环。
        """
        engine = _get_qa_engine(app)

        def event_generator():
            try:
                for event_type, payload in engine.answer_stream(
                    request.question,
                    session_id=request.session_id,
                    user_id=request.user_id,
                ):
                    data = {"type": event_type}
                    data.update(payload)
                    yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
            except Exception as e:
                yield (
                    f"data: {json.dumps({'type': 'error', 'detail': str(e)}, ensure_ascii=False)}\n\n"
                )

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @app.get("/tools")
    def list_tools():
        registry = app.state.tool_registry
        if registry is None and app.state.allow_lazy_production:
            try:
                _get_qa_engine(app)
                registry = app.state.tool_registry
            except Exception:
                registry = None
        if registry is None:
            return {"tools": []}
        return {
            "tools": [
                tool.definition.model_dump(mode="json")
                for tool in registry.list_tools()
            ]
        }

    @app.get("/memory/candidates")
    def list_memory_candidates(
        user_id: str,
        status: str | None = None,
        limit: int = 100,
    ):
        store = _require(
            app.state.memory_candidate_store,
            "记忆候选存储未配置",
        )
        candidates = store.list(user_id=user_id, status=status, limit=limit)
        return {
            "candidates": [
                candidate.model_dump(mode="json")
                for candidate in candidates
            ]
        }

    @app.post("/memory/candidates/{candidate_id}/review")
    def review_memory_candidate(candidate_id: str, request: ReviewRequest):
        service = _require(
            app.state.profile_review_service,
            "画像审核服务未配置",
        )
        if request.action == "approve":
            profile = service.approve(
                candidate_id,
                request.user_id,
                review_note=request.review_note,
            )
        else:
            profile = service.reject(
                candidate_id,
                request.user_id,
                review_note=request.review_note,
            )
        return {
            "success": profile is not None if request.action == "approve" else profile,
            "profile_id": profile.id if hasattr(profile, "id") else None,
        }

    @app.get("/skills/candidates")
    def list_skill_candidates(
        user_id: str,
        status: str | None = None,
        limit: int = 100,
    ):
        store = _require(
            app.state.skill_candidate_store,
            "技能候选存储未配置",
        )
        candidates = store.list(user_id=user_id, status=status, limit=limit)
        return {
            "candidates": [
                candidate.model_dump(mode="json")
                for candidate in candidates
            ]
        }

    @app.post("/skills/candidates/{candidate_id}/review")
    def review_skill_candidate(candidate_id: str, request: ReviewRequest):
        service = _require(
            app.state.skill_review_service,
            "技能审核服务未配置",
        )
        if request.action == "approve":
            skill = service.approve(
                candidate_id,
                request.user_id,
                review_note=request.review_note,
            )
        else:
            skill = service.reject(
                candidate_id,
                request.user_id,
                review_note=request.review_note,
            )
        return {
            "success": skill is not None if request.action == "approve" else skill,
            "skill_id": skill.id if hasattr(skill, "id") else None,
        }

    @app.get("/skills")
    def list_skills(
        user_id: str,
        status: str | None = None,
        limit: int = 100,
    ):
        store = _require(app.state.skill_store, "技能存储未配置")
        skills = store.list_all(user_id=user_id, status=status, limit=limit)
        return {
            "skills": [skill.model_dump(mode="json") for skill in skills]
        }

    @app.post("/skills/{skill_id}/status")
    def change_skill_status(skill_id: str, request: SkillStatusRequest):
        service = _require(
            app.state.skill_lifecycle_service,
            "技能生命周期服务未配置",
        )
        if request.action == "enable":
            service.enable(skill_id, request.user_id)
        else:
            service.disable(skill_id, request.user_id)
        return {"success": True}

    @app.post("/skills/{skill_id}/feedback")
    def skill_feedback(skill_id: str, request: SkillFeedbackRequest):
        service = _require(
            app.state.skill_lifecycle_service,
            "技能生命周期服务未配置",
        )
        service.record_feedback(
            skill_id,
            request.user_id,
            positive=request.positive,
        )
        return {"success": True}

    @app.post("/documents/upload")
    async def upload_document(file: UploadFile = File(...)):
        pipeline = _require(
            app.state.ingestion_pipeline,
            "摄取流水线未配置",
        )
        suffix = Path(file.filename or "").suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            temp_path = tmp.name
        try:
            result = pipeline.ingest_file(temp_path)
        finally:
            Path(temp_path).unlink(missing_ok=True)
        return {
            "file_path": result.file_path,
            "document_id": result.document_id,
            "section_count": result.section_count,
            "chunk_count": result.chunk_count,
            "table_count": getattr(result, "table_count", 0),
            "figure_count": getattr(result, "figure_count", 0),
            "skipped": result.skipped,
            "error": result.error,
        }

    @app.get("/documents/{document_id}/figures/{key:path}")
    def get_figure_image(document_id: str, key: str):
        storage = _get_object_storage(app)
        full_key = f"figures/{document_id}/{key}"
        try:
            data = storage.get(full_key)
        except Exception:
            raise HTTPException(status_code=404, detail="图片不存在") from None
        return Response(content=data, media_type="image/png")

    return app


def _require(service, message: str):
    if service is None:
        raise HTTPException(status_code=503, detail=message)
    return service


def _get_qa_engine(app: FastAPI):
    if app.state.qa_engine is None:
        if not app.state.allow_lazy_production:
            raise HTTPException(status_code=503, detail="QAEngine 未配置")
        from knowresearch.bootstrap import (
            build_agent_engine,
            build_production_qa_engine,
        )
        from knowresearch.config import settings

        base_engine = build_production_qa_engine()
        if settings.agent.enabled:
            # ReAct 慢路径：复杂问题走多步循环，简单问题仍走 base_engine 快路径
            app.state.qa_engine = build_agent_engine(
                qa_engine=base_engine,
                long_term_memory=getattr(
                    base_engine.memory_context, "long_term_memory", None
                ),
                document_store=getattr(
                    base_engine.citation_builder, "document_store", None
                ),
            )
        else:
            app.state.qa_engine = base_engine
        app.state.tool_registry = getattr(
            base_engine,
            "tool_registry",
            None,
        )
    return app.state.qa_engine


def _get_object_storage(app: FastAPI):
    if app.state.object_storage is not None:
        return app.state.object_storage
    if not app.state.allow_lazy_production:
        raise HTTPException(status_code=503, detail="对象存储未配置")
    from knowresearch.bootstrap import build_object_storage

    app.state.object_storage = build_object_storage()
    return app.state.object_storage


app = create_app()