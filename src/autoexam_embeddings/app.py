"""FastAPI application for configured dense embeddings."""

from __future__ import annotations

import logging
import os
import re
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse

from .config import is_valid_locale, load_config
from .registry import (
    ModelFactory,
    ModelRegistry,
    ModelUnavailableError,
    RequestLimitError,
    log_event,
    reset_log_correlation,
    set_log_correlation,
)
from .schemas import (
    ChunkRequest,
    ChunkResponse,
    EmbeddingsRequest,
    EmbeddingsResponse,
    ModelsResponse,
    TokenLimits,
)

DEFAULT_CONFIG_PATH = "/etc/autoexam-embeddings/config.json"


def create_app(
    *, config_path: str | Path | None = None, model_factory: ModelFactory | None = None
) -> FastAPI:
    """Create an application whose configuration is read in its lifespan startup."""

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        resolved_path: str | Path
        if config_path is not None:
            resolved_path = config_path
        else:
            resolved_path = os.environ.get("EMBEDDINGS_CONFIG_PATH", DEFAULT_CONFIG_PATH)
        config = load_config(resolved_path)
        registry = ModelRegistry(
            config, **({"model_factory": model_factory} if model_factory else {})
        )
        application.state.registry = registry
        try:
            await registry.start()
            yield
        finally:
            await registry.shutdown()

    application = FastAPI(
        title="AutoExam Embeddings",
        version="0.1.0",
        lifespan=lifespan,
    )

    @application.middleware("http")
    async def correlation_logging(request: Request, call_next):  # type: ignore[no-untyped-def]
        def bounded(name: str, generated: str = "") -> str:
            value = request.headers.get(name, generated).strip()
            return (
                value
                if len(value) <= 200 and re.fullmatch(r"[A-Za-z0-9._:-]+", value)
                else generated
            )

        request_id = bounded("X-Request-ID", str(uuid.uuid4()))
        fields = {"request_id": request_id}
        for header, field in (
            ("X-Workflow-ID", "workflow_id"),
            ("X-RAG-Document-ID", "rag_document_id"),
            ("X-RAG-Attempt", "rag_attempt"),
        ):
            if value := bounded(header):
                fields[field] = value
        token = set_log_correlation(fields)
        started_at = time.perf_counter()
        log_event(
            "request_started",
            level=logging.DEBUG,
            method=request.method,
            path=request.url.path,
        )
        try:
            response = await call_next(request)
            response.headers["X-Request-ID"] = request_id
            log_event(
                "request_completed",
                level=logging.DEBUG,
                method=request.method,
                path=request.url.path,
                status=response.status_code,
                duration_ms=round((time.perf_counter() - started_at) * 1000, 3),
            )
            return response
        finally:
            reset_log_correlation(token)

    @application.get("/health/live")
    async def health_live() -> dict[str, str]:
        return {"status": "live"}

    @application.get("/health/ready")
    async def health_ready(request: Request) -> JSONResponse:
        registry = _registry(request)
        ready = registry.ready
        return JSONResponse(
            status_code=200 if ready else 503,
            content={"status": "ready" if ready else "not_ready"},
        )

    @application.get("/v1/models", response_model=ModelsResponse)
    async def list_models(
        request: Request,
        locale: str | None = Query(default=None),
    ) -> ModelsResponse:
        if locale is not None and not is_valid_locale(locale, allow_multilingual=True):
            raise HTTPException(status_code=422, detail="locale has invalid syntax")
        return ModelsResponse(data=_registry(request).describe(locale))

    @application.post(
        "/v1/models/{alias}/embeddings",
        response_model=EmbeddingsResponse,
    )
    async def create_embeddings(
        alias: str, payload: EmbeddingsRequest, request: Request
    ) -> EmbeddingsResponse:
        registry = _registry(request)
        if registry.get(alias) is None:
            raise HTTPException(status_code=404, detail="unknown model alias")
        try:
            return await registry.embed(alias, payload.input, payload.input_type)
        except RequestLimitError as error:
            raise HTTPException(status_code=413, detail=str(error)) from error
        except ModelUnavailableError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error

    @application.get("/v1/models/{alias}/limits", response_model=TokenLimits)
    async def model_limits(alias: str, request: Request) -> TokenLimits:
        registry = _registry(request)
        if registry.get(alias) is None:
            raise HTTPException(status_code=404, detail="unknown model alias")
        try:
            return await registry.token_limits(alias)
        except ModelUnavailableError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error

    @application.post("/v1/models/{alias}/chunks", response_model=ChunkResponse)
    async def chunks(alias: str, payload: ChunkRequest, request: Request) -> ChunkResponse:
        registry = _registry(request)
        if registry.get(alias) is None:
            raise HTTPException(status_code=404, detail="unknown model alias")
        try:
            return await registry.chunk(alias, payload)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        except ModelUnavailableError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error

    return application


def _registry(request: Request) -> ModelRegistry:
    registry: ModelRegistry | None = getattr(request.app.state, "registry", None)
    if registry is None:
        raise HTTPException(status_code=503, detail="service startup is incomplete")
    return registry


logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO").upper())
app = create_app()
