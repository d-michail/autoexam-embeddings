from __future__ import annotations

import asyncio
import json
import threading
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi import FastAPI

from autoexam_embeddings.app import create_app
from autoexam_embeddings.chunking import fingerprint
from autoexam_embeddings.config import ModelConfig, ServiceConfig
from autoexam_embeddings.registry import ModelRegistry


class FakeModel:
    def __init__(self, dimension: int = 3) -> None:
        self.max_seq_length = 0
        self.dimension = dimension
        self.calls: list[tuple[str, list[str], dict[str, object]]] = []

    def get_sentence_embedding_dimension(self) -> int:
        return self.dimension

    def encode_query(self, sentences: list[str], **kwargs: object) -> list[list[float]]:
        self.calls.append(("query", sentences, kwargs))
        return self._vectors(sentences)

    def encode_document(self, sentences: list[str], **kwargs: object) -> list[list[float]]:
        self.calls.append(("document", sentences, kwargs))
        return self._vectors(sentences)

    def _vectors(self, sentences: list[str]) -> list[list[float]]:
        return [[float(index + 1), 0.5, -0.5] for index, _ in enumerate(sentences)]


def configured_model(
    alias: str,
    languages: list[str],
    *,
    load_strategy: str = "lazy",
    batch_size: int = 4,
) -> dict[str, Any]:
    return {
        "alias": alias,
        "model_id": f"owner/{alias}",
        "revision": "a" * 40,
        "languages": languages,
        "load_strategy": load_strategy,
        "batch_size": batch_size,
        "max_sequence_length": 512,
        "normalize_embeddings": True,
        "max_concurrency": 1,
    }


def write_config(tmp_path: Path, models: list[dict[str, Any]]) -> Path:
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"models": models}), encoding="utf-8")
    return path


@pytest.mark.asyncio
async def test_health_discovery_dispatch_ordering_and_limits(tmp_path: Path) -> None:
    fake_models: dict[str, FakeModel] = {}

    def factory(config: ModelConfig) -> FakeModel:
        model = FakeModel()
        model.max_seq_length = config.max_sequence_length
        fake_models[config.alias] = model
        return model

    path = write_config(
        tmp_path,
        [
            configured_model("multilingual", ["mul"], load_strategy="eager"),
            configured_model("portuguese", ["pt"]),
            configured_model("brazilian", ["pt_br"]),
            configured_model("english", ["en"]),
        ],
    )
    app = create_app(config_path=path, model_factory=factory)
    async with lifespan_client(app) as client:
        assert (await client.get("/health/live")).json() == {"status": "live"}
        assert (await client.get("/health/ready")).status_code == 200

        unfiltered = (await client.get("/v1/models")).json()["data"]
        assert [item["alias"] for item in unfiltered] == [
            "multilingual",
            "portuguese",
            "brazilian",
            "english",
        ]
        assert unfiltered[0]["status"] == "ready"
        assert unfiltered[0]["dimension"] == 3
        assert unfiltered[1]["status"] == "unloaded"
        assert unfiltered[1]["dimension"] is None
        assert unfiltered[2]["endpoint"] == "/v1/models/brazilian/embeddings"

        filtered = (await client.get("/v1/models", params={"locale": "pt_br"})).json()["data"]
        assert [(item["alias"], item["language_match"]) for item in filtered] == [
            ("brazilian", "exact"),
            ("portuguese", "base"),
            ("multilingual", "multilingual"),
        ]
        assert [
            item["alias"] for item in (await client.get("/v1/models?locale=mul")).json()["data"]
        ] == ["multilingual"]
        assert (await client.get("/v1/models?locale=de")).json()["data"][0][
            "alias"
        ] == "multilingual"
        assert (await client.get("/v1/models?locale=zz")).json()["data"][0][
            "alias"
        ] == "multilingual"
        assert (await client.get("/v1/models?locale=pt-BR")).status_code == 422

        response = await client.post(
            "/v1/models/english/embeddings",
            json={"input": ["first", "second"], "input_type": "query"},
        )
        assert response.status_code == 200
        assert response.json() == {
            "fingerprint": fingerprint(
                ModelConfig.model_validate(configured_model("english", ["en"]))
            ),
            "model": "english",
            "input_type": "query",
            "dimension": 3,
            "data": [
                {"index": 0, "embedding": [1.0, 0.5, -0.5]},
                {"index": 1, "embedding": [2.0, 0.5, -0.5]},
            ],
        }
        assert fake_models["english"].calls[0] == (
            "query",
            ["first", "second"],
            {
                "batch_size": 4,
                "normalize_embeddings": True,
                "convert_to_numpy": True,
                "show_progress_bar": False,
            },
        )
        assert fake_models["english"].max_seq_length == 512

        response = await client.post(
            "/v1/models/english/embeddings",
            json={"input": ["document"], "input_type": "document"},
        )
        assert response.status_code == 200
        assert fake_models["english"].calls[-1][0] == "document"

        too_large = await client.post(
            "/v1/models/english/embeddings",
            json={"input": ["1", "2", "3", "4", "5"], "input_type": "query"},
        )
        assert too_large.status_code == 413
        assert (
            await client.post(
                "/v1/models/missing/embeddings", json={"input": ["x"], "input_type": "query"}
            )
        ).status_code == 404


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"input": [], "input_type": "query"},
        {"input": [""], "input_type": "query"},
        {"input": ["x"], "input_type": "search"},
        {"input": ["x"], "input_type": "query", "locale": "en"},
    ],
)
@pytest.mark.asyncio
async def test_invalid_payloads_return_422(tmp_path: Path, payload: dict[str, object]) -> None:
    path = write_config(tmp_path, [configured_model("model", ["en"])])
    app = create_app(config_path=path, model_factory=lambda _: FakeModel())
    async with lifespan_client(app) as client:
        assert (await client.post("/v1/models/model/embeddings", json=payload)).status_code == 422


@pytest.mark.asyncio
async def test_no_locale_match_is_empty(tmp_path: Path) -> None:
    path = write_config(tmp_path, [configured_model("english", ["en"])])
    app = create_app(config_path=path, model_factory=lambda _: FakeModel())
    async with lifespan_client(app) as client:
        response = await client.get("/v1/models?locale=de")
        assert response.status_code == 200
        assert response.json() == {"data": []}


@pytest.mark.asyncio
async def test_readiness_is_unavailable_before_startup() -> None:
    config = ServiceConfig.model_validate({"models": [configured_model("model", ["en"])]})
    registry = ModelRegistry(config, model_factory=lambda _: FakeModel())
    app = create_app()
    app.state.registry = registry
    transport = httpx.ASGITransport(app=app)
    try:
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/health/ready")
        assert response.status_code == 503
        assert response.json() == {"status": "not_ready"}
    finally:
        await registry.shutdown()


@pytest.mark.asyncio
async def test_eager_load_failure_aborts_startup(tmp_path: Path) -> None:
    path = write_config(tmp_path, [configured_model("eager", ["en"], load_strategy="eager")])

    def failing_factory(_: ModelConfig) -> FakeModel:
        raise OSError("unavailable")

    with pytest.raises(RuntimeError, match="could not be loaded"):
        app = create_app(config_path=path, model_factory=failing_factory)
        async with app.router.lifespan_context(app):
            pass


@pytest.mark.asyncio
async def test_lazy_load_failure_is_retryable(tmp_path: Path) -> None:
    attempts = 0

    def flaky_factory(_: ModelConfig) -> FakeModel:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("temporary")
        return FakeModel()

    path = write_config(tmp_path, [configured_model("lazy", ["en"])])
    app = create_app(config_path=path, model_factory=flaky_factory)
    async with lifespan_client(app) as client:
        payload = {"input": ["x"], "input_type": "query"}
        assert (await client.post("/v1/models/lazy/embeddings", json=payload)).status_code == 503
        failed = (await client.get("/v1/models")).json()["data"][0]
        assert failed["status"] == "failed"
        assert (await client.post("/v1/models/lazy/embeddings", json=payload)).status_code == 200
        assert attempts == 2


@pytest.mark.asyncio
async def test_inference_failure_returns_503(tmp_path: Path) -> None:
    class FailingModel(FakeModel):
        def encode_query(self, sentences: list[str], **kwargs: object) -> list[list[float]]:
            raise RuntimeError("inference failed")

    path = write_config(tmp_path, [configured_model("model", ["en"])])
    app = create_app(config_path=path, model_factory=lambda _: FailingModel())
    async with lifespan_client(app) as client:
        response = await client.post(
            "/v1/models/model/embeddings",
            json={"input": ["private input must not be logged"], "input_type": "query"},
        )
    assert response.status_code == 503


@pytest.mark.asyncio
async def test_concurrent_first_requests_load_once(tmp_path: Path) -> None:
    loads = 0
    load_lock = threading.Lock()

    def slow_factory(_: ModelConfig) -> FakeModel:
        nonlocal loads
        with load_lock:
            loads += 1
        time.sleep(0.05)
        return FakeModel()

    path = write_config(tmp_path, [configured_model("lazy", ["en"])])
    app = create_app(config_path=path, model_factory=slow_factory)
    async with lifespan_client(app) as client:
        payload = {"input": ["x"], "input_type": "query"}
        first, second = await asyncio.gather(
            client.post("/v1/models/lazy/embeddings", json=payload),
            client.post("/v1/models/lazy/embeddings", json=payload),
        )
    assert first.status_code == second.status_code == 200
    assert loads == 1


@asynccontextmanager
async def lifespan_client(app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            yield client
