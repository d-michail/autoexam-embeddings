from __future__ import annotations

from typing import Any

import pytest

from autoexam_embeddings import registry as registry_module
from autoexam_embeddings.config import ModelConfig, ServiceConfig
from autoexam_embeddings.registry import ModelRegistry, ModelUnavailableError, default_model_factory


def make_model_config(**overrides: object) -> ModelConfig:
    config: dict[str, object] = {
        "alias": "model-one",
        "model_id": "owner/model-one",
        "revision": "a" * 40,
        "languages": ["en"],
        "load_strategy": "lazy",
        "batch_size": 4,
        "max_sequence_length": 512,
        "normalize_embeddings": True,
        "max_concurrency": 1,
    }
    config.update(overrides)
    return ModelConfig.model_validate(config)


def test_default_model_factory_forwards_configured_device(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    class FakeSentenceTransformer:
        def __init__(
            self, model_id: str, *, revision: str, device: str, trust_remote_code: bool
        ) -> None:
            captured["device"] = device
            self.max_seq_length = 0

    monkeypatch.setattr("sentence_transformers.SentenceTransformer", FakeSentenceTransformer)

    default_model_factory(make_model_config(), device="cuda")

    assert captured["device"] == "cuda"


def test_model_registry_builds_default_factory_with_configured_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(registry_module.torch.cuda, "is_available", lambda: True)
    config = ServiceConfig.model_validate(
        {"models": [make_model_config().model_dump()], "device": "cuda"}
    )

    registry = ModelRegistry(config)

    assert registry._model_factory.func is default_model_factory  # type: ignore[attr-defined]
    assert registry._model_factory.keywords == {"device": "cuda"}  # type: ignore[attr-defined]


def test_model_registry_rejects_cuda_without_a_visible_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(registry_module.torch.cuda, "is_available", lambda: False)
    config = ServiceConfig.model_validate(
        {"models": [make_model_config().model_dump()], "device": "cuda"}
    )

    with pytest.raises(ModelUnavailableError, match="cuda"):
        ModelRegistry(config)


def test_model_registry_skips_cpu_thread_tuning_for_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(registry_module.torch.cuda, "is_available", lambda: True)
    calls: list[int] = []
    monkeypatch.setattr(registry_module.torch, "set_num_threads", calls.append)
    config = ServiceConfig.model_validate(
        {"models": [make_model_config().model_dump()], "device": "cuda"}
    )

    ModelRegistry(config)

    assert calls == []


def test_model_registry_tunes_cpu_threads_for_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []
    monkeypatch.setattr(registry_module.torch, "set_num_threads", calls.append)
    config = ServiceConfig.model_validate(
        {"models": [make_model_config().model_dump()], "cpu_limit": 4}
    )

    ModelRegistry(config)

    assert calls == [4]
