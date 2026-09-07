from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from autoexam_embeddings.config import ServiceConfig, is_valid_locale, load_config


def model_config(**overrides: object) -> dict[str, object]:
    config: dict[str, object] = {
        "alias": "model-one",
        "model_id": "owner/model-one",
        "revision": "a" * 40,
        "languages": ["en"],
        "load_strategy": "lazy",
        "batch_size": 8,
        "max_sequence_length": 512,
        "normalize_embeddings": True,
        "max_concurrency": 2,
    }
    config.update(overrides)
    return config


@pytest.mark.parametrize("locale", ["en", "el", "de", "fr", "pt_br", "abc_xy"])
def test_valid_content_locales(locale: str) -> None:
    assert is_valid_locale(locale)


@pytest.mark.parametrize("locale", ["mul", "EN", "pt-BR", "e", "engl", "pt_bra", "../en"])
def test_invalid_content_locales(locale: str) -> None:
    assert not is_valid_locale(locale)


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"revision": ""}, "revision"),
        ({"revision": "main"}, "revision"),
        ({"alias": "Bad Alias"}, "alias"),
        ({"model_id": "/tmp/model"}, "model_id"),
        ({"model_id": "https://example.test/model"}, "model_id"),
        ({"languages": ["en-US"]}, "language"),
        ({"languages": ["en", "en"]}, "unique"),
        ({"languages": ["mul", "en"]}, "only"),
        ({"batch_size": 0}, "greater than 0"),
        ({"max_sequence_length": -1}, "greater than 0"),
        ({"max_concurrency": 0}, "greater than 0"),
    ],
)
def test_rejects_invalid_model_configuration(override: dict[str, object], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        ServiceConfig.model_validate({"models": [model_config(**override)]})


def test_normalization_defaults_to_true() -> None:
    raw = model_config()
    del raw["normalize_embeddings"]
    config = ServiceConfig.model_validate({"models": [raw]})
    assert config.models[0].normalize_embeddings is True


def test_rejects_empty_models_and_duplicate_aliases() -> None:
    with pytest.raises(ValidationError, match="at least 1"):
        ServiceConfig.model_validate({"models": []})
    with pytest.raises(ValidationError, match="unique"):
        ServiceConfig.model_validate({"models": [model_config(), model_config()]})


def test_load_config_reads_json(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"models": [model_config()]}), encoding="utf-8")
    assert load_config(path).models[0].alias == "model-one"
