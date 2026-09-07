"""Strict JSON configuration for the embeddings service."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, PositiveInt, field_validator, model_validator

LOCALE_PATTERN = re.compile(r"^[a-z]{2,3}(?:_[a-z]{2})?$")
ALIAS_PATTERN = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$")
MODEL_ID_PATTERN = re.compile(
    r"^[A-Za-z0-9](?:[A-Za-z0-9._-]{0,94}[A-Za-z0-9])?/[A-Za-z0-9](?:[A-Za-z0-9._-]{0,94}[A-Za-z0-9])?$"
)
REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")


def is_valid_locale(value: str, *, allow_multilingual: bool = False) -> bool:
    """Return whether a locale uses AutoExam's content-locale syntax."""

    if value == "mul":
        return allow_multilingual
    return LOCALE_PATTERN.fullmatch(value) is not None


class ModelConfig(BaseModel):
    """Configuration for one public Hugging Face Sentence Transformer model."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    alias: str
    model_id: str
    revision: str
    languages: list[str] = Field(min_length=1)
    load_strategy: Literal["eager", "lazy"]
    batch_size: PositiveInt
    max_sequence_length: PositiveInt
    normalize_embeddings: bool = True
    max_concurrency: PositiveInt

    @field_validator("alias")
    @classmethod
    def validate_alias(cls, value: str) -> str:
        if ALIAS_PATTERN.fullmatch(value) is None:
            raise ValueError("alias must contain lowercase letters, digits, or interior hyphens")
        return value

    @field_validator("model_id")
    @classmethod
    def validate_model_id(cls, value: str) -> str:
        if MODEL_ID_PATTERN.fullmatch(value) is None:
            raise ValueError("model_id must be a public Hugging Face owner/repository identifier")
        return value

    @field_validator("revision")
    @classmethod
    def validate_revision(cls, value: str) -> str:
        if REVISION_PATTERN.fullmatch(value) is None:
            raise ValueError("revision must be an immutable 40-character lowercase commit hash")
        return value

    @field_validator("languages")
    @classmethod
    def validate_languages(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("language hints must be unique")
        invalid = [
            language for language in value if not is_valid_locale(language, allow_multilingual=True)
        ]
        if invalid:
            raise ValueError(f"invalid language hints: {', '.join(invalid)}")
        if "mul" in value and len(value) != 1:
            raise ValueError("mul must be the only language hint")
        return value


class ServiceConfig(BaseModel):
    """Complete startup configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    models: list[ModelConfig] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_aliases(self) -> ServiceConfig:
        aliases = [model.alias for model in self.models]
        if len(aliases) != len(set(aliases)):
            raise ValueError("model aliases must be unique")
        return self


def load_config(path: str | Path) -> ServiceConfig:
    """Load and validate a service configuration file."""

    config_path = Path(path)
    with config_path.open(encoding="utf-8") as config_file:
        raw = json.load(config_file)
    return ServiceConfig.model_validate(raw)
