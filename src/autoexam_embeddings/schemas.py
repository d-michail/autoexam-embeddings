"""HTTP request and response schemas."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

NonEmptyText = Annotated[str, StringConstraints(min_length=1)]


class EmbeddingsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input: list[NonEmptyText] = Field(min_length=1)
    input_type: Literal["query", "document"]


class EmbeddingItem(BaseModel):
    index: int
    embedding: list[float]


class EmbeddingsResponse(BaseModel):
    model: str
    input_type: Literal["query", "document"]
    dimension: int
    data: list[EmbeddingItem]


class ModelDescription(BaseModel):
    alias: str
    model_id: str
    languages: list[str]
    endpoint: str
    load_strategy: Literal["eager", "lazy"]
    status: Literal["unloaded", "loading", "ready", "failed"]
    dimension: int | None
    language_match: Literal["exact", "base", "multilingual"] | None = None


class ModelsResponse(BaseModel):
    data: list[ModelDescription]
