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
    fingerprint: str
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


class TokenLimits(BaseModel):
    document: int
    query: int
    fingerprint: str
    batch_size: int


class ChunkRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1, max_length=65536)
    chunk_size_tokens: int | None = Field(default=None, gt=0)
    chunk_overlap_tokens: int | None = Field(default=None, ge=0)


class ChunkItem(BaseModel):
    text: str
    start: int
    end: int
    token_count: int


class ChunkResponse(BaseModel):
    fingerprint: str
    chunks: list[ChunkItem]
