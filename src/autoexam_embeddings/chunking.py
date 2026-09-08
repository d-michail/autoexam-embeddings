"""Tokenizer-aware source slices; offsets are Unicode code-point offsets."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from .config import ModelConfig
from .schemas import ChunkItem, ChunkRequest, ChunkResponse, TokenLimits


def fingerprint(config: ModelConfig) -> str:
    identity = config.model_dump(
        exclude={"alias", "languages", "load_strategy", "batch_size", "max_concurrency"}
    )
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def prompt(model: Any, kind: str) -> str:
    prompts = getattr(model, "prompts", {})
    names = ["query"] if kind == "query" else ["document", "passage", "corpus"]
    for name in names:
        if name in prompts:
            return str(prompts[name])
    return str(prompts.get(getattr(model, "default_prompt_name", None), ""))


def token_count(model: Any, text: str, kind: str) -> int:
    return len(
        model.tokenizer(prompt(model, kind) + text, add_special_tokens=True, truncation=False)[
            "input_ids"
        ]
    )


def limits(model: Any, config: ModelConfig) -> TokenLimits:
    return TokenLimits(
        document=config.max_sequence_length - token_count(model, "", "document"),
        query=config.max_sequence_length - token_count(model, "", "query"),
        fingerprint=fingerprint(config),
        batch_size=config.batch_size,
    )


def chunk(model: Any, config: ModelConfig, payload: ChunkRequest) -> ChunkResponse:
    available = limits(model, config)
    size = payload.chunk_size_tokens or min(512, available.document)
    overlap = payload.chunk_overlap_tokens
    if overlap is None:
        overlap = min(64, size // 8)
    if size <= 0 or size > available.document or overlap >= size:
        raise ValueError("chunk size must fit the model; overlap must be smaller than size")
    text = payload.text
    # Fast tokenizer offsets retain the original Unicode text, unlike decoding token IDs.
    encoded = model.tokenizer(
        text, add_special_tokens=False, truncation=False, return_offsets_mapping=True
    )
    offsets = encoded["offset_mapping"]
    chunks: list[ChunkItem] = []
    start = 0
    while start < len(offsets):
        end = min(start + size, len(offsets))
        lo = 0 if start == 0 else offsets[start][0]
        hi = len(text) if end == len(offsets) else offsets[end][0]
        # Retokenization at a slice boundary and prompt concatenation may add tokens.
        while token_count(model, text[lo:hi], "document") > min(
            config.max_sequence_length, size + token_count(model, "", "document")
        ):
            end -= 1
            if end <= start:
                raise ValueError("model context cannot fit a source character")
            hi = offsets[end][0]
        if hi > lo and text[lo:hi].strip():
            chunks.append(
                ChunkItem(
                    text=text[lo:hi],
                    start=lo,
                    end=hi,
                    token_count=token_count(model, text[lo:hi], "document"),
                )
            )
        if end == len(offsets):
            break
        start = max(start + 1, end - overlap)
    return ChunkResponse(fingerprint=available.fingerprint, chunks=chunks)
