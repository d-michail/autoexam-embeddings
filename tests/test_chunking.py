from __future__ import annotations

from typing import Any, ClassVar

import pytest
from test_api import FakeModel, configured_model, lifespan_client, write_config

from autoexam_embeddings.app import create_app
from autoexam_embeddings.chunking import chunk, fingerprint, limits
from autoexam_embeddings.config import ModelConfig
from autoexam_embeddings.schemas import ChunkRequest


class CharacterTokenizer:
    def __call__(
        self, text: str, *, add_special_tokens: bool = True, **kwargs: Any
    ) -> dict[str, Any]:
        return {
            "input_ids": list(range(len(text) + (2 if add_special_tokens else 0))),
            "offset_mapping": [(i, i + 1) for i in range(len(text))],
        }


class TokenModel(FakeModel):
    tokenizer = CharacterTokenizer()
    prompts: ClassVar[dict[str, str]] = {"query": "Q: ", "document": "D: "}


def test_unicode_slices_overlap_and_prompt_budget() -> None:
    config = ModelConfig.model_validate(
        {**configured_model("test", ["mul"]), "max_sequence_length": 16}
    )
    model = TokenModel()
    assert limits(model, config).document == 11
    text = "Ελληνικά 中文 résumé 🐟 " * 3
    response = chunk(
        model, config, ChunkRequest(text=text, chunk_size_tokens=11, chunk_overlap_tokens=3)
    )
    assert response.fingerprint == fingerprint(config)
    assert response.chunks[0].start == 0
    assert response.chunks[-1].end == len(text)
    for i, part in enumerate(response.chunks):
        assert text[part.start : part.end] == part.text
        assert part.token_count <= 16
        if i:
            assert part.start < response.chunks[i - 1].end
    with pytest.raises(ValueError):
        chunk(model, config, ChunkRequest(text=text, chunk_size_tokens=12))
    with pytest.raises(ValueError):
        chunk(model, config, ChunkRequest(text=text, chunk_size_tokens=5, chunk_overlap_tokens=5))


def test_fingerprint_tracks_vector_configuration_not_batching() -> None:
    config = ModelConfig.model_validate(configured_model("test", ["mul"]))
    assert fingerprint(config) == fingerprint(config.model_copy(update={"batch_size": 1}))
    assert fingerprint(config) != fingerprint(config.model_copy(update={"revision": "b" * 40}))
    assert fingerprint(config) != fingerprint(
        config.model_copy(update={"normalize_embeddings": False})
    )


@pytest.mark.asyncio
async def test_chunk_and_limits_api_and_no_silent_truncation(tmp_path: Any) -> None:
    app = create_app(
        config_path=write_config(tmp_path, [configured_model("test", ["mul"])]),
        model_factory=lambda _: TokenModel(),
    )
    async with lifespan_client(app) as client:
        result = await client.get("/v1/models/test/limits")
        assert result.status_code == 200
        assert result.json()["document"] == 507
        result = await client.post(
            "/v1/models/test/chunks",
            json={"text": "Hello world", "chunk_size_tokens": 5, "chunk_overlap_tokens": 1},
        )
        assert result.status_code == 200
        assert len(result.json()["chunks"]) > 1
        assert (
            await client.post(
                "/v1/models/test/chunks", json={"text": "x", "chunk_size_tokens": 999}
            )
        ).status_code == 422
        assert (await client.get("/v1/models/missing/limits")).status_code == 404
        assert (
            await client.post(
                "/v1/models/test/embeddings", json={"input": ["x" * 508], "input_type": "query"}
            )
        ).status_code == 413
