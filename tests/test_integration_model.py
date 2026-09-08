from __future__ import annotations

import os

import numpy as np
import pytest
from sentence_transformers import SentenceTransformer

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "b8903db39f65d93ae28d49a37c4f3fa90c5f94e0"


@pytest.mark.integration
def test_pinned_minilm_produces_normalized_finite_stable_vectors() -> None:
    permit_download = os.environ.get("RUN_MODEL_INTEGRATION") == "1"
    try:
        model = SentenceTransformer(
            MODEL_ID,
            revision=MODEL_REVISION,
            device="cpu",
            trust_remote_code=False,
            local_files_only=not permit_download,
        )
    except Exception as error:
        pytest.skip(
            f"pinned model is unavailable from the configured cache/network: {type(error).__name__}"
        )

    query = model.encode_query(["What is formative assessment?"], normalize_embeddings=True)
    document = model.encode_document(
        ["Formative assessment provides feedback during learning."], normalize_embeddings=True
    )

    assert query.shape == document.shape == (1, 384)
    assert np.isfinite(query).all() and np.isfinite(document).all()
    assert np.linalg.norm(query[0]) == pytest.approx(1.0, abs=1e-5)
    assert np.linalg.norm(document[0]) == pytest.approx(1.0, abs=1e-5)


@pytest.mark.integration
def test_pinned_minilm_chunking_respects_context_and_unicode() -> None:
    from autoexam_embeddings.chunking import chunk, limits, token_count
    from autoexam_embeddings.config import ModelConfig
    from autoexam_embeddings.schemas import ChunkRequest

    try:
        model = SentenceTransformer(
            MODEL_ID,
            revision=MODEL_REVISION,
            device="cpu",
            trust_remote_code=False,
            local_files_only=os.environ.get("RUN_MODEL_INTEGRATION") != "1",
        )
    except Exception as error:
        pytest.skip(f"pinned model unavailable: {type(error).__name__}")
    config = ModelConfig.model_validate(
        {
            "alias": "minilm",
            "model_id": MODEL_ID,
            "revision": MODEL_REVISION,
            "languages": ["en"],
            "load_strategy": "lazy",
            "batch_size": 8,
            "max_sequence_length": 256,
            "normalize_embeddings": True,
            "max_concurrency": 1,
        }
    )
    text = "Assessment Ελληνικά 中文 🐟 feedback supports learning. " * 100
    result = chunk(
        model, config, ChunkRequest(text=text, chunk_size_tokens=64, chunk_overlap_tokens=8)
    )
    assert limits(model, config).document == 254
    assert len(result.chunks) > 1
    assert result.chunks[0].start == 0 and result.chunks[-1].end == len(text)
    for part in result.chunks:
        assert part.text == text[part.start : part.end]
        assert token_count(model, part.text, "document") <= 66
    vectors = model.encode_document(
        [part.text for part in result.chunks], normalize_embeddings=True
    )
    assert vectors.shape == (len(result.chunks), 384)
    assert np.isfinite(vectors).all()
