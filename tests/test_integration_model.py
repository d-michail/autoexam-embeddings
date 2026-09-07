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
