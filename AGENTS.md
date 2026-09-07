# autoexam-embeddings

Python/FastAPI service for internal dense Sentence Transformers inference.

## Boundaries

- Keep the service independent from the AutoExam backend and database.
- Expose dense float embeddings only. Sparse vectors, ColBERT, reranking, Ray,
  GPUs, quantization, and private or gated Hugging Face repositories are out of
  scope.
- Do not add application authentication or public routing. Kubernetes namespace
  isolation and the ClusterIP service are the trust boundary.
- Never log input text or embedding vectors.
- Keep `trust_remote_code=False` and require immutable Hugging Face commit
  revisions in configuration.

## Development

Dependencies are managed by Poetry from `pyproject.toml` and `poetry.lock`.

```bash
make install
make lint
make typecheck
make test
make integration-test  # downloads the pinned MiniLM model when enabled
make container-smoke    # builds and exercises the production image
```

Set `RUN_MODEL_INTEGRATION=1` to permit the integration test to download a
model. Without it, the test uses only the local Hugging Face cache and skips
when the pinned model is unavailable.

## Implementation notes

- Configuration is loaded once during application startup. Configuration
  changes take effect after a pod restart.
- Keep exactly one Uvicorn worker per container so model memory is not
  duplicated.
- Run model loading and inference in worker threads, serialize lazy model
  initialization, and retain each model's concurrency semaphore.
- Add tests for configuration, discovery ordering, model state transitions,
  request limits, and query/document dispatch with any behavior change.
