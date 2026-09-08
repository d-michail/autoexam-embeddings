# AutoExam Embeddings

Internal CPU-only FastAPI service for normalized dense embeddings from
operator-configured public Sentence Transformers models. The Kubernetes service
is reachable only inside the namespace at `http://autoexam-embeddings-service`;
there is no ingress or application-level authentication.

BGE-M3 is exposed through its 1024-dimensional dense Sentence Transformers
output. Sparse, ColBERT, and reranking outputs are intentionally not exposed.

## Configuration

Set `EMBEDDINGS_CONFIG_PATH` to a mounted JSON file. Every model must use a
public Hugging Face `owner/repository` identifier and an immutable 40-character
commit revision. Remote model code is never trusted. See
[`examples/config.json`](examples/config.json) for the development configuration.

Each configured model declares an alias, language discovery hints, eager or lazy
loading, request/inference batch size, maximum sequence length, normalization,
and maximum concurrent inference calls. `mul` is reserved for multilingual
models and cannot be combined with another language hint. Language hints aid
discovery only; embedding routes neither accept nor enforce a locale.

Configuration is read at process startup. In Kubernetes, ConfigMap reloader
restarts the pod after the mounted configuration changes.

## API

```bash
curl http://localhost:8000/v1/models
curl 'http://localhost:8000/v1/models?locale=pt_br'
curl -H 'Content-Type: application/json' \
  -d '{"input":["How does photosynthesis work?"],"input_type":"query"}' \
  http://localhost:8000/v1/models/bge-m3/embeddings
curl -H 'Content-Type: application/json' \
  -d '{"input":["Plants convert light into chemical energy."],"input_type":"document"}' \
  http://localhost:8000/v1/models/bge-m3/embeddings
```

`input_type` is required. Query requests use `encode_query()` and document
requests use `encode_document()`, preserving model-specific retrieval prompts
and routing. Responses preserve input order and include the output dimension.

Discovery filtering ranks exact locale support first, regional-to-base support
second, and multilingual support last while retaining JSON configuration order
inside each tier. The first item is the recommended model; callers remain in
control of route selection.

Errors use `404` for unknown aliases, `422` for malformed payloads/locales,
`413` when the input count exceeds the configured batch limit, and `503` when a
model cannot load or infer. Failed lazy loads can be retried by a later request.

## Development

Python 3.10 or newer and Poetry 1.8 are required.

```bash
make install
make lint
make typecheck
make test
EMBEDDINGS_CONFIG_PATH=examples/config.json make run
```

`make integration-test` checks the pinned MiniLM model. Set
`RUN_MODEL_INTEGRATION=1` to allow a download; otherwise the test skips when the
revision is absent from the local Hugging Face cache. `make container-smoke`
builds the production image, mounts the development JSON, waits for eager BGE-M3
readiness, and exercises both configured routes, so it requires substantial
download time, memory, and cache space.

The container runs as UID/GID 10001, uses one Uvicorn worker, stores Hugging Face
artifacts under `/var/cache/huggingface`, and handles `SIGTERM` through Uvicorn's
normal graceful shutdown.

## RAG tokenizer API

`GET /v1/models/{alias}/limits` lazily loads the selected model and returns
`document`/`query` content token budgets (after prompts and special tokens),
`batch_size`, and an immutable-configuration `fingerprint`.

`POST /v1/models/{alias}/chunks` accepts `text` (1–65,536 Unicode code points),
optional `chunk_size_tokens`, and optional `chunk_overlap_tokens`. Defaults are
`min(512, document_budget)` and `min(64, size // 8)`. The response contains
`fingerprint` and `chunks`, each with `text`, `start`, `end`, and `token_count`.
Offsets reference the original input's Unicode code points; token counts include
prompts and special tokens. Text slices preserve the source rather than decoding
token IDs. These endpoints use the model's existing worker pool, lazy-load lock
and concurrency semaphore. A fast tokenizer with offset mappings is required.

Embedding responses also include `fingerprint`. It covers the pinned model ID,
revision and vector-affecting configuration, excluding deployment batching,
concurrency and discovery metadata. Embedding inputs exceeding the configured
context are rejected with HTTP 413, avoiding silent truncation. Applications
should compare fingerprints across chunking, embedding and stored collections.
