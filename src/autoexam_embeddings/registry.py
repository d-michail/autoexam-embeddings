"""Concurrent model lifecycle and inference management."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Literal, Protocol, TypeVar, cast

from . import chunking
from .config import ModelConfig, ServiceConfig
from .schemas import (
    ChunkRequest,
    ChunkResponse,
    EmbeddingItem,
    EmbeddingsResponse,
    ModelDescription,
    TokenLimits,
)

logger = logging.getLogger("autoexam_embeddings")


class EmbeddingModel(Protocol):
    max_seq_length: int

    def get_sentence_embedding_dimension(self) -> int | None: ...

    def encode_query(self, sentences: list[str], **kwargs: object) -> Any: ...

    def encode_document(self, sentences: list[str], **kwargs: object) -> Any: ...


ModelFactory = Callable[[ModelConfig], EmbeddingModel]
LoadStatus = Literal["unloaded", "loading", "ready", "failed"]
WorkerResult = TypeVar("WorkerResult")


class ModelUnavailableError(RuntimeError):
    """Raised when a configured model cannot be loaded or used."""


class RequestLimitError(ValueError):
    """Raised when a request exceeds its configured model batch limit."""


def _log_event(event: str, **fields: object) -> None:
    logger.info(json.dumps({"event": event, **fields}, sort_keys=True, separators=(",", ":")))


def default_model_factory(config: ModelConfig) -> EmbeddingModel:
    """Load one CPU model without executing repository-provided Python code."""

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(
        config.model_id,
        revision=config.revision,
        device="cpu",
        trust_remote_code=False,
    )
    model.max_seq_length = config.max_sequence_length
    return cast(EmbeddingModel, model)


@dataclass
class ModelRuntime:
    config: ModelConfig
    model: EmbeddingModel | None = None
    status: LoadStatus = "unloaded"
    dimension: int | None = None
    load_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    semaphore: asyncio.Semaphore = field(init=False)

    def __post_init__(self) -> None:
        self.semaphore = asyncio.Semaphore(self.config.max_concurrency)


class ModelRegistry:
    """Ordered registry of configured model runtimes."""

    def __init__(self, config: ServiceConfig, model_factory: ModelFactory = default_model_factory):
        self._runtimes = {model.alias: ModelRuntime(model) for model in config.models}
        self._model_factory = model_factory
        worker_count = sum(model.max_concurrency for model in config.models)
        self._executor = ThreadPoolExecutor(
            max_workers=worker_count,
            thread_name_prefix="embedding-worker",
        )
        self.ready = False

    async def start(self) -> None:
        """Load all eager models before declaring the registry ready."""

        for runtime in self._runtimes.values():
            if runtime.config.load_strategy == "eager":
                await self._get_or_load(runtime)
        self.ready = True

    async def shutdown(self) -> None:
        """Stop accepting ready traffic and release model references."""

        self.ready = False
        for runtime in self._runtimes.values():
            runtime.model = None
            runtime.dimension = None
            runtime.status = "unloaded"
        self._executor.shutdown(wait=True, cancel_futures=True)

    def get(self, alias: str) -> ModelRuntime | None:
        return self._runtimes.get(alias)

    def describe(self, locale: str | None = None) -> list[ModelDescription]:
        descriptions = [self._describe_runtime(runtime) for runtime in self._runtimes.values()]
        if locale is None:
            return descriptions

        tiers: dict[str, int] = {"exact": 0, "base": 1, "multilingual": 2}
        matched: list[tuple[int, int, ModelDescription]] = []
        for position, description in enumerate(descriptions):
            language_match = self._language_match(description.languages, locale)
            if language_match is not None:
                description.language_match = language_match
                matched.append((tiers[language_match], position, description))
        return [description for _, _, description in sorted(matched)]

    async def embed(
        self, alias: str, inputs: list[str], input_type: Literal["query", "document"]
    ) -> EmbeddingsResponse:
        runtime = self._runtimes[alias]
        if len(inputs) > runtime.config.batch_size:
            raise RequestLimitError(
                f"input batch contains {len(inputs)} items; limit is {runtime.config.batch_size}"
            )

        model = await self._get_or_load(runtime)
        started_at = time.perf_counter()
        try:
            async with runtime.semaphore:
                vectors = await self._run_in_worker(
                    partial(self._encode, runtime.config, model, inputs, input_type)
                )
        except (ModelUnavailableError, RequestLimitError):
            raise
        except Exception as error:
            _log_event(
                "embedding_failed",
                model_alias=alias,
                batch_size=len(inputs),
                input_type=input_type,
                error_type=type(error).__name__,
            )
            raise ModelUnavailableError(f"model {alias!r} failed during inference") from error

        dimension = len(vectors[0])
        runtime.dimension = dimension
        elapsed_ms = round((time.perf_counter() - started_at) * 1000, 3)
        _log_event(
            "embedding_completed",
            model_alias=alias,
            batch_size=len(inputs),
            input_type=input_type,
            dimension=dimension,
            duration_ms=elapsed_ms,
            load_state=runtime.status,
        )
        return EmbeddingsResponse(
            fingerprint=chunking.fingerprint(runtime.config),
            model=alias,
            input_type=input_type,
            dimension=dimension,
            data=[
                EmbeddingItem(index=index, embedding=vector) for index, vector in enumerate(vectors)
            ],
        )

    async def token_limits(self, alias: str) -> TokenLimits:
        runtime = self._runtimes[alias]
        model = await self._get_or_load(runtime)
        async with runtime.semaphore:
            return await self._run_in_worker(partial(chunking.limits, model, runtime.config))

    async def chunk(self, alias: str, payload: ChunkRequest) -> ChunkResponse:
        runtime = self._runtimes[alias]
        model = await self._get_or_load(runtime)
        async with runtime.semaphore:
            return await self._run_in_worker(
                partial(chunking.chunk, model, runtime.config, payload)
            )

    async def _get_or_load(self, runtime: ModelRuntime) -> EmbeddingModel:
        if runtime.model is not None:
            return runtime.model

        async with runtime.load_lock:
            if runtime.model is not None:
                return runtime.model
            runtime.status = "loading"
            started_at = time.perf_counter()
            _log_event("model_loading", model_alias=runtime.config.alias, load_state=runtime.status)
            try:
                model = await self._run_in_worker(partial(self._model_factory, runtime.config))
                dimension = model.get_sentence_embedding_dimension()
                if dimension is not None and dimension <= 0:
                    raise ValueError("model reported a non-positive embedding dimension")
            except Exception as error:
                runtime.status = "failed"
                runtime.model = None
                runtime.dimension = None
                _log_event(
                    "model_load_failed",
                    model_alias=runtime.config.alias,
                    duration_ms=round((time.perf_counter() - started_at) * 1000, 3),
                    load_state=runtime.status,
                    error_type=type(error).__name__,
                )
                raise ModelUnavailableError(
                    f"model {runtime.config.alias!r} could not be loaded"
                ) from error

            runtime.model = model
            runtime.dimension = dimension
            runtime.status = "ready"
            _log_event(
                "model_loaded",
                model_alias=runtime.config.alias,
                dimension=dimension,
                duration_ms=round((time.perf_counter() - started_at) * 1000, 3),
                load_state=runtime.status,
            )
            return model

    @staticmethod
    def _encode(
        config: ModelConfig,
        model: EmbeddingModel,
        inputs: list[str],
        input_type: Literal["query", "document"],
    ) -> list[list[float]]:
        if hasattr(model, "tokenizer"):
            if any(
                chunking.token_count(model, text, input_type) > config.max_sequence_length
                for text in inputs
            ):
                raise RequestLimitError("input exceeds model token limit")
        encode = model.encode_query if input_type == "query" else model.encode_document
        raw_vectors = encode(
            inputs,
            batch_size=config.batch_size,
            normalize_embeddings=config.normalize_embeddings,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        if hasattr(raw_vectors, "tolist"):
            raw_vectors = raw_vectors.tolist()
        if not isinstance(raw_vectors, Sequence) or len(raw_vectors) != len(inputs):
            raise ModelUnavailableError("model returned an unexpected number of embeddings")

        vectors: list[list[float]] = []
        dimension: int | None = None
        for raw_vector in raw_vectors:
            if not isinstance(raw_vector, Sequence) or isinstance(raw_vector, str | bytes):
                raise ModelUnavailableError("model returned a non-vector embedding")
            vector = [float(value) for value in raw_vector]
            if not vector or not all(math.isfinite(value) for value in vector):
                raise ModelUnavailableError("model returned an empty or non-finite embedding")
            if dimension is None:
                dimension = len(vector)
            elif len(vector) != dimension:
                raise ModelUnavailableError("model returned inconsistent embedding dimensions")
            vectors.append(vector)
        return vectors

    async def _run_in_worker(self, function: Callable[[], WorkerResult]) -> WorkerResult:
        """Run blocking work in the owned executor without using the global executor."""

        future = self._executor.submit(function)
        try:
            # Polling avoids coupling the owned executor to asyncio's global
            # executor lifecycle while still yielding the event loop.
            while not future.done():  # noqa: ASYNC110
                await asyncio.sleep(0.005)
            return future.result()
        except BaseException:
            future.cancel()
            raise

    @staticmethod
    def _language_match(
        languages: list[str], locale: str
    ) -> Literal["exact", "base", "multilingual"] | None:
        if locale == "mul":
            return "multilingual" if languages == ["mul"] else None
        if locale in languages:
            return "exact"
        if "_" in locale and locale.split("_", maxsplit=1)[0] in languages:
            return "base"
        if languages == ["mul"]:
            return "multilingual"
        return None

    @staticmethod
    def _describe_runtime(runtime: ModelRuntime) -> ModelDescription:
        return ModelDescription(
            alias=runtime.config.alias,
            model_id=runtime.config.model_id,
            languages=list(runtime.config.languages),
            endpoint=f"/v1/models/{runtime.config.alias}/embeddings",
            load_strategy=runtime.config.load_strategy,
            status=runtime.status,
            dimension=runtime.dimension,
        )
