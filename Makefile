.DEFAULT_GOAL := test

.PHONY: install lock format lint typecheck test integration-test run container-smoke

install:
	poetry install

lock:
	poetry lock --no-update

format:
	poetry run ruff format .
	poetry run ruff check --fix .

lint:
	poetry run ruff format --check .
	poetry run ruff check .

typecheck:
	poetry run mypy

test:
	poetry run pytest -m "not integration"

integration-test:
	poetry run pytest -m integration

run:
	EMBEDDINGS_CONFIG_PATH=$${EMBEDDINGS_CONFIG_PATH:-examples/config.json} poetry run uvicorn autoexam_embeddings.app:app --host 0.0.0.0 --port 8000 --workers 1

container-smoke:
	./scripts/container-smoke.sh
