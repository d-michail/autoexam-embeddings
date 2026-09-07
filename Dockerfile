FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    HF_HOME=/var/cache/huggingface \
    EMBEDDINGS_CONFIG_PATH=/etc/autoexam-embeddings/config.json

WORKDIR /app

RUN pip install --no-cache-dir poetry==1.8.2 \
    && groupadd --gid 10001 autoexam \
    && useradd --uid 10001 --gid autoexam --no-create-home --shell /usr/sbin/nologin autoexam \
    && mkdir -p /var/cache/huggingface /etc/autoexam-embeddings \
    && chown -R autoexam:autoexam /var/cache/huggingface /etc/autoexam-embeddings

COPY pyproject.toml poetry.lock ./
RUN poetry install --only main --no-root --no-interaction --no-ansi

COPY src ./src
ENV PYTHONPATH=/app/src

USER 10001:10001
EXPOSE 8000

STOPSIGNAL SIGTERM
CMD ["uvicorn", "autoexam_embeddings.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
