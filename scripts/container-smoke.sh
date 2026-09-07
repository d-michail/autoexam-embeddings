#!/usr/bin/env bash
set -euo pipefail

image="${EMBEDDINGS_IMAGE:-autoexam-embeddings:smoke}"
container="autoexam-embeddings-smoke-$$"
config="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/examples/config.json"

cleanup() {
  docker rm -f "$container" >/dev/null 2>&1 || true
}
trap cleanup EXIT

docker build -t "$image" "$(dirname "$config")/.."
docker run -d --name "$container" -p 127.0.0.1::8000 \
  -v "$config:/etc/autoexam-embeddings/config.json:ro" "$image" >/dev/null
port="$(docker port "$container" 8000/tcp | sed 's/.*://')"

for _ in $(seq 1 180); do
  if curl --fail --silent "http://127.0.0.1:${port}/health/ready" >/dev/null; then
    break
  fi
  if ! docker inspect -f '{{.State.Running}}' "$container" | grep -q true; then
    docker logs "$container"
    exit 1
  fi
  sleep 2
done

curl --fail --silent "http://127.0.0.1:${port}/v1/models" >/dev/null
for alias in bge-m3 all-minilm-l6-v2; do
  curl --fail --silent -H 'Content-Type: application/json' \
    -d '{"input":["AutoExam container smoke test"],"input_type":"query"}' \
    "http://127.0.0.1:${port}/v1/models/${alias}/embeddings" >/dev/null
done

echo "Container smoke test passed for both configured models."
