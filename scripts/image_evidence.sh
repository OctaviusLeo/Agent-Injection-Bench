#!/usr/bin/env bash
# Record the pinned llama.cpp server image: repo digest, image id, `llama-server --version`, and `--help`.
# Usage: bash scripts/image_evidence.sh   (writes docs/evidence/llama_image.txt and llama_server_help.txt)
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
image="ghcr.io/ggml-org/llama.cpp:server-cuda12-b11580"
out=docs/evidence
mkdir -p "$out"
{
  echo "image: $image"
  echo "repo_digest: $(docker image inspect --format '{{index .RepoDigests 0}}' "$image")"
  echo "image_id: $(docker image inspect --format '{{.Id}}' "$image")"
  echo "created: $(docker image inspect --format '{{.Created}}' "$image")"
  echo "entrypoint: $(docker image inspect --format '{{json .Config.Entrypoint}}' "$image")"
  echo "recorded: $(date -u +%Y-%m-%dT%H:%M:%SZ) with docker $(docker version --format '{{.Server.Version}}')"
  echo "--- llama-server --version"
  docker run --rm --gpus all --entrypoint /app/llama-server "$image" --version 2>&1
} > "$out/llama_image.txt"
docker run --rm --gpus all --entrypoint /app/llama-server "$image" --help > "$out/llama_server_help.txt" 2>&1
cat "$out/llama_image.txt"
