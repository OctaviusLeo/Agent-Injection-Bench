#!/usr/bin/env bash
# Start or stop this project's llama.cpp server (Compose project "aib", service "llm").
# Usage: bash scripts/serve.sh up <model key> | down
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
exec uv run --frozen python -m aib.server "$@"
