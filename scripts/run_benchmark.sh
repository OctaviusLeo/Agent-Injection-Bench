#!/usr/bin/env bash
# Run one model's benchmark grid: commit check, model server up (local models), the grid, server down.
# Usage: bash scripts/run_benchmark.sh <model key> [--pilot]
# Writes results/raw/<key>.jsonl (results/pilot/<key>.jsonl with --pilot); traces go to ~/aib-runs/traces/.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
key="$1"; shift
uv run --frozen python -c "from aib.repro import require_clean; require_clean()"
provider="$(uv run --frozen python -m aib.config provider "$key")"
if [ "$provider" = "llamacpp" ]; then
  trap 'bash scripts/serve.sh down' EXIT
  bash scripts/serve.sh up "$key"
fi
uv run --frozen python -m aib.experiments run --model "$key" "$@"
