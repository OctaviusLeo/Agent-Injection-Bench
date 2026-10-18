#!/usr/bin/env bash
# Run the full local-model benchmark: every enabled local model in config/models.yaml, one after the other.
# Usage: bash scripts/run_all_local.sh
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
for key in $(uv run --frozen python -m aib.config local-models); do
  bash scripts/run_benchmark.sh "$key"
done
