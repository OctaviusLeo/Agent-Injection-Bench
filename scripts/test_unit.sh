#!/usr/bin/env bash
# The CI test command: every CPU-only unit and property test with coverage, then the per-module coverage check.
# Usage: bash scripts/test_unit.sh [extra pytest args]
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
mkdir -p runs
uv run --frozen pytest -m "not gpu and not docker and not network and not classifier" \
  --cov=aib --cov=scripts --cov-report=json:runs/coverage.json --cov-report=term "$@"
uv run --frozen python scripts/check_coverage.py runs/coverage.json
