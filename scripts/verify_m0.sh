#!/usr/bin/env bash
# M0 gate: unit tests, then license evidence, model files, smoke tests and probes, hosted check, docs, and pins.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
bash scripts/test_unit.sh
uv run --frozen python -m aib.gates m0
