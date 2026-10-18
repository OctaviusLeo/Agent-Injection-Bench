#!/usr/bin/env bash
# M3 gate: unit tests, then each local model's complete store, errored share, filter decisions, calibration and pilot.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
bash scripts/test_unit.sh
uv run --frozen python -m aib.gates m3
