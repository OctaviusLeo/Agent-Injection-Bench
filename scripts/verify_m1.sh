#!/usr/bin/env bash
# M1 gate: unit tests, then the catalog checks (40 tasks, ground truth passes, placeholders, regeneration).
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
bash scripts/test_unit.sh
uv run --frozen python -m aib.gates m1
