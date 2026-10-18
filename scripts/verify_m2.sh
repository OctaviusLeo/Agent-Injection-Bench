#!/usr/bin/env bash
# M2 gate: unit tests, then the stub-validation checks (400 cases, policies' expected outcomes, styles, goals).
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
bash scripts/test_unit.sh
uv run --frozen python -m aib.gates m2
