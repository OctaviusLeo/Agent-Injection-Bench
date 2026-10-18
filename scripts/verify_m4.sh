#!/usr/bin/env bash
# M4 gate: unit tests, then the hosted store and ledger, metrics coverage and intervals, regeneration, and plots.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
bash scripts/test_unit.sh
uv run --frozen python -m aib.gates m4
