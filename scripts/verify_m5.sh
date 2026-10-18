#!/usr/bin/env bash
# M5 gate (end to end): unit tests, traceability, a second build_results.sh run that must change nothing but stamps,
# then the Report, README, CI, THIRD_PARTY, and design checks. Run scripts/record_ci_status.ps1 from PowerShell first.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
bash scripts/test_unit.sh
uv run --frozen python scripts/check_traceability.py
rm -rf runs/m5_before && mkdir -p runs/m5_before/results runs/m5_before/docs
cp results/summary.md runs/m5_before/results/summary.md
cp docs/REPORT.md runs/m5_before/docs/REPORT.md
cp README.md runs/m5_before/README.md
bash scripts/build_results.sh
uv run --frozen python -m aib.gates m5
