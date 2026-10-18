#!/usr/bin/env bash
# Regenerate every derived result from the Results_Store, in order: metrics, Results_Table, plots, Report, README,
# then the README/Report number-traceability check. Usage: bash scripts/build_results.sh
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
uv run --frozen python -c "from aib.repro import require_clean; require_clean()"
uv run --frozen python -m aib.metrics --store results/raw --seed 20261017 --resamples 2000 --out results/metrics.json
uv run --frozen python -m aib.results_table
uv run --frozen python -m aib.plots
uv run --frozen python -m aib.report
uv run --frozen python -m aib.docs_build
uv run --frozen python scripts/check_traceability.py
