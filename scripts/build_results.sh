#!/usr/bin/env bash
# Regenerate every derived result from the Results_Store, in order: metrics, Results_Table, plots, Report, README,
# then the README/Report number-traceability check. Each step has a time limit and one retry through a different
# launch path (scripts/step.sh). Usage: bash scripts/build_results.sh
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
bash scripts/step.sh 120 -c "from aib.repro import require_clean; require_clean()"
bash scripts/step.sh 1800 -m aib.metrics --store results/raw --seed 20261017 --resamples 2000 --out results/metrics.json
bash scripts/step.sh 300 -m aib.results_table
bash scripts/step.sh 600 -m aib.plots
bash scripts/step.sh 300 -m aib.report
bash scripts/step.sh 300 -m aib.docs_build
bash scripts/step.sh 300 scripts/check_traceability.py
