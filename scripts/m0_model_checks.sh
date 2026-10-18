#!/usr/bin/env bash
# Run the M0 smoke test and the tool-call probe for one Local_Model, then stop the server.
# Usage: bash scripts/m0_model_checks.sh <model key> [probe attempt]
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
key="$1"; attempt="${2:-1}"
trap 'bash scripts/serve.sh down' EXIT
uv run --frozen python scripts/smoke_test.py --model "$key"; smoke=$?
probe=1
if [ "$smoke" -eq 0 ]; then
  uv run --frozen python scripts/tool_probe.py --model "$key" --attempt "$attempt"; probe=$?
fi
echo "smoke exit $smoke, probe exit $probe"
[ "$smoke" -eq 0 ] && [ "$probe" -eq 0 ]
