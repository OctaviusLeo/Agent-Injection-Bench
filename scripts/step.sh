#!/usr/bin/env bash
# Run one Python build step with a time limit. If it fails or hangs, retry it once through a different launch
# path (the project venv's interpreter directly instead of `uv run`), then fail naming the step and the cause.
# Usage: bash scripts/step.sh <seconds> <python arguments...>   e.g. bash scripts/step.sh 1800 -m aib.metrics ...
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
limit="$1"; shift
venv_python="${UV_PROJECT_ENVIRONMENT:-$HOME/.venvs/aib}/bin/python"
for attempt in 1 2; do
  if [ "$attempt" -eq 1 ]; then runner=(uv run --frozen python); how="uv run"; else runner=("$venv_python"); how="venv python"; fi
  timeout --kill-after=30 "$limit" "${runner[@]}" "$@"
  code=$?
  [ "$code" -eq 0 ] && exit 0
  if [ "$code" -eq 124 ] || [ "$code" -eq 137 ]; then cause="timed out after ${limit}s"; else cause="exit status $code"; fi
  echo "step 'python $*' attempt $attempt ($how) failed: $cause" >&2
done
echo "step 'python $*' failed with both launch paths" >&2
exit 1
