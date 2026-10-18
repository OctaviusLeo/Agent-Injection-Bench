#!/usr/bin/env bash
# Run one command in WSL from the repository root with the project's environment.
# From PowerShell: wsl -d Ubuntu-24.04 -- bash /mnt/c/dev/Agent-Injection-Bench/scripts/wsl_run.sh <command> [args...]
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
export UV_PROJECT_ENVIRONMENT="$HOME/.venvs/aib"
export AIB_MODELS="$HOME/aib-models"
export AIB_RUNS="$HOME/aib-runs"
export PATH="$HOME/.local/bin:$UV_PROJECT_ENVIRONMENT/bin:$PATH"
mkdir -p "$AIB_MODELS" "$AIB_RUNS"
secrets="$HOME/.config/aib/secrets.env"
if [ -f "$secrets" ]; then set -a; . "$secrets"; set +a; fi
exec "$@"
