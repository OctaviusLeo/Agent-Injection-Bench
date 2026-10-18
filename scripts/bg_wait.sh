#!/usr/bin/env bash
# Wait until a detached job ends or [seconds] pass (default 1200), then report it like bg_status.sh.
# Usage: bash scripts/bg_wait.sh <name> [seconds] [lines]
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
name="$1"; limit="${2:-1200}"; lines="${3:-20}"; d=runs/logs
alive() { local s; s="$(ps -o stat= -p "$(cat "$d/$name.pid" 2>/dev/null)" 2>/dev/null | tr -d ' ')"; [ -n "$s" ] && [ "${s#Z}" = "$s" ]; }
end=$(( $(date +%s) + limit ))
while [ ! -f "$d/$name.exit" ] && alive && [ "$(date +%s)" -lt "$end" ]; do sleep 5; done
exec bash scripts/bg_status.sh "$name" "$lines"
