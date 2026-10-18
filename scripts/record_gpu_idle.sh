#!/usr/bin/env bash
# Record the GPU's idle memory reading (MiB) with nothing of this project running: runs/gpu_idle_mib.txt.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
mkdir -p runs
if [ -n "$(docker ps -q --filter label=com.docker.compose.project=aib)" ]; then
  echo "an aib container is running; stop it first (bash scripts/serve.sh down)" >&2; exit 1
fi
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n1 | tr -d ' ' > runs/gpu_idle_mib.txt
echo "idle reading: $(cat runs/gpu_idle_mib.txt) MiB"
