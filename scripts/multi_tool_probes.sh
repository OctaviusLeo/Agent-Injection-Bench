#!/usr/bin/env bash
# Evidence for the 8B-slot fallback (design.md "M0 outcome"): serve each candidate in turn and run the multi-tool probe.
# Usage: bash scripts/multi_tool_probes.sh   (writes results/m0_multi_tool_probe_<model>_<variant>.json)
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
trap 'bash scripts/serve.sh down >/dev/null 2>&1 || true' EXIT
probe() {  # <model key> <variant label> [chat template under ~/aib-models]
  uv run --frozen python -m aib.server_variant "$1" "${3:-}"
  uv run --frozen python -m aib.multi_tool_probe --model "$1" --variant "$2"
  bash scripts/serve.sh down
}
probe llama3.1-8b gguf-template
probe llama3.1-8b llamacpp-bundled-template templates/meta-llama-Llama-3.1-8B-Instruct.jinja
probe qwen2.5-7b gguf-template
probe qwen3-4b gguf-template
