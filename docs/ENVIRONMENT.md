# Environment

Exact versions of every tool, model, API, and dependency this project relies on, each with a verifiable artifact obtained on the target machine (Req 0.5, 0.6). Command output backing these entries is saved under `docs/evidence/`.

## Machine

| Item | Value | Artifact |
| --- | --- | --- |
| GPU | NVIDIA GeForce RTX 3080, total memory 10240 MiB (`nvidia-smi`) | `docs/evidence/nvidia_smi_qwen3-4b.txt`, `docs/evidence/nvidia_smi_llama3.1-8b.txt` (full `nvidia-smi` output captured by the smoke test) |
| NVIDIA driver | Windows kernel-mode driver 617.42, seen in WSL through `nvidia-smi` 615.78.05; CUDA user-mode driver 13.4 | same files (header line of the `nvidia-smi` output) |
| OS | Windows 11 Pro 10.0.26200 with WSL2 `Ubuntu-24.04` | `docs/evidence/python_version.txt` |
| Docker | Docker Engine 29.9.1 inside WSL with the NVIDIA Container Toolkit; `docker run --rm --gpus all ubuntu nvidia-smi` passed | `results/m0_smoke_qwen3-4b.json` (`docker_ok`, `docker_gpu_ok`) |
| GPU idle reading | 1422 MiB used with nothing of this project running (the Windows desktop); the idle rule allows up to 1024 MiB above it before a server start | `scripts/record_gpu_idle.sh` |

## Python and tooling

| Item | Version | Artifact |
| --- | --- | --- |
| WSL distro | Ubuntu 24.04 (`Ubuntu-24.04`) | `docs/evidence/python_version.txt` (Ubuntu's `/usr/bin/python3.12`) |
| Python | 3.12.3 | `docs/evidence/python_version.txt` (`python --version` in `~/.venvs/aib`) |
| uv | 0.13.1 | `docs/evidence/uv_version.txt` (`uv --version`) |
| Python dependencies | pinned in `pyproject.toml`, full closure in `uv.lock` | `docs/evidence/uv_tree.txt` (`uv tree --frozen`); licenses in `docs/evidence/python_licenses.json` |
| agentdojo | 0.1.35 | `docs/evidence/agentdojo_license.json` (installed distribution metadata); the unit tests in `tests/` exercise it |

Direct dependencies (exact pins in `pyproject.toml`): agentdojo 0.1.35, openai 3.35.0, transformers 5.20.1, torch 2.14.1 (CPU wheel `2.14.1+cpu` from the PyTorch CPU index), huggingface_hub 2.2.0, numpy 2.5.4, matplotlib 3.11.2, pyyaml 6.0.3, pydantic 2.14.1, pytest 9.1.1, pytest-cov 7.1.0, hypothesis 6.170.1. Each appears with that version in `docs/evidence/uv_tree.txt`.

## Model server

| Item | Value | Artifact |
| --- | --- | --- |
| Model_Server | llama.cpp `llama-server`, build `b11580-4413a1924` (from `/props`; `--version` reports build 11580, commit `4413a1924`) | `results/m0_smoke_qwen3-4b.json` (`server.build_info`), `docs/evidence/llama_image.txt` |
| Image | `ghcr.io/ggml-org/llama.cpp:server-cuda12-b11580`, pinned as `ghcr.io/ggml-org/llama.cpp@sha256:c4348afb91499b10fad12f92626c7a8ca8b69d6475d10c793b810cb61237ce00` in `docker/compose.yaml` | `docs/evidence/llama_image.txt` (`docker image inspect`) |
| Flags | `--jinja --n-gpu-layers 99 --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0`, plus `--ctx-size` and `--parallel` per model; spellings checked against the image's `--help` | `docs/evidence/llama_server_help.txt`, `config/models.yaml` |
| Endpoint | Compose project `aib`, service `llm`, `127.0.0.1:8097` (port 8091 is published by another project's container on this machine) | `docker/compose.yaml` |

## Local models

Downloaded by `scripts/fetch_models.py` at the pinned revisions; sizes match `config/models.yaml` and each SHA-256 matches the Hugging Face LFS object id (`results/m0_models.json`). Licenses from the Hugging Face API at those revisions (`docs/evidence/model_licenses.txt`).

| Key | Model | Repository and revision | Files (size, SHA-256) | Quantization | License | Slots x context per slot |
| --- | --- | --- | --- | --- | --- | --- |
| `qwen3-4b` | Qwen3-4B-Instruct-2507 | `unsloth/Qwen3-4B-Instruct-2507-GGUF` at revision `a06e946bb6b655725eafa393f4a9745d460374c9` | `Qwen3-4B-Instruct-2507-Q4_K_M.gguf` (2497281120 bytes, SHA-256 `3605803b982cb64aead44f6c1b2ae36e3acdb41d8e46c8a94c6533bc4c67e597`) | Q4_K_M | Apache-2.0 | 2 x 32768 tokens |
| `qwen2.5-7b` | Qwen2.5-7B-Instruct | `Qwen/Qwen2.5-7B-Instruct-GGUF` at revision `bb5d59e06d9551d752d08b292a50eb208b07ab1f` | `qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf` (3993201344 bytes, SHA-256 `dfce12e3862a5283ccfb88221b48480e58745165de856439950d0f22590580db`), `qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf` (689872288 bytes, SHA-256 `539cf93f78e887edea1c04e2d7d8cdaca9d01dae9c9025bcb8accbe29df3d72a`) | Q4_K_M | Apache-2.0 | 1 x 32768 tokens |

**8B slot: `qwen2.5-7b` replaced `llama3.1-8b`.** Llama-3.1-8B-Instruct (`bartowski/Meta-Llama-3.1-8B-Instruct-GGUF` at revision `bf5b95e96dac0462e2a09145ec66cae9a3f12067`, downloaded and verified in `results/m0_models.json`) passed the single-call smoke test and tool-call probe (`results/m0_smoke_llama3.1-8b.json`, `results/m0_tool_probe_llama3.1-8b.json`), but when a turn needs two tools it joins two JSON calls with `; `, which llama.cpp build b11580 cannot parse: the server answers HTTP 500 and the first benchmark attempt stopped after 46 cases. The multi-tool probe (`scripts/multi_tool_probes.sh`; three two-tool prompts x 8 seeds x 3 `parallel_tool_calls` settings) parsed 0 of 72 Llama turns with the GGUF's template and 0 of 72 with llama.cpp's bundled template from the same build tag (`results/m0_multi_tool_probe_llama3.1-8b_gguf-template.json`, `results/m0_multi_tool_probe_llama3.1-8b_llamacpp-bundled-template.json`), against 72 of 72 for `qwen2.5-7b` and 72 of 72 for `qwen3-4b`. Following design.md's fallback order, `qwen2.5-7b` is the 8B-slot model; its partial store was discarded to `~/aib-runs/discarded/`.

## M0 smoke tests

One tool-free prompt per model with a 120-second limit; GPU memory sampled about every 100 ms during generation (`scripts/smoke_test.py`, results in `results/m0_smoke_<model>.json`), then one request with AgentDojo's `get_current_day` tool (`scripts/tool_probe.py`, results in `results/m0_tool_probe_<model>.json`). Memory figures are total GPU memory in use, including the Windows desktop.

| Model | Python | Docker | Total MiB | Baseline MiB | Peak MiB | Context per slot | Decode tokens/s | Response | Tool-call probe |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `qwen3-4b` (in use) | 3.12.3 | yes | 10240 | 9016 | 9018 | 32768 | 113.6 | A unit test checks whether a single unit of code (like a function or method) produces the expected output when given specific inputs. | passed (`get_current_day`) |
| `qwen2.5-7b` (in use) | 3.12.3 | yes | 10240 | 6445 | 6445 | 32768 | 102.5 | A unit test checks whether individual units of code, such as functions or methods, behave as expected under specific conditions. | passed (`get_current_day`) |
| `llama3.1-8b` (replaced (see above)) | 3.12.3 | yes | 10240 | 8159 | 8161 | 32768 | 94.1 | A unit test checks whether a specific piece of code, or unit, performs its intended function correctly under given conditions. | passed (`get_current_day`) |

Stamped commits of the models in use: `8ef04ba7dbafdf3a270008830ebd3413b72b59db`, `31a0db3372b0d73741ce10b7440c2a7b89829b3a` (seed 0).

## Hosted model and Spend_Cap

| Item | Value | Artifact |
| --- | --- | --- |
| Hosted model | `gpt-6-luna` (OpenAI Chat Completions); response `model` field: `gpt-6-luna` | `results/m0_hosted_check.json` (stamp: `python scripts/hosted_check.py`, seed 0, commit `1f9719b39f53e85b12e52279ba46fd78b5506ceb`) |
| Spend_Cap | $3.00 per month (`spend_cap_usd: 3.00` in `config/hosted.yaml`), enforced by `aib/ledger.py` as a total over every hosted request | `results/hosted_ledger.jsonl` |
| Request settings | `reasoning_effort: none`, `max_completion_tokens: 1024` | `config/hosted.yaml` |
| Accepted sampling parameters | `temperature`, `top_p`, `seed` (none rejected) | `config/hosted.yaml` `accepted_params`, `results/m0_hosted_check.json` |
| API key | read only from `OPENAI_API_KEY`, exported by `scripts/wsl_run.sh` from `~/.config/aib/secrets.env` (outside the repository) | `tests/test_llm.py::test_missing_secret_raises_before_any_request` |
| openai (Python client) | 3.35.0 | `docs/evidence/uv_tree.txt`; `tests/test_llm.py` exercises it against a fake server |

## Filter classifier

| Item | Value | Artifact |
| --- | --- | --- |
| Classifier | `protectai/deberta-v3-base-prompt-injection-v2` at revision `90c9989b1a342275dd0d1a95aad283c04e075671` (base `microsoft/deberta-v3-base`), files in `~/aib-models/classifier` | `results/m0_models.json` (sizes and SHA-256 of every file) |
| transformers | 5.20.1 | `docs/evidence/uv_tree.txt`; `tests/test_calibration.py::test_real_classifier_scores_planted_and_clean_text` passes against it |
| torch | 2.14.1+cpu (PyTorch CPU wheel index) | `docs/evidence/uv_tree.txt`; the same test |
| Settings | threshold 0.5, windows of 512 tokens with 64-token overlap, label `INJECTION` | `config/defenses.yaml` |
| Calibration check | 324 distinct ground-truth outputs scored in 4761.44 CPU seconds (524.57 s wall) | `results/m3_filter_calibration.json` (stamp: `python scripts/filter_calibration.py`, seed 0, commit `081cf580edca759c814006a2dfa59634e1350196`) |

## CI

| Item | Version | Artifact |
| --- | --- | --- |
| GitHub Actions runner | `ubuntu-latest` (GitHub-hosted) | `.github/workflows/ci.yml` |
| actions/checkout | v7.0.1 (latest release, published 2026-07-20) | `gh api repos/actions/checkout/releases/latest`, read 2026-10-18; pinned in `.github/workflows/ci.yml` |
| astral-sh/setup-uv | v10.3.2 (published 2026-10-22) | `gh api repos/astral-sh/setup-uv/releases/tags/v10.3.2`, read 2026-10-28; pinned in `.github/workflows/ci.yml` (bumped from v10.3.0 by Dependabot, merged 2026-10-27) |
