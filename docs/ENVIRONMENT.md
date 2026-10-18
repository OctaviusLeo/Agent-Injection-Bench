# Environment

Exact versions of every tool, model, API, and dependency this project relies on, each with a verifiable artifact obtained on the target machine (Req 0.5, 0.6). Command output backing these entries is saved under `docs/evidence/`.

## Machine

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

## Local models

## M0 smoke tests

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

## CI
