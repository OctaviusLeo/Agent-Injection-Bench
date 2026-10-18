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

## Filter classifier

## CI
