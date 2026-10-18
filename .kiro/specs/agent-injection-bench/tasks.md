# Implementation Plan: Agent Injection Bench

## Overview

This plan converts the design into incremental, code-focused tasks organized under milestones M0–M5. It is written to run **autonomously** with Kiro's "Run all tasks": every task is agent-implementable, and every milestone ends with an **automated verification gate** (a single repository command whose exit status decides pass or fail) instead of an owner review stop. The agent moves on when the gate exits 0. If a gate exits non-zero, the agent fixes the cause and re-runs it, up to 3 fix attempts for the same failure; after that it records the failure in `docs/BLOCKERS.md` (task, command, exit code, log path, suspected cause) and stops that task. Targets, thresholds, the Spend_Cap, and tests are never loosened to make a gate pass.

**Earlier work:** an earlier run wrote a custom sandbox, tools, tasks, and statistics code (`aib/`, `tests/`, `scripts/`) before the decisions in design.md existed. Design.md replaces its sandbox, tools, tasks, injections, goals, runner, and defenses with AgentDojo's workspace suite, because those modules had validity defects (design.md, "Prior work and the build-or-reuse decision"). Task 0.1 removes them; the modules design.md still uses are rewritten by the tasks named below rather than started over.

Implementation language: **Python 3.12** (Ubuntu 24.04's Python, with `uv`). Package root: `aib/`. The benchmark substrate is `agentdojo==0.1.35`'s workspace suite at benchmark version v1.2.2 with the v1.1.2 task fixes, exactly as design.md's "Benchmark substrate" builds it. Every reported number comes from a repository script, recorded with its command, seed, and full 40-character commit hash; `results/summary.md`, `docs/REPORT.md`, `README.md`, and the plots are generated files, never hand-edited. By every milestone gate, each non-trivial Python module (more than one public function or more than 50 source lines) has unit tests that keep it at 80% statement coverage or more under `bash scripts/test_unit.sh`; a task that adds such a module and has no separate test task adds those tests itself. Every tool is simulated inside the Sandbox; no task reaches a real account, website, or system (Req 0.14). Commit after each task (`task <id>: <summary>`).

**Where this runs:** on the owner's Windows 11 desktop with an RTX 3080 (design.md, "Execution environment"; `.kiro/steering/environment.md`). Python and shell scripts run in WSL `Ubuntu-24.04` through `scripts/wsl_run.sh`; the llama.cpp server runs in Docker on the GPU, one model at a time; Git runs on Windows; CI runs on GitHub-hosted runners. Anything that may take longer than 10 minutes runs detached through `scripts/bg.sh` and is waited on with `scripts/bg_wait.sh`. The owner's only inputs (the empty private GitHub repository, the earlier files in place, the OpenAI key in `~/.config/aib/secrets.env`, and OpenAI credit) are ready before the run starts.

## Milestone Dependency Graph

```mermaid
flowchart TB
    T0[Setup: scaffold + uv project + helpers + repro + config]
    M0[M0: license evidence + model client + ledger + model server + smoke tests + hosted check]
    M1[M1: AgentDojo workspace Sandbox + 40 User_Tasks + checks + catalog]
    M2[M2: 5 Injection_Styles + 6 Injected_Goals + classification + stub + case runner]
    M3[M3: 4 defenses + classifier check + grid runner + pilot + 2 local models]
    M4[M4: metrics + statistics + table + plots + hosted run]
    M5[M5: Report + traceability + CI + README]

    T0 --> M0 --> M1 --> M2 --> M3 --> M4 --> M5
```

Every wave holds one task, so tasks run strictly in this order: no two tasks edit files, use Git, or use the GPU at the same time.

```json
{
  "waves": [
    { "id": 0, "tasks": ["0.1"] },
    { "id": 1, "tasks": ["0.2"] },
    { "id": 2, "tasks": ["0.3"] },
    { "id": 3, "tasks": ["1.1"] },
    { "id": 4, "tasks": ["1.2"] },
    { "id": 5, "tasks": ["1.3"] },
    { "id": 6, "tasks": ["1.4"] },
    { "id": 7, "tasks": ["1.5"] },
    { "id": 8, "tasks": ["1.6"] },
    { "id": 9, "tasks": ["2.1"] },
    { "id": 10, "tasks": ["2.2"] },
    { "id": 11, "tasks": ["2.3"] },
    { "id": 12, "tasks": ["2.4"] },
    { "id": 13, "tasks": ["3.1"] },
    { "id": 14, "tasks": ["3.2"] },
    { "id": 15, "tasks": ["3.3"] },
    { "id": 16, "tasks": ["3.4"] },
    { "id": 17, "tasks": ["3.5"] },
    { "id": 18, "tasks": ["3.6"] },
    { "id": 19, "tasks": ["4.1"] },
    { "id": 20, "tasks": ["4.2"] },
    { "id": 21, "tasks": ["4.3"] },
    { "id": 22, "tasks": ["4.4"] },
    { "id": 23, "tasks": ["4.5"] },
    { "id": 24, "tasks": ["4.6"] },
    { "id": 25, "tasks": ["4.7"] },
    { "id": 26, "tasks": ["4.8"] },
    { "id": 27, "tasks": ["5.1"] },
    { "id": 28, "tasks": ["5.2"] },
    { "id": 29, "tasks": ["5.3"] },
    { "id": 30, "tasks": ["5.4"] },
    { "id": 31, "tasks": ["5.5"] },
    { "id": 32, "tasks": ["5.6"] },
    { "id": 33, "tasks": ["6.1"] },
    { "id": 34, "tasks": ["6.2"] },
    { "id": 35, "tasks": ["6.3"] },
    { "id": 36, "tasks": ["6.4"] },
    { "id": 37, "tasks": ["6.5"] },
    { "id": 38, "tasks": ["6.6"] },
    { "id": 39, "tasks": ["6.7"] }
  ]
}
```

## Tasks

## Setup (prerequisite scaffolding)

- [x] 0.1 Bring the repository scaffold, Python project, helper scripts, and GitHub remote in line with the design
  - Delete what the earlier run left that design.md replaces: `aib/sandbox.py`, `aib/tools.py`, `aib/tasks.py`, `aib/injections.py`, `aib/goals.py`, `aib/stub_model.py`, `aib/runner.py`, `aib/experiments.py`, `aib/defenses.py`, `aib/model_client.py`, `scripts/run_local.py`, `scripts/run_hosted.py`, `scripts/smoke_test.py`, `scripts/ci_smoke.py`, every test that imports one of them, `requirements.txt` (`uv.lock` replaces it), and an `.agents/` folder if one exists (another tool's notes from the earlier run; they do not apply to this plan).
  - Keep `aib/repro.py`, `aib/metrics.py`, `aib/results_table.py`, `aib/plots.py`, `aib/report.py`, `aib/traceability.py`, `scripts/check_traceability.py`, and their tests; tasks 0.2, 5.1, 5.3, 6.1, and 6.2 rewrite them to design.md's interfaces. Move any kept file whose tests fail after the deletions, with its tests, to `runs/earlier/` (gitignored) as reference for the task that rewrites it.
  - Create the remaining layout from design.md: `config/`, `docker/`, `docs/evidence/`, `results/raw/`, `results/plots/`, `.github/workflows/`, and `runs/` (gitignored).
  - Make `pyproject.toml` a `uv` project with `requires-python = ">=3.12,<3.13"`, `.python-version` holding `3.12`, and one dependency set, every dependency pinned to an exact version: `agentdojo==0.1.35`, openai, transformers, torch (from the PyTorch CPU index exactly as design.md's "Platform, language, tooling" shows), huggingface_hub, numpy, matplotlib, pyyaml, pydantic, pytest, pytest-cov, hypothesis. Register the pytest markers `gpu`, `docker`, `network`, and `classifier`. Generate `uv.lock` with `wsl -d Ubuntu-24.04 -- bash /mnt/c/dev/Agent-Injection-Bench/scripts/wsl_run.sh uv lock`, then create the environment with `... wsl_run.sh uv sync --frozen`.
  - Create `scripts/wsl_run.sh`, `scripts/bg.sh`, `scripts/bg_status.sh`, and `scripts/bg_wait.sh` exactly as written in `.kiro/steering/environment.md`.
  - Write `scripts/test_unit.sh`, the CI test command: `uv run --frozen pytest -m "not gpu and not docker and not network and not classifier"` with coverage over `aib/` and `scripts/` written to `runs/coverage.json`, then `uv run --frozen python scripts/check_coverage.py runs/coverage.json`. Write `scripts/check_coverage.py`, with its own unit tests, which fails and lists every module in `aib/` or `scripts/` that has more than one public function or more than 50 source lines and is under 80% statement coverage.
  - `.gitignore`: every secret-bearing pattern (`.env`, `*.env`, `secrets.*`, `*.key`, `*.pem`, `*.token`), `runs/`, `.venv/`, `__pycache__/`, `.pytest_cache/`, `.hypothesis/`, `.coverage`, `htmlcov/`, and model files (`*.gguf`, `*.safetensors`, `*.onnx`, `*.bin`). `.gitattributes` exactly as design.md specifies.
  - Make sure `docs/ENVIRONMENT.md` and `THIRD_PARTY.md` (at the repository root) exist with section headers, and that `README.md` states the repository is private. Confirm `bash scripts/test_unit.sh` exits 0.
  - From PowerShell: if the folder is not a Git repository, run `git init -b main`; otherwise keep its current branch. Set `origin` to `https://github.com/OctaviusLeo/Agent-Injection-Bench.git` (the owner made it private and it is empty), commit, and `git push -u origin HEAD`.
  - _Requirements: 0.5, 0.7, 0.9, 0.10, 0.11, 0.12_

- [x] 0.2 Implement the shared reproducibility helper
  - Rewrite `aib/repro.py` to design.md's interface: `commit_hash()` raises unless `git rev-parse HEAD` gives 40 hex characters (the earlier version returned 40 zeros when there was no commit, which would stamp a fake hash), `require_clean()` with design.md's allowed paths, the `Stamp` dataclass `{command, seed, commit}`, and `make_stamp()`. Every result-writing script uses them.
  - Unit tests: the hash is 40 hex characters; `require_clean()` in a throwaway repository in a temporary directory accepts a change under `results/` and refuses one under `aib/`, listing the path; a stamp round-trips through JSON.
  - _Requirements: 0.1, 0.2_

- [x] 0.3 Write the configuration files and their loader
  - Write `config/experiment.yaml`, `config/models.yaml`, `config/hosted.yaml`, and `config/defenses.yaml` with exactly the values in design.md ("Data Models", "Models", "Defense_Conditions", "Injection_Styles", and "Injected_Goals"), including `spend_cap_usd: 3.00` copied from design.md's Spend_Cap, the fallback model `qwen2.5-7b` (disabled until task 1.4 enables it), and the hosted conditions.
  - Write `aib/config.py` with pydantic models that validate every value on load.
  - Unit tests: each file loads; an unknown defense, style, or model key, fewer than 3 Repetitions, a negative Spend_Cap, or a threshold outside [0, 1] is rejected with an error naming the field.
  - _Requirements: 0.2, 1.11, 4.5, 5.3, 5.4_

## M0 — Prior-work evidence, model server, and smoke tests

- [x] 1.1 Record the prior-work license evidence
  - Write `scripts/license_evidence.py`: it reads the installed `agentdojo` distribution's metadata (`importlib.metadata`: version, classifiers, and the bundled `LICENSE` file) and fetches InjecAgent's license file from `https://raw.githubusercontent.com/uiuc-kang-lab/InjecAgent/main/LICENCE`, and saves both, with the access date, under `docs/evidence/`.
  - Confirm both are MIT. If either is not, record a blocker: the PriorWork_Decision in design.md depends on it.
  - Record in `THIRD_PARTY.md`: AgentDojo (`agentdojo` 0.1.35, MIT, its copyright line, and the reuse mode design.md describes: pinned package, workspace suite, injection tasks 0–5, pipeline elements, the spotlighting-with-delimiting defense, and the attack templates), InjecAgent (MIT, the "enhanced" attack sentence used through AgentDojo's `injecagent` attack), and every Python package in `uv.lock` with its version and license identifier taken from its installed metadata.
  - Record in `docs/ENVIRONMENT.md`, with artifacts: Python's version, uv's version, and the `uv tree` output saved under `docs/evidence/`. Fill the AgentDojo and InjecAgent lines of design.md's "M0 outcome" block.
  - _Requirements: 1.1, 1.2, 1.3, 0.5, 0.6, 0.7, 0.8_

- [x] 1.2 Implement the model element and the spend ledger
  - Write `aib/llm.py` (`ChatLLM`, an AgentDojo `BasePipelineElement`) and `aib/ledger.py` (`SpendLedger`) exactly as design.md's "Model element, spend ledger, and Stub_Model" describes: message conversion with the `system` role, tool schemas, sampling settings and per-request seed, local and hosted parameter sets, malformed-argument handling, retries, `CallRecord`s in `extra_args`, the missing-secret check, the conservative cost formula, and the precheck that refuses a request that could push spend past `spend_cap_usd`.
  - Unit tests against a fake OpenAI-compatible server started in the test on `127.0.0.1`:
    - a two-turn tool conversation converts both ways, including tool-call ids, tool errors, and multiple tool calls in one message;
    - invalid JSON arguments become `{}` with the flag set;
    - 429 and 500 responses are retried, and a timeout surfaces as a typed error;
    - an unset or empty `OPENAI_API_KEY` raises `MissingSecretError` naming it, and the fake server receives no request;
    - the ledger refuses a request whose worst case would exceed the cap, before sending, and records cost from `usage` with cached tokens.
  - _Requirements: 0.9, 0.13, 1.5, 5.2, 5.3_

- [x] 1.3 Fetch the model files and set up the model server
  - Write `scripts/fetch_models.py`: it downloads with `huggingface_hub` at the revisions pinned in `config/models.yaml` into `~/aib-models`: the `qwen3-4b` and `llama3.1-8b` GGUF files, and the classifier snapshot (only `config.json`, `model.safetensors`, `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`, `added_tokens.json`, and `spm.model`; never `onnx/` or `training_args.bin`). It accepts `--model <key>` to fetch one entry (task 1.4 uses this for the fallback), checks each file's size against the config, computes its SHA-256, and writes a stamped `results/m0_models.json`. Run it detached.
  - Write `docker/compose.yaml` (project `aib`, service `llm`, image `ghcr.io/ggml-org/llama.cpp:server-cuda12-b11580`, port `127.0.0.1:8091:8080`, `${AIB_MODELS}` mounted read-only, a GPU reservation, a `/health` healthcheck, and the per-model model path, context, and slots taken from environment variables) and `scripts/serve.sh up <model key> | down`, with its logic in `aib/server.py` as design.md describes (idle-GPU rule, `docker compose -p aib`, waiting for `/health` to return 200 for up to 900 s, printing `/props`).
  - Pull the image detached, record its digest, and pin it in `docker/compose.yaml` as `image: ghcr.io/ggml-org/llama.cpp:server-cuda12-b11580@sha256:<digest>`. Save `llama-server --help` from the image to `docs/evidence/llama_server_help.txt` and use the flag spellings it lists for design.md's server flags.
  - Unit tests for `aib/server.py` with fakes for `docker`, `nvidia-smi`, and HTTP: the idle rule waits and then raises; `up` passes the model's settings; `/health` polling times out cleanly.
  - _Requirements: 0.5, 0.6, 0.10, 1.8_

- [ ] 1.4 Run the smoke test and the tool-call probe for each Local_Model
  - With nothing of this project running, record the idle reading of `nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits` in `runs/gpu_idle_mib.txt`.
  - Write `scripts/smoke_test.py --model <key>`, with its logic in `aib/smoke.py` and unit tests that fake `nvidia-smi`, Docker, and HTTP. It does the following and writes a stamped `results/m0_smoke_<key>.json`:
    - verifies Python ≥ 3.11 and that Docker works (`docker version`, then `docker run --rm --gpus all ubuntu nvidia-smi`);
    - captures `nvidia-smi` output, including total GPU memory in MiB;
    - starts the server with `scripts/serve.sh up <key>` and records the per-slot context from `/props`;
    - sends exactly one tool-free prompt with a 120-second limit, and records the generation speed the server reports (`timings.predicted_per_second`);
    - samples `nvidia-smi --query-gpu=memory.used` about every 100 ms from just before the request until it completes, and records the baseline and the peak in MiB;
    - records the response;
    - on any failure or an unmet prerequisite, records the failed condition and the observed cause and exits non-zero.
  - Write `scripts/tool_probe.py --model <key>`: one request with AgentDojo's `get_current_day` tool schema and the user message "What day is it today? Use the tool." It passes when the response holds a structured tool call to `get_current_day` with arguments that parse as JSON, and it writes `results/m0_tool_probe_<key>.json`.
  - Run both scripts for `qwen3-4b`, then for `llama3.1-8b`, detached, with `scripts/serve.sh down` in between.
    - If a model's 2-slot configuration runs out of memory or its peak exceeds the total, apply design.md's slot fallback.
    - If `llama3.1-8b` fails the probe after the retry cap, apply design.md's fallback order: llama.cpp's bundled Llama 3.1 template from the same build tag, then `qwen2.5-7b`. Fetch `qwen2.5-7b`'s files with `fetch_models.py --model qwen2.5-7b`, enable it in `config/models.yaml` in place of `llama3.1-8b`, and rerun both scripts for it.
  - Record in `docs/ENVIRONMENT.md`, each beside its evidence file under `docs/evidence/`: the `nvidia-smi` output and total MiB, each model's peak MiB and response, per-slot context, the server build from `/props`, the image digest, and each model's repository, revision, file, SHA-256, quantization, and license. Record the models, llama.cpp (MIT), and the image in `THIRD_PARTY.md`, and fill the model lines of design.md's "M0 outcome" block.
  - _Requirements: 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 1.10, 0.4, 0.5, 0.6, 0.7_

- [x] 1.5 Run the hosted check
  - Write `scripts/hosted_check.py`. Through `ChatLLM` and the ledger, it sends one tool-free request and one request with the `get_current_day` tool to `gpt-6-luna`, with `reasoning_effort: "none"`, `max_completion_tokens: 1024`, and design.md's sampling settings.
    - If the API answers 400 naming an unsupported parameter, drop that parameter and retry. Write the parameters that worked to `config/hosted.yaml` as `accepted_params`.
    - Record the response `model` field, token usage, and cost in a stamped `results/m0_hosted_check.json`. Every request goes through the ledger.
    - If `OPENAI_API_KEY` is missing, the script exits non-zero naming it. That is a blocker for the owner, not something to work around.
  - Record in `docs/ENVIRONMENT.md`: the Spend_Cap ($3.00 per month), the hosted model, the accepted parameters, and the `openai` package version. Fill the hosted line of design.md's "M0 outcome" block.
  - _Requirements: 1.11, 5.2, 0.9, 0.13_

- [ ] 1.6 **Automated gate (M0):** run `bash scripts/verify_m0.sh`, which runs `bash scripts/test_unit.sh` and checks:
  - `results/m0_models.json` lists every model file with matching size and a SHA-256, and a stamp with a 40-character commit;
  - for each Local_Model in use: `results/m0_smoke_<key>.json` shows Python ≥ 3.11, Docker working, a non-empty response, and a peak MiB at or below the total MiB; `results/m0_tool_probe_<key>.json` shows a parsed tool call;
  - `results/m0_hosted_check.json` shows a non-empty response and a tool call, and the ledger total is at or below `spend_cap_usd`;
  - `docs/ENVIRONMENT.md` has no `TBD`, and every evidence path it cites exists; design.md's "M0 outcome" block has no `TBD`;
  - `THIRD_PARTY.md` lists AgentDojo, InjecAgent, each model in use, the classifier, and llama.cpp with license identifiers;
  - `docker/compose.yaml` pins the image by digest; `.gitignore` and `.gitattributes` hold the design's entries.
  - Proceed when it exits 0. No manual review stop.

## M1 — Sandbox, simulated tools, user tasks, success checks

- [x] 2.1 Build the workspace Sandbox and the User_Task catalog
  - Write `aib/suite.py` exactly as design.md's "Benchmark substrate" and "Sandbox, User_Tasks, and Task_Success_Check" describe: the suite construction order with the v1.1.2 registration, `load_user_tasks()` (40 tasks ordered by number), `injection_candidates()` (canary run on the ground truth), `init_case_environment()`, `CheckOutcome`, `task_success_check()` with its reason, and `validate_catalog()`.
  - Unit tests:
    - the version mapping in design.md (injection tasks 0, 4, and 5 and user tasks 13 and 33 from the `v1_1_2` modules, and so on);
    - 40 user tasks and 24 tools;
    - every user task's ground truth passes its check, and an empty run fails with a non-empty reason;
    - `validate_catalog` reports a task whose check is removed (monkeypatched) and the total count.
  - _Requirements: 2.1, 2.3, 2.4, 2.5, 2.6, 2.7, 0.14_

- [x] 2.2 Write property tests for the Sandbox and the checks
  - **Property 1: Sandbox never performs real network I/O**, **Property 2: Sandbox initialization is deterministic under a seed**, and **Property 3: Success checks are deterministic boolean predicates with a reason on failure** — one Hypothesis test each, as design.md specifies.
  - **Validates: Requirements 0.14, 2.1, 2.2, 2.4, 2.5, 2.6, 3.3, 3.4**

- [x] 2.3 Generate the catalog evidence
  - Write `scripts/catalog.py`, which writes a stamped `results/m1_catalog.json`:
    - per task: id, prompt, ground-truth tool calls, the injection placeholders it reads, and whether its ground truth passes its check;
    - the tool list and count;
    - the AgentDojo version and the version mapping;
    - the `validate_catalog` report.
  - Commit first, then run it.
  - _Requirements: 2.1, 2.3, 2.7, 0.1_

- [x] 2.4 **Automated gate (M1):** run `bash scripts/verify_m1.sh`, which runs `bash scripts/test_unit.sh` and checks:
  - `results/m1_catalog.json` has 40 user tasks, each passing its ground truth and reading at least one placeholder, at least 4 tools, an empty `validate_catalog` failure list, and a stamp;
  - regenerating the catalog into `runs/` gives the same content apart from the stamp.
  - Proceed when it exits 0. No manual review stop.

## M2 — Injection styles, injected goals, classification, case runner

- [x] 3.1 Implement the Injection_Styles
  - Write `aib/attacks.py` exactly as design.md's "Injection_Styles" table and interfaces describe: the five templates verbatim, `StyleAttack`, `plant()`, `normalize()`, and `contains_injection()`.
  - Unit tests: the four reused templates equal AgentDojo's (`DirectAttack`, `SystemMessageAttack`, `InjecAgentAttack`, and `ImportantInstructionsAttack._JB_STRING`); the five ids are distinct and include `direct`, `fake_system`, and `hidden_field`; `normalize` handles YAML's `\n` escapes and line wrapping (a planted text found in a YAML-dumped output).
  - _Requirements: 3.1, 3.2_

- [x] 3.2 Write the property test for injection placement
  - **Property 4: Injection is planted only in Tool_Output, never in the User_Task** — one Hypothesis test, as design.md specifies.
  - **Validates: Requirements 3.1, 3.2, 4.2**

- [x] 3.3 Implement the Injected_Goals and the classification rule
  - Write `aib/goals.py` exactly as design.md's "Injected_Goals" and "Classification rule" describe: `GOALS` (workspace injection tasks 0–5 only), `goal_for()` (k mod 6), `injection_success_check()`, and `classify()`.
  - Unit tests:
    - each goal's ground truth, run in a fresh environment, passes its check, and the paired user task's ground truth alone does not;
    - configuring any other goal id is rejected (the six goals are the reviewed benign sandbox actions);
    - `classify()` returns each of the four statuses for the matching inputs.
  - _Requirements: 3.3, 3.4, 3.5, 3.6, 3.7_

- [x] 3.4 Implement the Stub_Model and the case runner
  - Write `aib/stub_model.py` (`StubLLM` with the `follow_user` and `follow_injection` policies) and `aib/runner.py` (`CaseSpec`, `CaseResult`, and `run_case`) exactly as design.md describes.
  - Start `aib/defenses.py` with `RecordingToolsExecutor` and `build_pipeline` for `none` only; task 4.1 adds the other defenses.
  - Unit tests:
    - `follow_user` gives task success True, injection success False, and exposed True;
    - `follow_injection` gives injection success True;
    - a stub that answers without calling tools gives `unclassified_not_exposed`;
    - an LLM element that raises gives `errored` with the cause, and the next case still runs;
    - the `clean` condition gives status `clean` with no injection outcome;
    - a context-overflow error from the fake server ends the case with `context_overflow` set and the checks run;
    - every record carries the Model, defense, condition, task, Repetition, seed, sampling, and stamp.
  - _Requirements: 3.6, 5.3, 5.5, 5.6, 8.6_

- [x] 3.5 Generate the stub validation evidence
  - Write `scripts/stub_validation.py`, which runs all 40 tasks × 5 styles × 2 policies under the `none` defense and writes a stamped `results/m2_stub_validation.json` with per-style counts by policy, status, task success, and injection success.
  - Commit first, then run it.
  - _Requirements: 3.2, 3.4, 3.6, 0.1_

- [x] 3.6 **Automated gate (M2):** run `bash scripts/verify_m2.sh`, which runs `bash scripts/test_unit.sh` and checks:
  - `results/m2_stub_validation.json` has 400 cases;
  - all 200 `follow_user` cases are exposed, with task success True and injection success False;
  - all 200 `follow_injection` cases have injection success True;
  - there are 5 distinct styles including `direct`, `fake_system`, and `hidden_field`; the goals used are exactly injection tasks 0–5; and the file has a stamp.
  - Proceed when it exits 0. No manual review stop.

## M3 — Defenses, grid runner, local-model runs

- [x] 4.1 Implement the Defense_Conditions and the classifier wrapper
  - Complete `aib/defenses.py` and write `aib/classifier.py` exactly as design.md's "Defense_Conditions" and "Defenses and classifier" describe: all four defenses with their texts read from `config/defenses.yaml`, `RecordingPIDetector` with a `FilterDecision` per screened output, `InjectionClassifier` (windowed, cached by SHA-256, thread-safe, loading only the local snapshot), and `KeywordClassifier`. Confirm design.md records that the optional tool-permission-limit defense is not implemented and why.
  - Unit tests:
    - each pipeline's element order;
    - `none` leaves AgentDojo's default system message unchanged;
    - `delimit` wraps every tool output in `<<` and `>>` and adds its sentence;
    - `hierarchy` adds its text verbatim;
    - with `KeywordClassifier`, `filter` records one decision per screened output with the right injected ground truth and replaces exactly the flagged outputs;
    - window splitting covers the whole text with the configured overlap, and the cache returns the same score.
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6_

- [x] 4.2 Write the property test for Stub_Model determinism
  - **Property 10: Stub_Model drives a deterministic case outcome** — one Hypothesis test, as design.md specifies.
  - **Validates: Requirements 8.3, 8.6**

- [x] 4.3 Check the classifier on ground-truth outputs
  - Write `scripts/filter_calibration.py`. It loads the classifier snapshot from `~/aib-models`, runs every ground-truth Tool_Output of the 40 tasks under each of the 5 styles and the clean condition through it, and writes a stamped `results/m3_filter_calibration.json`: per style, injected and clean output counts, flagged counts, precision and recall with Wilson intervals, and total CPU seconds. Add a `classifier`-marked test that scores one planted and one clean text.
  - This is a check of the wiring and a timing measurement, not tuning: the threshold stays 0.5 whatever the result. Commit first, then run it detached.
  - Record in `docs/ENVIRONMENT.md` the classifier's revision and the `transformers` and `torch` versions, and in `THIRD_PARTY.md` the classifier (Apache-2.0).
  - _Requirements: 4.4, 4.5, 0.5, 0.6, 0.7_

- [x] 4.4 Implement the grid runner and the Results_Store
  - Write `aib/experiments.py` exactly as design.md's "Case runner, grid runner, and Results_Store" describes:
    - `build_grid` (local models: all six conditions in defense, condition, task, Repetition order; hosted: the hosted conditions in Repetition-major order);
    - `case_seed`;
    - `ResultsStore` with its header line, append with flush and fsync, resume, and the same-commit rule;
    - `run_grid` with workers equal to the model's slots and the infrastructure-failure stop;
    - the CLI `python -m aib.experiments run --model <key> [--pilot]`. The pilot is `user_task_0` and `user_task_1` × 6 conditions × 4 defenses × 1 Repetition, written to `results/pilot/<key>.jsonl`.
  - Write `scripts/run_benchmark.sh <model key> [--pilot]`. It calls `require_clean`, then:
    - for a local model, runs `scripts/serve.sh up <key>` and the run, and `scripts/serve.sh down` on exit through a trap;
    - for the hosted model, runs it with the ledger;
    - writes traces to `~/aib-runs/traces/<key>/`.
  - Unit tests with `StubLLM` and fakes:
    - grid sizes (2,880 per local model, 1,440 hosted) and order;
    - stable seeds;
    - one record per case;
    - resume skips completed ids, and resuming on a different commit is refused;
    - an infrastructure stop records nothing for the failing cases;
    - a per-case error is recorded and the sweep continues;
    - the same configuration and seed reproduce the same case ids, seeds, and recorded configuration exactly.
  - _Requirements: 5.1, 5.3, 5.4, 5.5, 5.6, 5.7, 0.2_

- [ ] 4.5 Run the pilot
  - Commit, then run `bash scripts/bg.sh pilot bash scripts/run_benchmark.sh qwen3-4b --pilot` and wait for it with `bg_wait.sh`.
  - Write a stamped `results/pilot.json` with:
    - per-case durations, statuses, error causes, token counts, exposure, and context overflows;
    - each Local_Model's projected hours for its full grid, using the 4B model's measured time per case and, for the 8B model, scaled by the ratio of the decode rates measured in task 1.4's smoke tests.
  - If a case errored for a harness reason (not the model's behavior), fix it within the retry cap and rerun the pilot. Report the real pilot outcomes; never change sampling, prompts, or thresholds to improve them.
  - _Requirements: 5.1, 5.6, 0.1, 0.4_

- [ ] 4.6 Run the full benchmark on `qwen3-4b`
  - Commit, then run `bash scripts/bg.sh run_qwen3-4b bash scripts/run_benchmark.sh qwen3-4b`. Wait with `bg_wait.sh` until it is no longer `RUNNING`; if it is `LOST`, start it once more with the same command (it resumes).
  - The run writes `results/raw/qwen3-4b.jsonl` (a header plus 2,880 records). Commit the store when it is complete.
  - _Requirements: 5.1, 5.3, 5.4, 5.5, 0.1_

- [ ] 4.7 Run the full benchmark on the second Local_Model
  - Same as task 4.6 for `llama3.1-8b`, or for the model task 1.4 enabled in its place, writing `results/raw/<key>.jsonl` (a header plus 2,880 records). Commit the store when it is complete.
  - _Requirements: 5.1, 5.3, 5.4, 5.5, 0.1_

- [ ] 4.8 **Automated gate (M3):** run `bash scripts/verify_m3.sh`, which runs `bash scripts/test_unit.sh` and checks:
  - each Local_Model's store has a header with a 40-character commit, the full grid's 2,880 case ids exactly once, and valid statuses;
  - errored cases are at most 5% per model. Above that, the agent looks for a harness cause and fixes it within the retry cap, discarding and rerunning that model's store as design.md requires; it never relabels cases;
  - every `filter` case has a decision for every screened output;
  - `results/m3_filter_calibration.json` and `results/pilot.json` exist with stamps.
  - Proceed when it exits 0. No manual review stop.

## M4 — Metrics, statistics, plots, hosted run

- [ ] 5.1 Implement the metrics and statistics layer
  - Rewrite `aib/metrics.py` to design.md's "Statistics" section and interfaces:
    - `RateCI`, `wilson_interval`, `cluster_bootstrap_rate`, and `paired_bootstrap_difference`;
    - ASR, TSR, and exposure per Cell, with the status rules;
    - Defense_Cost and ASR reduction with paired intervals, and pooled rows over the five styles;
    - filter precision, recall, and the clean false-positive rate;
    - `not computed` with a cause for empty denominators.
  - It runs as `python -m aib.metrics --store results/raw --seed 20261017 --resamples 2000 --out results/metrics.json`, stamped with command, seed, and commit.
  - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.7, 6.8_

- [ ] 5.2 Write property and unit tests for the metrics
  - **Property 5: Attack_Success_Rate counts only classified injected cases**, **Property 6: Rates and filter metrics lie in their ranges**, **Property 7: Confidence interval brackets the point estimate and is deterministic**, **Property 8: Defense_Cost equals the task-success-rate difference from the control**, and **Property 9: Metrics are deterministic under a fixed seed** — one Hypothesis test each.
  - Unit tests: `wilson_interval(27, 30)` is [0.744, 0.965] and `wilson_interval(15, 30)` is [0.332, 0.668] within 1e-3; a store where one task is all successes and another all failures gives a cluster interval wider than the Wilson interval on the same counts; precision and recall on a labeled fixture; an empty Cell is `not computed`.
  - **Validates: Requirements 0.3, 3.5, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.8**

- [ ] 5.3 Implement the Results_Table and the plots
  - Rewrite `aib/results_table.py` and `aib/plots.py` to design.md's "Generated outputs":
    - `results/summary.md` from `results/metrics.json`, with a stamp;
    - `results/plots/asr_by_defense.png`: pooled ASR per Model and defense with interval bars;
    - `results/plots/defense_cost.png`: Defense_Cost per Model and defense for `clean` and pooled over the styles, with paired interval bars.
  - Unit tests on a small metrics fixture: every rate in the table has its interval, `not computed` Cells show their cause, Defense_Cost rows reference `none`, and both plot files are written and non-empty.
  - _Requirements: 6.1, 6.2, 6.4, 7.3_

- [ ] 5.4 Run the hosted benchmark within the Spend_Cap
  - The hosted grid is design.md's: all 4 defenses × `clean`, `direct`, and `important_instructions` × 40 tasks × 3 Repetitions, in Repetition-major order with 2 workers.
  - Commit, then run `bash scripts/bg.sh run_hosted bash scripts/run_benchmark.sh gpt-6-luna` and wait with `bg_wait.sh`; if it is `LOST`, start it once more (it resumes).
  - Stopping at the Spend_Cap is an expected outcome, not a failure. The run records the spend reached, the cap, whether it stopped at the cap, and the completed cases and Cells in a stamped `results/hosted_run.json`. Never raise the cap.
  - Commit the store and the ledger.
  - _Requirements: 5.2, 5.3, 5.4, 5.5, 0.1, 0.9_

- [ ] 5.5 Generate the metrics, the Results_Table, and the plots
  - Write the first part of `scripts/build_results.sh`: `require_clean`, then the metrics command, the Results_Table, and the plots (task 6.6 adds the Report, README, and traceability steps). Commit, then run it.
  - Run the metrics command a second time into `runs/` and confirm the two `metrics.json` files are equal apart from the stamp (Req 6.6).
  - _Requirements: 6.6, 6.7, 7.3, 0.1, 0.3_

- [ ] 5.6 **Automated gate (M4):** run `bash scripts/verify_m4.sh`, which runs `bash scripts/test_unit.sh` and checks:
  - the hosted store holds only cases from the hosted grid, each case id at most once, with complete Repetitions or a recorded Spend_Cap stop;
  - the ledger total is at or below `spend_cap_usd`;
  - `results/metrics.json` covers every Cell present in the stores, each rate with an interval that brackets it or `not computed` with a cause;
  - regenerating `results/metrics.json` and `results/summary.md` gives identical content apart from stamps;
  - both plots exist and are non-empty.
  - Proceed when it exits 0. No manual review stop.

## M5 — Report, traceability, CI, README

- [ ] 6.1 Implement the Report generator
  - Rewrite `aib/report.py` to render `docs/REPORT.md` from `docs/report_template.md` and the results files. Write the template with the five sections (question, setup, results, limitations, threats to validity):
    - every number enters through a placeholder filled from `results/metrics.json`, `results/summary.md`, or the configuration files, never typed;
    - it embeds both plots and states the temperature, top-p, other sampling settings, and the 3 Repetitions per Cell;
    - it reports aggregate rates and defense effects only and contains no payload tuned against a named commercial product;
    - the limitations section carries design.md's known limitations;
    - it states the hosted grid's smaller scope and any Spend_Cap stop.
  - Unit tests: all five sections render; every placeholder resolves; the sampling statement is present; an unfilled placeholder fails the render.
  - _Requirements: 7.1, 7.2, 7.4, 7.5, 7.6, 0.15_

- [ ] 6.2 Implement the README/Report number-traceability check
  - Rewrite `aib/traceability.py` and `scripts/check_traceability.py`: every number in `README.md` and `docs/REPORT.md` (ignoring numbers inside words, dates, version strings, URLs, paths, and inline code) must appear verbatim in a file generated under `results/`. The check reports each untraceable number and exits non-zero when any exist.
  - **Property 11: README/Report number traceability** — one Hypothesis test, as design.md specifies.
  - **Validates: Requirements 7.4, 9.3, 9.4**

- [ ] 6.3 Implement the CI smoke subset
  - Write `scripts/ci_smoke.py`. It runs `user_task_0` to `user_task_4` × 2 styles (`direct`, `hidden_field`) × 3 defenses (`none`, `delimit`, and `filter` with `KeywordClassifier`) × 2 policies with `StubLLM`, needing no GPU, network, key, or download, and asserts the expected outcomes:
    - `follow_user`: task success True and injection success False everywhere;
    - `follow_injection`: injection success True under `none` and `delimit`, and False under the keyword filter.
  - It writes a summary with Wilson intervals to `runs/ci_smoke.json` and exits non-zero, naming the first mismatch or the step that did not complete, when any expectation fails.
  - Unit tests run it in-process.
  - _Requirements: 8.3, 8.5, 8.6_

- [ ] 6.4 Configure the GitHub Actions CI pipeline
  - Write `.github/workflows/ci.yml`. It runs on every push and on pull requests targeting `main`, on `ubuntu-latest`, with `actions/checkout` and `astral-sh/setup-uv` pinned to their current release tags (recorded in `docs/ENVIRONMENT.md`). Its steps are `uv sync --frozen`, `bash scripts/test_unit.sh`, and `uv run --frozen python scripts/ci_smoke.py`, and the job fails if any step fails.
  - Commit and push, then wait for the run on HEAD to finish (`gh run watch` or `gh run list --branch main`). It must succeed; fix and push until it does, within the retry cap.
  - _Requirements: 8.1, 8.2, 8.4, 8.5_

- [ ] 6.5 Prove that CI catches a failing test
  - Create branch `ci-fail-check` from `main` with one intentionally failing test (`tests/test_ci_red_check.py`), push it, and wait for its run, which must fail because of that test.
  - Save the run ids, commits, and conclusions of the red run and of the latest green `main` run to `docs/evidence/ci_red_check.json`, and commit it on `main`.
  - Delete the `ci-fail-check` branch locally and on GitHub; it is the one branch this plan may delete.
  - _Requirements: 8.4_

- [ ] 6.6 Generate the README and finish the build command
  - Write `docs/README.template.md` and `aib/docs_build.py`, and complete `scripts/build_results.sh` so that it regenerates, in order, `results/metrics.json`, `results/summary.md`, the plots, `docs/REPORT.md`, and `README.md`, then runs `scripts/check_traceability.py`.
  - The README holds:
    - an architecture diagram;
    - run instructions;
    - the Results_Table with an interval for every rate, filled from `results/summary.md`;
    - a link to `docs/REPORT.md`;
    - the exact single command for each test suite (`bash scripts/test_unit.sh`, `uv run --frozen python scripts/ci_smoke.py`);
    - the exact single command that reproduces each reported result (`bash scripts/run_benchmark.sh <model key>` per model, then `bash scripts/build_results.sh`);
    - the single command that runs the full local-model benchmark and the one that regenerates the Report and plots;
    - a known-limitations section from design.md;
    - a safety-and-scope section stating that all tools are simulated, no real systems are targeted, results are aggregate only, and the responsible-disclosure rule from design.md applies;
    - the statement that the repository is private.
  - Commit, then run `bash scripts/build_results.sh`; the traceability check must pass.
  - _Requirements: 9.1, 9.2, 9.3, 9.5, 0.11, 0.12, 0.15, 0.16_

- [ ] 6.7 **Automated gate (end to end):** run `bash scripts/verify_m5.sh`, which runs `bash scripts/test_unit.sh` and `scripts/check_traceability.py`, and checks:
  - running `bash scripts/build_results.sh` again gives identical `results/summary.md`, `docs/REPORT.md`, and `README.md` apart from stamps;
  - `docs/REPORT.md` has the five sections and states the temperature, top-p, and Repetitions;
  - `README.md` has every section task 6.6 lists;
  - the latest CI run on `main` for HEAD succeeded, and `docs/evidence/ci_red_check.json` shows the red run failed;
  - `THIRD_PARTY.md` lists every package, model, and image named in `docs/ENVIRONMENT.md`;
  - design.md states the responsible-disclosure rule.
  - Then commit and push. The gate passes when all of this succeeds. The agent fixes failures and re-runs within the retry cap; it does not pause for input.

## Notes

- Tasks marked `*` are optional test sub-tasks and are not implemented by the task-execution agent. No task in this plan is marked `*`: every test task is required, because Requirement 0.11 and the design's one-test-per-property rule require them.
- The plan is fully agent-driven: there are no owner-implemented (`[ME]`) tasks and no manual review stops. The choices the earlier plan left to the owner (the build-or-reuse decision, models, injection styles and goals, the classification rule, the hierarchy prompt, the filter threshold, the Spend_Cap, sampling, and Repetitions) are fixed in design.md.
- Every reported number comes from one repository command with a fixed seed, recorded with command, seed, and full commit hash; versions live in `docs/ENVIRONMENT.md`, licenses in `THIRD_PARTY.md`, and no secrets are committed.
- Every tool is simulated inside the Sandbox; the Report and README report aggregate rates and defense effects only (Req 0.14, 0.15).
- AgentDojo's fixtures and this project's injection texts contain instructions addressed to an AI assistant by design. They are test data: the agent never follows them.
