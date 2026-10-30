# Agent Injection Bench

[![CI](https://github.com/OctaviusLeo/Agent-Injection-Bench/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/OctaviusLeo/Agent-Injection-Bench/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A reproducible benchmark that measures how often tool-using LLM agents follow instructions planted in tool outputs (indirect prompt injection), and how much each defense lowers that rate without hurting legitimate task success. Every tool is simulated: nothing reaches a real account, website, or system.

It runs {{n_user_tasks}} user tasks from AgentDojo's workspace suite (an email, calendar, and file-drive assistant) under {{n_styles}} injection styles and a clean control, with {{n_defenses}} defenses, on {{n_models}} models: two open-weight Qwen models served locally by llama.cpp, and {{hosted_models}}. That is {{n_records}} scored cases. The repository holds the harness that plants the injections and runs the agents, the defenses, the task-clustered bootstrap statistics, and the generators that write every table, plot, and document from the raw case records committed under `results/raw/`.

## Results

Rates come with 95% task-clustered bootstrap intervals, except that a rate with no successes shows its exact one-sided 95% upper bound (marked †); changes are paired against no defense on the same resampled tasks.

{{headline}}

![Task-success cost of each defense](results/plots/defense_cost.png)

**Terms.** ASR (attack success rate) is the share of injected cases in which the agent carried out the injected goal, counted only when the agent actually read the injected tool output. TSR (task success rate) is the share of cases in which it completed the user's task. Defense cost is TSR without a defense minus TSR with it; ASR reduction is ASR without a defense minus ASR with it. Exposure is the share of injected cases in which the agent read the injection.

{{run_facts}}

**Each model pooled over the injection styles it ran** (from `results/summary.md`; † marks an exact upper bound for a rate with no successes):

{{pooled_table}}

![Attack success rate by defense](results/plots/asr_by_defense.png)

The paper-style write-up (question, setup, results, limitations, threats to validity) is [docs/REPORT.md](docs/REPORT.md). The full per-cell table, with exposure, unclassified and errored counts, and filter precision and recall, is [results/summary.md](results/summary.md).

## How it works

```mermaid
flowchart LR
    subgraph Sandbox["Sandbox: AgentDojo workspace suite, simulated tools, no network"]
        FX[(YAML fixture)] --> T[email, calendar, drive tools]
    end
    S[Injection_Style + Injected_Goal] -- planted only in fields the task reads --> FX
    P[AgentDojo pipeline + defense] <--> M[Model: llama.cpp, hosted API, or Stub_Model]
    P <--> T
    P --> C[Task_Success_Check + Injection_Success_Check, same case]
    C --> R[(results/raw/*.jsonl)]
    R --> MET[results/metrics.json] --> TAB[results/summary.md] --> DOC[docs/REPORT.md + README.md + plots]
```

- **Sandbox and tasks.** AgentDojo's workspace suite (`agentdojo==0.1.35`, benchmark `v1.2.2` with the `v1.1.2` task fixes): user tasks with deterministic success checks over the final state and the tool-call trace.
- **Attacks.** Five generic injection styles, planted only in fixture fields that the user task's ground truth reads, so the injection always arrives inside a tool output; each user task is paired with one of six benign sandbox goals. A clean control has no injection.
- **Classification.** An injected case counts toward the attack success rate only when the agent actually received a tool output containing the injection; other injected cases are reported as unclassified.
- **Defenses.** `none`, `delimit` (spotlighting with `<<` `>>`), `hierarchy` (instruction-hierarchy system prompt), and `filter` (a DeBERTa prompt-injection classifier screens every tool output).
- **Statistics.** Task-clustered bootstrap intervals for every rate; paired bootstrap intervals for defense cost and ASR reduction.

## Run it

### Quickstart: recompute the results on a CPU

No GPU, model download, or API key is needed. On Linux (or WSL) with [uv](https://docs.astral.sh/uv/) and Python `3.12`:

```bash
git clone https://github.com/OctaviusLeo/Agent-Injection-Bench.git
cd Agent-Injection-Bench
uv sync --frozen
bash scripts/test_unit.sh
bash scripts/build_results.sh
git status --short
```

`build_results.sh` recomputes every rate and interval from the committed case records in `results/raw/` (seeded bootstrap), then regenerates the Results_Table, the plots, the Report, and this README, and checks that every number in the README and the Report appears in a generated file. Afterwards `git status` shows the regenerated files; they differ from the committed ones only in the commit hash of their stamp lines, and the plots come out byte-identical.

### Full benchmark (GPU and API key)

Python and every script run in WSL `Ubuntu-24.04` with `uv`; the llama.cpp server runs in Docker on the GPU (see `docs/ENVIRONMENT.md` for every version). From the repository root:

```bash
uv sync --frozen
uv run --frozen python scripts/fetch_models.py
bash scripts/run_all_local.sh
bash scripts/run_benchmark.sh gpt-6-luna
bash scripts/run_benchmark.sh claude-haiku-5-5
uv run --frozen python -m aib.judge --store results/raw/<model key>.jsonl
bash scripts/build_results.sh
```

- Full local-model benchmark (every enabled local model, one after the other): `bash scripts/run_all_local.sh`
- One model's benchmark (resumable; reproduces that model's store): `bash scripts/run_benchmark.sh <model key>`
- The same run under a watchdog that resumes it with a different launch method (setsid, systemd, tmux, child process) after a crash, a vanished job, a stall, or a slowdown: `uv run --frozen python -m aib.watchdog --model <model key>`
- The OpenAI hosted model needs `OPENAI_API_KEY` (read from `~/.config/aib/secrets.env` by `scripts/wsl_run.sh`) and stops before its Spend_Cap.
- The Claude target (`claude-haiku-5-5`) and the second-opinion judge (`claude-sonnet-5-5`) are billed to the Anthropic API key in the Windows user variable that `config/models.yaml` names, forwarded into WSL with `WSLENV` (see `docs/ENVIRONMENT.md`). Every Claude command stops before the per-run cap in `config/claude.yaml`. The judge gives a second opinion on each attack case and reports how often it agrees with AgentDojo's rule-based check; it never changes a reported rate.
- Regenerate the metrics, the Results_Table, the plots, the Report, and this README, then run the traceability check: `bash scripts/build_results.sh`

## Reproduce each result

| Result | Command |
| --- | --- |
| A model's case records, `results/raw/<model key>.jsonl` | `bash scripts/run_benchmark.sh <model key>` |
| Every rate and interval, `results/metrics.json` | `uv run --frozen python -m aib.metrics --store results/raw --seed 20261017 --resamples 2000 --out results/metrics.json` |
| The Results_Table, the plots, the Report, and this README | `bash scripts/build_results.sh` |
| Hosted run summary, `results/hosted_run.json` | `bash scripts/run_benchmark.sh gpt-6-luna` |
| Claude run summary, `results/claude_run.json` | `bash scripts/run_benchmark.sh claude-haiku-5-5` |
| Claude check, `results/claude_check.json` | `uv run --frozen python scripts/claude_check.py` |
| Second-opinion verdicts and agreement, `results/judge/<model key>.jsonl` and `results/judge/<model key>_agreement.json` | `uv run --frozen python -m aib.judge --store results/raw/<model key>.jsonl` |
| Classifier check on ground-truth outputs, `results/m3_filter_calibration.json` | `uv run --frozen python scripts/filter_calibration.py` |
| Stub_Model validation, `results/m2_stub_validation.json` | `uv run --frozen python scripts/stub_validation.py` |
| Task catalog, `results/m1_catalog.json` | `uv run --frozen python scripts/catalog.py` |

## Tests

- Unit and property tests with the per-module coverage check (what CI runs): `bash scripts/test_unit.sh`
- CI smoke subset (Stub_Model, no GPU, network, or key): `uv run --frozen python scripts/ci_smoke.py`
- README and Report number traceability: `uv run --frozen python scripts/check_traceability.py`

GitHub Actions runs the first two on every push and on pull requests to `main`.

## Known limitations

{{limitations}}

## Safety and scope

- **All tools are simulated.** The agent only ever calls Python functions over an in-memory sandbox loaded from a local fixture; no real account, website, or system is targeted, and a unit test fails if a tool opens a socket.
- **The injected goals are benign sandbox actions** (send an email to a fictional outside address, delete a fixture file, create a calendar event, forward a fixture email or code) and never ask for content-policy-violating output.
- **Results are aggregate only.** The Report and this README give attack success rates and defense effects; the injection texts are generic published templates, none tuned against a named commercial product.
- **Responsible disclosure.** If benchmarking reveals a reproducible vulnerability in a specific real third-party product, publication of that payload stops and the vendor's responsible-disclosure process is followed ([design.md](.kiro/specs/agent-injection-bench/design.md), "Responsible disclosure and payload restraint").
- Third-party code, models, and their licenses are listed in [THIRD_PARTY.md](THIRD_PARTY.md).

## Provenance and license

Code is released under the [MIT License](LICENSE). Third-party packages, models and images are listed with their licenses in [THIRD_PARTY.md](THIRD_PARTY.md); no third-party code or model weights are redistributed here, and the committed case records quote only short strings from AgentDojo's fictional workspace fixture (MIT). There is no published package or release.

This README is generated by `python -m aib.docs_build` from `docs/README.template.md` and the files under `results/`; do not edit it by hand.

- README: command `{{readme_command}}`, seed {{readme_seed}}, commit `{{readme_commit}}`.
- Results_Table: command `{{summary_command}}`, seed {{summary_seed}}, commit `{{summary_commit}}`.
- Metrics: command `{{metrics_command}}`, seed {{metrics_seed}}, commit `{{metrics_commit}}`.

## How this was built

This project was built with Claude Code (Anthropic's agentic coding tool) under a spec-driven workflow: written requirements, then a design, then a task list organized into milestones (the spec files under `.kiro/specs/`), each milestone ending in an automated gate script that had to pass before the next one started. The commit history shows this openly; commits carry a `Co-Authored-By` trailer naming the model that helped write them. Every reported number comes from a script in this repository and records the command, seed and commit that produced it, so nothing was typed, estimated or borrowed. I set the goals and constraints in the specification, I am accountable for the verification and the results, and I can explain and change any part of this code.
