# Tasks: Claude target and second-opinion judge

Each task commits as `task C<id>: <summary>` after its check passes. The gate is `bash scripts/test_unit.sh` plus `uvx ruff@0.17.3 check .`.

- [x] C1 Ledger: run-scoped totals and the long-context price tier (Req C2.2-C2.5)
  - `SpendLedger(..., run_id=...)` counts only that run's lines and tags every line it writes; without `run_id` it behaves as before.
  - `Prices` takes an optional tier: above `long_context_tokens` prompt tokens a request is priced at `long_context`.
  - Unit tests for both, and for the unchanged OpenAI behaviour.
- [x] C2 Configuration (Req C1.2, C2.1)
  - The `anthropic` provider and the `claude-haiku-5-5` entry in `config/models.yaml`; `config/claude.yaml` with the Run_Cap, request settings, grid, ledger, judge, and prices; cross-file checks in `load_all`.
  - Unit tests: the files load; a missing price, an unknown condition, a non-Claude target, or a negative Run_Cap is rejected with an error naming the field.
- [x] C3 `ClaudeLLM` (Req C1.1-C1.5)
  - Message conversion, request parameters, response parsing with verbatim replay of Claude's content blocks, refusal records, error mapping and retries, ledger reservation and settlement.
  - Unit tests against a fake client built from the SDK's own response types.
- [x] C4 Benchmark wiring and the Claude check (Req C1.6, C1.7, C4.2)
  - `run_cli` and the watchdog run the Claude grid with the Claude ledger and client; `results/claude_run.json`; `scripts/claude_check.py`.
  - Unit tests with stubs and fakes.
- [x] C5 Judge (Req C3)
  - `aib/judge.py`, `scripts/judge.py`, and unit tests with a fake client.
- [ ] C6 Dependency pin and documentation
  - `anthropic==1.15.0` in `pyproject.toml` and `uv.lock`; the Claude section of `docs/ENVIRONMENT.md`; the README template and the regenerated README.
- [ ] C7 Gate and live smoke test (Req C4)
  - `bash scripts/test_unit.sh` and `uvx ruff@0.17.3 check .` pass.
  - With the key forwarded: `python scripts/claude_check.py` passes within the Run_Cap, and `python -m aib.judge --store results/raw/qwen3-4b.jsonl --limit 2` writes two verdicts.
