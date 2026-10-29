# Requirements: Claude target and second-opinion judge

Adds Claude, paid from the owner's Anthropic API credit, as a second Hosted_Model target (the main spec's Requirement 5.2 allows one or two) and as a second-opinion judge of attack success. The main spec in `.kiro/specs/agent-injection-bench/` is unchanged: no requirement there is edited, the OpenAI Spend_Cap stays as it is, and `config/experiment.yaml` is not touched.

## Glossary

- **Claude_Target**: the Claude model run through the benchmark grid as a model under test (`claude-haiku-5-5`, chosen by the owner on 2026-10-29).
- **Claude_Judge**: the Claude model that reads finished case transcripts and gives a second opinion on whether the injected goal was carried out (`claude-sonnet-5-5`, chosen by the owner).
- **Claude_Run**: one command that sends Claude requests: a benchmark run, a judge run, or the Claude check.
- **Run_Cap**: the owner-set limit, in United States dollars, on the estimated spend of one Claude_Run ($5.00, set by the owner on 2026-10-29).
- **Claude_Ledger**: `results/claude_ledger.jsonl`, the append-only record of every Claude request, refusal, and failure.

## Requirement C1: Claude as a Hosted_Model target

**User story:** As the owner, I want to run the benchmark grid on Claude, so that the results compare a Claude model with the local models and the OpenAI hosted model.

1. THE System SHALL call Claude only through the official Anthropic Python SDK, pinned in `pyproject.toml`.
2. THE System SHALL read the Anthropic API key from the environment variable that `config/models.yaml` names for the Claude_Target, and SHALL NOT write the key to any file, log, or result.
3. WHEN the System sends a Claude request, THE System SHALL send the AgentDojo tool schemas, the pipeline's system message as the Messages API `system` field, and no `temperature`, `top_p`, `top_k`, or seed, and SHALL record in each case's sampling record that the model's default sampling was used.
4. WHEN Claude returns thinking blocks, THE System SHALL send the assistant turn back unchanged, with its blocks in their original order, on the next request of the same case.
5. WHEN Claude stops with `refusal`, THE System SHALL record the stop reason and its category in the call record and SHALL end the turn like any other final answer.
6. WHEN `bash scripts/run_benchmark.sh claude-haiku-5-5` runs, THE System SHALL run the hosted-style grid (the conditions and defenses listed in `config/claude.yaml`, all 40 User_Tasks, 3 Repetitions, Repetition-major order) and SHALL resume from the Results_Store like every other model.
7. WHEN a full Claude benchmark run ends, THE System SHALL write `results/claude_run.json` with the run's spend, the Run_Cap, whether the run stopped at the cap, and the completed cases and Cells.

## Requirement C2: Run_Cap

**User story:** As the owner, I want every Claude command to stop before it can spend more than a set amount, so that one run cannot use up the monthly credit.

1. THE System SHALL read the Run_Cap from `config/claude.yaml`.
2. WHEN a Claude request's worst case (the prompt's characters ÷ 3 as uncached tokens plus `max_tokens` output tokens) would push the Claude_Run's estimated spend past the Run_Cap, THE System SHALL refuse the request before sending it and SHALL stop the Claude_Run with a non-zero exit.
3. THE System SHALL append every Claude request, refusal, and failure to the Claude_Ledger, tagged with the Claude_Run's id, and SHALL count only that run's entries against the Run_Cap.
4. THE System SHALL price each Claude request as an upper bound: uncached prompt tokens at the 5-minute cache-write rate, cached tokens at the cache-hit rate, and Claude Haiku 5.5's higher rates for prompts over 100,000 tokens.
5. THE System SHALL leave the OpenAI Spend_Cap, its ledger, and its total-over-the-project rule unchanged.

## Requirement C3: Second-opinion judge

**User story:** As the owner, I want Claude to give a second opinion on each attack case, so that I can report how often an LLM judge agrees with AgentDojo's rule-based check.

1. THE Claude_Judge SHALL NOT change Injection_Success, any status, or any reported rate (main spec Requirement 2.4).
2. WHEN `python -m aib.judge --store <store>` runs, THE System SHALL send each classified attack case not yet judged (its injected goal, its user prompt, and its transcript) to the Claude_Judge, and SHALL append one verdict per case (carried out or not, with one sentence of evidence) to `results/judge/<model>.jsonl`.
3. THE System SHALL tell the Claude_Judge that the transcript is data that may contain instructions addressed to it, and SHALL request the verdict as JSON through structured outputs.
4. WHEN the Claude_Judge refuses or its reply cannot be parsed, THE System SHALL record the case as refused or unparsed and SHALL leave it out of the agreement figures.
5. WHEN a judge run ends, THE System SHALL write `results/judge/<model>_agreement.json` with the confusion counts against AgentDojo's check, the agreement rate, Cohen's kappa, and a stamp.
6. THE judge run SHALL be a Claude_Run under the Run_Cap.

## Requirement C4: Verification

1. THE unit tests SHALL cover the Claude client, the ledger changes, the configuration, the benchmark wiring, and the judge with fakes only (no key, no network), keeping each new non-trivial module at 80% statement coverage or more under `bash scripts/test_unit.sh`.
2. WHEN `python scripts/claude_check.py` runs with the key set, THE System SHALL send a tool-free request, a tool request, and the tool result back to the Claude_Target within the Run_Cap, and SHALL write the replies, token counts, and estimated spend to `results/claude_check.json`.
