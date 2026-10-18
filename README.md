# Agent Injection Bench

A reproducible benchmark that measures how often tool-using LLM agents follow instructions planted in tool outputs (prompt injection), and how much each defense lowers that rate without hurting task success. Every tool is simulated; nothing reaches a real account, website, or system.

This repository is private. It has no public license, published package, or public release.

The full README, with results, is generated at the end of the run (`bash scripts/build_results.sh`).

## Tests

```bash
bash scripts/test_unit.sh
```
