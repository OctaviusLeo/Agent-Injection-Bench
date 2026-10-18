# Agent Injection Bench

A reproducible benchmark that measures how often tool-using LLM agents follow instructions planted in tool outputs (prompt injection), and how much each defense lowers that rate without hurting task success. Every tool is simulated; nothing reaches a real account, website, or system.

This repository is private. It has no public license, published package, or public release.

The full README, with results, is generated at the end of the run (`bash scripts/build_results.sh`).

## Tests

```bash
bash scripts/test_unit.sh
```

## How this was built

This project was built with Claude Code (Anthropic's agentic coding tool) as a pair programmer, under a spec-driven workflow: written requirements, then a design, then a task list with milestone gates, each gate reviewed before the next milestone started. The commit history shows this openly; commits carry a `Co-Authored-By` trailer naming the model that helped write them. Every reported number comes from a script in this repository and records the command, seed and commit that produced it, so nothing was typed, estimated or borrowed. I own the design decisions, the verification and the results, and I can explain and change any part of this code.
