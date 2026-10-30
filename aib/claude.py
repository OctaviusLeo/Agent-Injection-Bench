"""Claude helpers and the live Claude check (claude-provider Req C2, C4.2).

`make_claude_ledger` opens the Claude ledger for one Claude run: a fresh run id, so only that run's requests count
against `run_cap_usd`. `make_claude_llm` builds the Claude target's `ClaudeLLM` from config/models.yaml and
config/claude.yaml. The check sends a tool-free request, a request with AgentDojo's `get_current_day` tool, and the
tool's result back (replaying Claude's blocks, thinking included when the model thought; each turn records how many
thinking blocks it returned), and writes results/claude_check.json.
"""

from __future__ import annotations

import datetime as dt
import json
import sys
import uuid
from pathlib import Path

from agentdojo.default_suites.v1.tools.calendar_client import get_current_day
from agentdojo.functions_runtime import FunctionsRuntime, make_function
from agentdojo.types import ChatToolResultMessage, ChatUserMessage, get_text_content_as_str, text_content_block_from_string

from aib.claude_llm import CLAUDE_CONTENT, ClaudeLLM
from aib.config import AllConfig, ClaudeConfig, TieredPrices, load_all
from aib.hosted import TOOL_FREE_PROMPT, TOOL_PROMPT
from aib.ledger import Prices, SpendCapExceeded, SpendLedger
from aib.llm import LLMError, MissingSecretError
from aib.repro import REPO_ROOT, make_stamp, require_clean

CHECK_DAY = "2024-05-15"  # the tool output sent back in the check: AgentDojo's workspace calendar day


def ledger_prices(p: TieredPrices) -> Prices:
    """config/claude.yaml prices -> the ledger's Prices, with the long-context tier when the model has one."""
    tier = None if p.long_context is None else Prices(cache_write=p.long_context.cache_write,
                                                       cached_input=p.long_context.cached_input,
                                                       output=p.long_context.output)
    return Prices(cache_write=p.cache_write, cached_input=p.cached_input, output=p.output,
                  long_context_tokens=p.long_context_tokens, long_context=tier)


def new_run_id(label: str) -> str:
    """`<label>-<UTC time>-<random>`: unique per run, and sortable by start time within a label."""
    return f"{label}-{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:6]}"


def make_claude_ledger(claude: ClaudeConfig, model: str, root: Path = REPO_ROOT, run_id: str | None = None) -> SpendLedger:
    """The Claude ledger for one Claude run, priced for `model` (the target or the judge)."""
    return SpendLedger(root / claude.ledger, claude.run_cap_usd, ledger_prices(claude.prices_per_million_usd[model]),
                       claude.worst_case_chars_per_token, run_id=run_id or new_run_id(model))


def make_claude_llm(cfg: AllConfig, model: str, ledger: SpendLedger, purpose: str = "", client=None) -> ClaudeLLM:
    c, m = cfg.claude, cfg.models.models[model]
    return ClaudeLLM(model, m.name, m.api_key_env, c.max_tokens, c.effort, m.timeout_s, ledger, purpose=purpose,
                     client=client)


def _user(text: str) -> ChatUserMessage:
    return ChatUserMessage(role="user", content=[text_content_block_from_string(text)])


def _ask(llm: ClaudeLLM, messages: list, runtime: FunctionsRuntime) -> tuple[list, dict]:
    _, _, _, out, extra = llm.query("", runtime, messages=messages, extra_args={})
    reply, rec = out[-1], extra["aib_calls"][-1]
    return list(out), {
        "content": get_text_content_as_str(reply["content"]) if reply["content"] else None,
        "tool_calls": [{"function": c.function, "args": dict(c.args)} for c in (reply["tool_calls"] or [])],
        "thinking_blocks": sum(b["type"] in ("thinking", "redacted_thinking") for b in reply.get(CLAUDE_CONTENT, [])),
        **{k: rec[k] for k in ("response_model", "finish_reason", "stop_category", "prompt_tokens", "cached_tokens",
                               "cache_write_tokens", "completion_tokens", "cost_usd")},
    }


def run_check(root: Path = REPO_ROOT, client=None) -> tuple[int, dict]:
    stamp = make_stamp("python scripts/claude_check.py", 0, cwd=root)
    cfg = load_all(root / "config")
    c = cfg.claude
    ledger = make_claude_ledger(c, c.model, root, run_id=new_run_id("claude_check"))
    result: dict = {"model": c.model, "stamp": stamp.to_dict(), "run_id": ledger.run_id, "run_cap_usd": c.run_cap_usd,
                    "effort": c.effort, "max_tokens": c.max_tokens, "sampling": "model default (none sent)"}
    tool_runtime = FunctionsRuntime([make_function(get_current_day)])
    try:
        llm = make_claude_llm(cfg, c.model, ledger, purpose="claude_check", client=client)
        _, result["tool_free"] = _ask(llm, [_user(TOOL_FREE_PROMPT)], FunctionsRuntime([]))
        out, result["tool_request"] = _ask(llm, [_user(TOOL_PROMPT)], tool_runtime)
        results = [ChatToolResultMessage(role="tool", content=[text_content_block_from_string(CHECK_DAY)],
                                         tool_call_id=call.id, tool_call=call, error=None)
                   for call in (out[-1]["tool_calls"] or [])]
        if results:
            _, result["tool_result_turn"] = _ask(llm, [*out, *results], tool_runtime)
        ok_text = bool((result["tool_free"]["content"] or "").strip())
        ok_tool = any(t["function"] == "get_current_day" for t in result["tool_request"]["tool_calls"])
        ok_after = bool(((result.get("tool_result_turn") or {}).get("content") or "").strip())
        result["passed"] = ok_text and ok_tool and ok_after
        if not result["passed"]:
            result["failed_condition"] = ("non-empty tool-free response" if not ok_text else
                                          "tool call to get_current_day" if not ok_tool else
                                          "non-empty answer after the tool result")
    except MissingSecretError as exc:
        result.update({"passed": False, "failed_condition": "Anthropic API key accepted", "cause": str(exc)})
    except (LLMError, SpendCapExceeded) as exc:
        result.update({"passed": False, "failed_condition": "Claude request", "cause": f"{type(exc).__name__}: {exc}"})
    result["run_spend_usd"] = round(ledger.total_usd, 9)  # drops float noise such as 0.00015749999999999998
    result["within_cap"] = ledger.total_usd <= c.run_cap_usd
    out_path = root / "results" / "claude_check.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return (0 if result["passed"] and result["within_cap"] else 1), result


def main() -> int:
    require_clean()
    code, result = run_check()
    print(json.dumps({k: result.get(k) for k in ("model", "passed", "failed_condition", "cause", "run_spend_usd",
                                                  "run_cap_usd")}, indent=2))
    for key in ("tool_free", "tool_request", "tool_result_turn"):
        if key in result:
            r = result[key]
            print(f"{key}: model={r['response_model']} stop={r['finish_reason']} thinking={r['thinking_blocks']} "
                  f"tokens={r['prompt_tokens']}/{r['cached_tokens']}/{r['completion_tokens']} cost=${r['cost_usd']:.6f} "
                  f"content={r['content']!r} tool_calls={r['tool_calls']}")
    return code


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
