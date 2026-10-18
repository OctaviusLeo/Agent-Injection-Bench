"""Hosted model helpers and the M0 hosted check (task 1.5; Req 1.11, 5.2, 0.9, 0.13).

The check sends one tool-free request and one request with AgentDojo's `get_current_day` tool to
the hosted model through `ChatLLM` and the `SpendLedger`. If the API rejects a sampling parameter
(HTTP 400 naming it), that parameter is dropped and the request retried; the parameters that worked
are written to config/hosted.yaml as `accepted_params`, and every later hosted request uses them.
"""

from __future__ import annotations

import json
import re
import sys
from collections.abc import Callable
from pathlib import Path

from agentdojo.default_suites.v1.tools.calendar_client import get_current_day
from agentdojo.functions_runtime import FunctionsRuntime, make_function
from agentdojo.types import ChatUserMessage, text_content_block_from_string

from aib.config import ExperimentConfig, HostedConfig, ModelsConfig, load_experiment, load_hosted, load_models
from aib.ledger import Prices, SpendCapExceeded, SpendLedger
from aib.llm import HOSTED_SAMPLING_PARAMS, ChatLLM, LLMError, MissingSecretError, Sampling, UnsupportedParameterError
from aib.repro import REPO_ROOT, make_stamp, require_clean

TOOL_FREE_PROMPT = "In one short sentence, explain what a unit test checks."
TOOL_PROMPT = "What day is it today? Use the tool."


def make_ledger(hosted: HostedConfig, root: Path = REPO_ROOT) -> SpendLedger:
    p = hosted.prices_per_million_usd
    return SpendLedger(root / hosted.ledger, hosted.spend_cap_usd,
                       Prices(cache_write=p.cache_write, cached_input=p.cached_input, output=p.output),
                       hosted.worst_case_chars_per_token)


def make_hosted_llm(models: ModelsConfig, hosted: HostedConfig, exp: ExperimentConfig, ledger: SpendLedger,
                    accepted_params: tuple[str, ...] | None = None, purpose: str = "",
                    base_url: str | None = None) -> ChatLLM:
    m = models.models[hosted.model]
    sampling = Sampling(exp.sampling.temperature, exp.sampling.top_p, hosted.max_completion_tokens)
    accepted = accepted_params if accepted_params is not None else (
        tuple(hosted.accepted_params) if hosted.accepted_params is not None else HOSTED_SAMPLING_PARAMS)
    return ChatLLM(hosted.model, base_url or m.base_url, hosted.model, m.api_key_env, sampling, m.timeout_s,
                   reasoning_effort=hosted.reasoning_effort, ledger=ledger, provider="openai",
                   accepted_params=accepted, purpose=purpose)


def unsupported_param(exc: UnsupportedParameterError, candidates: tuple[str, ...]) -> str | None:
    if exc.param in candidates:
        return exc.param
    for name in candidates:
        if re.search(rf"\b{name}\b", str(exc)):
            return name
    return None


def send_with_negotiation(send: Callable[[tuple[str, ...]], dict], params: tuple[str, ...]) -> tuple[tuple[str, ...], dict, list[str]]:
    """Call `send(params)`; on a 400 naming one of `params`, drop it and retry. Returns (params, reply, dropped)."""
    dropped: list[str] = []
    while True:
        try:
            return params, send(params), dropped
        except UnsupportedParameterError as exc:
            name = unsupported_param(exc, params)
            if name is None:
                raise
            dropped.append(name)
            params = tuple(p for p in params if p != name)


def _request(llm: ChatLLM, prompt: str, runtime: FunctionsRuntime, seed: int) -> dict:
    _, _, _, messages, extra = llm.query(
        prompt, runtime, messages=[ChatUserMessage(role="user", content=[text_content_block_from_string(prompt)])],
        extra_args={"aib_seed": seed},
    )
    reply = messages[-1]
    rec = extra["aib_calls"][-1]
    return {
        "content": reply["content"][0]["content"] if reply.get("content") else None,
        "tool_calls": [{"function": c.function, "args": dict(c.args)} for c in (reply.get("tool_calls") or [])],
        "response_model": rec["response_model"],
        "prompt_tokens": rec["prompt_tokens"],
        "cached_tokens": rec["cached_tokens"],
        "completion_tokens": rec["completion_tokens"],
        "cost_usd": rec["cost_usd"],
        "finish_reason": rec["finish_reason"],
    }


def write_accepted_params(path: Path, params: tuple[str, ...]) -> None:
    text = path.read_text(encoding="utf-8")
    line = f"accepted_params: [{', '.join(params)}]  # written by scripts/hosted_check.py (task 1.5)"
    new, n = re.subn(r"(?m)^accepted_params:.*$", line, text)
    if n != 1:
        raise ValueError(f"{path}: expected exactly one accepted_params line")
    path.write_text(new, encoding="utf-8")


def run_check(root: Path = REPO_ROOT, base_url: str | None = None) -> tuple[int, dict]:
    stamp = make_stamp("python scripts/hosted_check.py", 0, cwd=root)
    models = load_models(root / "config" / "models.yaml")
    hosted = load_hosted(root / "config" / "hosted.yaml")
    exp = load_experiment(root / "config" / "experiment.yaml")
    ledger = make_ledger(hosted, root)
    result: dict = {"model": hosted.model, "stamp": stamp.to_dict(), "spend_cap_usd": hosted.spend_cap_usd,
                    "reasoning_effort": hosted.reasoning_effort, "max_completion_tokens": hosted.max_completion_tokens,
                    "requested_sampling": {"temperature": exp.sampling.temperature, "top_p": exp.sampling.top_p,
                                           "seed": "per request"}}
    tool_runtime = FunctionsRuntime([make_function(get_current_day)])
    try:
        def send(prompt, runtime):
            def _send(params):
                llm = make_hosted_llm(models, hosted, exp, ledger, params, purpose="m0_hosted_check", base_url=base_url)
                return _request(llm, prompt, runtime, exp.base_seed)
            return _send

        params, tool_free, dropped1 = send_with_negotiation(send(TOOL_FREE_PROMPT, FunctionsRuntime([])),
                                                            HOSTED_SAMPLING_PARAMS)
        params, tool_reply, dropped2 = send_with_negotiation(send(TOOL_PROMPT, tool_runtime), params)
        result.update({"accepted_params": list(params), "dropped_params": dropped1 + dropped2,
                       "tool_free": tool_free, "tool_request": tool_reply})
        ok_text = bool((tool_free["content"] or "").strip())
        ok_tool = any(c["function"] == "get_current_day" for c in tool_reply["tool_calls"])
        result["passed"] = ok_text and ok_tool
        if not result["passed"]:
            result["failed_condition"] = "non-empty tool-free response" if not ok_text else "tool call to get_current_day"
        write_accepted_params(root / "config" / "hosted.yaml", params)
    except MissingSecretError as exc:
        result.update({"passed": False, "failed_condition": "OPENAI_API_KEY set", "cause": str(exc)})
    except (LLMError, SpendCapExceeded) as exc:
        result.update({"passed": False, "failed_condition": "hosted request", "cause": f"{type(exc).__name__}: {exc}"})
    result["ledger_total_usd"] = ledger.total_usd
    result["within_cap"] = ledger.total_usd <= hosted.spend_cap_usd
    out = root / "results" / "m0_hosted_check.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return (0 if result["passed"] and result["within_cap"] else 1), result


def main() -> int:
    require_clean()
    code, result = run_check()
    print(json.dumps({k: result.get(k) for k in ("model", "passed", "failed_condition", "cause", "accepted_params",
                                                  "dropped_params", "ledger_total_usd")}, indent=2))
    for key in ("tool_free", "tool_request"):
        if key in result:
            r = result[key]
            print(f"{key}: model={r['response_model']} tokens={r['prompt_tokens']}/{r['cached_tokens']}/{r['completion_tokens']} "
                  f"cost=${r['cost_usd']:.6f} content={r['content']!r} tool_calls={r['tool_calls']}")
    return code


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
