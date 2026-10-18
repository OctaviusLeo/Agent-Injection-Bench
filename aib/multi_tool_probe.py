"""Multi-tool probe: can the served model make two tool calls in one turn that llama-server parses?

Added after the first llama3.1-8b benchmark attempt (2026-10-18): Llama-3.1-8B-Instruct joins two JSON tool
calls with "; ", a form llama.cpp's Llama 3.x parser rejects with HTTP 500, which stopped the run. This probe
sends three prompts that each need two tools, with every one of the 24 workspace tools offered, under eight
seeds and three `parallel_tool_calls` settings (unset, false, true), and counts parsed turns and failures.
It drove the 8B-slot fallback decision recorded in design.md's "M0 outcome".

Usage (server running): python -m aib.multi_tool_probe --model <key> [--variant <label>] [--seeds 8]
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from collections.abc import Callable

PROMPTS = (
    "Do I have any unread emails, and what day is it today? Use the tools for both.",
    "Create a 1-hour calendar event 'Lunch' with sarah.baker@gmail.com on 2024-05-19 at 12:00, and also check today's date.",
    "Search my emails from lily.white@gmail.com and also search my calendar for 'Introductory meeting'.",
)
MODES = ("unset", "false", "true")


def post(url: str, body: dict, timeout: float = 120) -> tuple[int, dict]:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read() or b"{}")
        except json.JSONDecodeError:
            return e.code, {}


def probe(url: str, tools: list[dict], sampling: dict, seeds: int, send: Callable = post) -> dict:
    out = {}
    for mode in MODES:
        counts: dict[str, int] = {}
        for prompt in PROMPTS:
            for seed in range(seeds):
                body = {"model": "probe", "messages": [{"role": "user", "content": prompt}], "tools": tools,
                        "tool_choice": "auto", **sampling, "seed": seed, "max_tokens": 512}
                if mode != "unset":
                    body["parallel_tool_calls"] = mode == "true"
                status, data = send(url, body)
                if status == 200:
                    n = len(((data.get("choices") or [{}])[0].get("message") or {}).get("tool_calls") or [])
                    key = f"parsed_{n}_calls"
                else:
                    key = f"http_{status}"
                counts[key] = counts.get(key, 0) + 1
        out[mode] = dict(sorted(counts.items()))
    total = len(PROMPTS) * seeds * len(MODES)
    parsed = sum(v for m in out.values() for k, v in m.items() if k.startswith("parsed_"))
    return {"per_parallel_tool_calls_setting": out, "requests": total, "parsed": parsed, "parse_rate": parsed / total}


def main(argv: list[str] | None = None) -> int:
    from aib.config import load_all
    from aib.llm import tool_schema
    from aib.repro import REPO_ROOT, make_stamp, require_clean
    from aib.server import http_get_json, props_summary
    from aib.suite import load_workspace_suite

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", required=True)
    p.add_argument("--variant", default="gguf-template")
    p.add_argument("--seeds", type=int, default=8)
    args = p.parse_args(argv)
    require_clean()
    cfg = load_all(REPO_ROOT / "config")
    command = f"python -m aib.multi_tool_probe --model {args.model} --variant {args.variant} --seeds {args.seeds}"
    stamp = make_stamp(command, 0).to_dict()
    _, props = http_get_json(f"{cfg.models.server.base_url}/props")
    sampling = {"temperature": cfg.experiment.sampling.temperature, "top_p": cfg.experiment.sampling.top_p,
                **cfg.experiment.sampling.local_extra.model_dump()}
    tools = [tool_schema(t) for t in load_workspace_suite().tools]
    result = probe(f"{cfg.models.server.base_url}/v1/chat/completions", tools, sampling, args.seeds)
    out = {"stamp": stamp, "model": args.model, "variant": args.variant, "server": props_summary(props),
           "prompts": list(PROMPTS), "seeds": args.seeds, "sampling": sampling, **result}
    path = REPO_ROOT / "results" / f"m0_multi_tool_probe_{args.model}_{args.variant}.json"
    path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("model", "variant", "per_parallel_tool_calls_setting", "parse_rate")}, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
