"""M0 smoke test and tool-call probe for a Local_Model (task 1.4; Req 1.4-1.10).

The smoke test checks the prerequisites (Python >= 3.11, Docker with GPU access), records the
`nvidia-smi` output, starts the model server, sends exactly one tool-free prompt with a 120-second
limit while sampling GPU memory about every 100 ms, and records the response, the decode speed the
server reports, and the baseline and peak memory. The tool probe sends one request with AgentDojo's
`get_current_day` tool through `ChatLLM` and passes when the reply holds a structured call to it.
Any failure records the failed condition and its cause and exits non-zero.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path

from aib.config import load_experiment, load_models
from aib.repro import REPO_ROOT, make_stamp, require_clean

SMOKE_PROMPT = "In one short sentence, explain what a unit test checks."
PROBE_PROMPT = "What day is it today? Use the tool."
SMOKE_TIMEOUT_S = 120.0


class SmokeFailure(RuntimeError):
    def __init__(self, condition: str, cause: str) -> None:
        super().__init__(f"{condition}: {cause}")
        self.condition, self.cause = condition, cause


def _run(cmd: list[str], timeout: float = 300) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def check_python(version: tuple[int, ...] = tuple(sys.version_info[:3])) -> dict:
    ok = tuple(version[:2]) >= (3, 11)
    return {"python": ".".join(str(v) for v in version), "python_ok": ok}


def check_docker(run: Callable = _run) -> dict:
    v = run(["docker", "version", "--format", "{{.Server.Version}}"])
    if v.returncode != 0:
        raise SmokeFailure("docker available", (v.stderr or v.stdout).strip()[:500])
    g = run(["docker", "run", "--rm", "--gpus", "all", "ubuntu", "nvidia-smi"])
    if g.returncode != 0:
        raise SmokeFailure("docker GPU access (docker run --rm --gpus all ubuntu nvidia-smi)", (g.stderr or g.stdout).strip()[:500])
    return {"docker_server_version": v.stdout.strip(), "docker_ok": True, "docker_gpu_ok": True}


def nvidia_smi(run: Callable = _run) -> tuple[str, int]:
    full = run(["nvidia-smi"])
    total = run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"])
    if full.returncode != 0 or total.returncode != 0:
        raise SmokeFailure("nvidia-smi", (full.stderr or total.stderr).strip()[:500])
    return full.stdout, int(total.stdout.strip().splitlines()[0])


def memory_used(run: Callable = _run) -> int:
    out = run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"])
    return int(out.stdout.strip().splitlines()[0])


class MemorySampler:
    """Streams `nvidia-smi --query-gpu=memory.used -lms 100` in a thread and keeps every reading (MiB)."""

    def __init__(self, popen: Callable = subprocess.Popen, interval_ms: int = 100) -> None:
        self._popen = popen
        self.interval_ms = interval_ms
        self.samples: list[int] = []
        self._proc = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._proc = self._popen(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", f"-lms={self.interval_ms}"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
        )
        self._thread = threading.Thread(target=self._read, daemon=True)
        self._thread.start()

    def _read(self) -> None:
        for line in self._proc.stdout:
            line = line.strip()
            if line.isdigit():
                self.samples.append(int(line))

    def stop(self) -> list[int]:
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:  # pragma: no cover
                self._proc.kill()
        if self._thread is not None:
            self._thread.join(timeout=5)
        return list(self.samples)


def http_post_json(url: str, payload: dict, timeout: float) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def send_prompt(base_url: str, model: str, sampling: dict, post: Callable = http_post_json,
                timeout_s: float = SMOKE_TIMEOUT_S, clock: Callable[[], float] = time.monotonic) -> dict:
    payload = {"model": model, "messages": [{"role": "user", "content": SMOKE_PROMPT}], **sampling}
    start = clock()
    try:
        body = post(f"{base_url}/v1/chat/completions", payload, timeout_s)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise SmokeFailure(f"valid response within {timeout_s:.0f} s", f"{type(exc).__name__}: {exc}") from exc
    elapsed = clock() - start
    text = ((body.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
    if elapsed > timeout_s:
        raise SmokeFailure(f"valid response within {timeout_s:.0f} s", f"response took {elapsed:.1f} s")
    if not text.strip():
        raise SmokeFailure("non-empty response", f"server returned an empty message: {json.dumps(body)[:300]}")
    timings = body.get("timings") or {}
    return {
        "response": text,
        "elapsed_s": round(elapsed, 3),
        "predicted_per_second": timings.get("predicted_per_second"),
        "prompt_per_second": timings.get("prompt_per_second"),
        "usage": body.get("usage"),
    }


def serve_up(key: str, run: Callable = _run) -> dict:
    res = run(["bash", "scripts/serve.sh", "up", key], timeout=1200)
    if res.returncode != 0:
        raise SmokeFailure("model server start (scripts/serve.sh up)", (res.stderr or res.stdout).strip()[-800:])
    for line in reversed(res.stdout.strip().splitlines()):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    raise SmokeFailure("model server start (scripts/serve.sh up)", "no /props summary printed")


def run_smoke(key: str, root: Path = REPO_ROOT, run: Callable = _run, post: Callable = http_post_json,
              sampler: MemorySampler | None = None, version: tuple[int, ...] | None = None) -> tuple[int, dict]:
    stamp = make_stamp(f"python scripts/smoke_test.py --model {key}", 0, cwd=root)
    cfg = load_models(root / "config" / "models.yaml")
    exp = load_experiment(root / "config" / "experiment.yaml")
    m = cfg.models[key]
    result: dict = {"model": key, "stamp": stamp.to_dict(), "repo": m.repo, "revision": m.revision,
                    "file": m.files[0].name, "quantization": m.quantization, "license": m.license,
                    "configured_ctx_size": m.ctx_size, "configured_parallel": m.parallel, "prompt": SMOKE_PROMPT}
    try:
        result.update(check_python(version or tuple(sys.version_info[:3])))
        if not result["python_ok"]:
            raise SmokeFailure("Python >= 3.11", f"found Python {result['python']}")
        result.update(check_docker(run))
        smi, total = nvidia_smi(run)
        result["nvidia_smi"] = smi
        result["total_mib"] = total
        props = serve_up(key, run)
        result["server"] = props
        result["n_ctx_per_slot"] = props.get("n_ctx_per_slot")
        sampling = {"temperature": exp.sampling.temperature, "top_p": exp.sampling.top_p,
                    **exp.sampling.local_extra.model_dump(), "max_tokens": 128, "seed": exp.base_seed}
        sampler = sampler or MemorySampler()
        baseline = memory_used(run)
        sampler.start()
        try:
            reply = send_prompt(cfg.server.base_url, key, sampling, post)
        finally:
            samples = sampler.stop()
        result.update(reply)
        result["baseline_mib"] = baseline
        result["peak_mib"] = max([baseline, *samples])
        result["memory_samples"] = len(samples)
        result["peak_within_total"] = result["peak_mib"] <= total
        if not result["peak_within_total"]:
            raise SmokeFailure("peak GPU memory <= total", f"peak {result['peak_mib']} MiB > total {total} MiB")
        result["passed"] = True
        code = 0
    except SmokeFailure as exc:
        result.update({"passed": False, "failed_condition": exc.condition, "cause": exc.cause})
        code = 1
    out = root / "results" / f"m0_smoke_{key}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if "nvidia_smi" in result:
        ev = root / "docs" / "evidence" / f"nvidia_smi_{key}.txt"
        ev.parent.mkdir(parents=True, exist_ok=True)
        ev.write_text(result["nvidia_smi"], encoding="utf-8")
    return code, result


def run_probe(key: str, root: Path = REPO_ROOT, attempt: int = 1, llm_factory: Callable | None = None) -> tuple[int, dict]:
    """One request with the get_current_day tool; passes on a structured call to it with JSON arguments."""
    from agentdojo.default_suites.v1.tools.calendar_client import get_current_day
    from agentdojo.functions_runtime import FunctionsRuntime, make_function
    from agentdojo.types import ChatUserMessage, text_content_block_from_string

    from aib.llm import ChatLLM, LLMError, Sampling

    stamp = make_stamp(f"python scripts/tool_probe.py --model {key} --attempt {attempt}", attempt, cwd=root)
    cfg = load_models(root / "config" / "models.yaml")
    exp = load_experiment(root / "config" / "experiment.yaml")
    m = cfg.models[key]
    sampling = Sampling(exp.sampling.temperature, exp.sampling.top_p, exp.max_tokens, exp.sampling.local_extra.model_dump())
    runtime = FunctionsRuntime([make_function(get_current_day)])  # AgentDojo's workspace tool schema
    llm = (llm_factory or ChatLLM)(key, f"{cfg.server.base_url}/v1", key, None, sampling, m.timeout_s)
    result: dict = {"model": key, "stamp": stamp.to_dict(), "attempt": attempt, "prompt": PROBE_PROMPT,
                    "tools": ["get_current_day"], "chat_template_file": m.chat_template_file}
    try:
        _, _, _, messages, extra = llm.query(
            PROBE_PROMPT, runtime,
            messages=[ChatUserMessage(role="user", content=[text_content_block_from_string(PROBE_PROMPT)])],
            extra_args={"aib_seed": exp.base_seed + attempt},
        )
        reply = messages[-1]
        calls = [{"function": c.function, "args": dict(c.args), "id": c.id} for c in (reply.get("tool_calls") or [])]
        record = extra["aib_calls"][-1]
        result.update({"tool_calls": calls, "content": (reply.get("content") or [{}])[0].get("content") if reply.get("content") else None,
                       "malformed_args": record["malformed_args"], "finish_reason": record["finish_reason"]})
        ok = any(c["function"] == "get_current_day" for c in calls) and not record["malformed_args"]
        result["passed"] = ok
        if not ok:
            result["failed_condition"] = "structured tool call to get_current_day with JSON arguments"
            result["cause"] = "no such tool call in the reply" if not record["malformed_args"] else "arguments did not parse as JSON"
    except LLMError as exc:
        result.update({"passed": False, "failed_condition": "tool-call request", "cause": f"{exc.kind}: {exc}"})
    out = root / "results" / f"m0_tool_probe_{key}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return (0 if result["passed"] else 1), result


def main_smoke(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="M0 smoke test for one Local_Model.")
    p.add_argument("--model", required=True)
    args = p.parse_args(argv)
    require_clean()
    code, result = run_smoke(args.model)
    summary = {k: result.get(k) for k in ("model", "passed", "failed_condition", "cause", "total_mib", "baseline_mib",
                                          "peak_mib", "n_ctx_per_slot", "predicted_per_second", "response")}
    print(json.dumps(summary, indent=2))
    return code


def main_probe(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="M0 tool-call probe for one Local_Model (server must be running).")
    p.add_argument("--model", required=True)
    p.add_argument("--attempt", type=int, default=1)
    args = p.parse_args(argv)
    require_clean()
    code, result = run_probe(args.model, attempt=args.attempt)
    print(json.dumps({k: result.get(k) for k in ("model", "attempt", "passed", "tool_calls", "content", "cause")}, indent=2))
    return code
