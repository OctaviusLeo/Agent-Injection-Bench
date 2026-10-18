"""Model server control: llama.cpp `llama-server` in Docker, one model at a time (task 1.3; Req 1.8).

`up <key>` waits until the GPU is idle (the idle-memory rule), starts the Compose service `llm` of
project `aib` with that model's settings, waits for `/health` to return 200, and prints `/props`.
`down` saves the server log under ~/aib-runs/server-logs/ and stops the service. Only the `aib`
project's containers are ever touched.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path

from aib.config import ModelsConfig, load_models
from aib.repro import REPO_ROOT

COMPOSE_FILE = REPO_ROOT / "docker" / "compose.yaml"
IDLE_FILE = REPO_ROOT / "runs" / "gpu_idle_mib.txt"


class ServerError(RuntimeError):
    pass


class GpuBusyError(ServerError):
    """GPU memory stayed more than the margin above the idle reading for the whole wait."""


class ServerTimeoutError(ServerError):
    pass


def _run(cmd: list[str], env: dict | None = None, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, check=check, env=env)


def gpu_memory_used_mib(run: Callable = _run) -> int:
    out = run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]).stdout
    return int(out.strip().splitlines()[0])


def gpu_memory_total_mib(run: Callable = _run) -> int:
    out = run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"]).stdout
    return int(out.strip().splitlines()[0])


def read_idle_mib(path: Path | None = None) -> int | None:
    try:
        return int((path or IDLE_FILE).read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def wait_gpu_idle(idle_mib: int, margin_mib: int = 1024, step_s: int = 300, max_wait_s: int = 1200,
                  reader: Callable[[], int] = gpu_memory_used_mib, sleep: Callable[[float], None] = time.sleep) -> int:
    """Return the current reading once it is within `margin_mib` of `idle_mib`; wait in steps, then raise."""
    waited = 0
    while True:
        used = reader()
        if used <= idle_mib + margin_mib:
            return used
        if waited >= max_wait_s:
            raise GpuBusyError(
                f"GPU memory used {used} MiB is more than {margin_mib} MiB above the idle reading {idle_mib} MiB "
                f"after waiting {waited} s; another process holds the GPU"
            )
        print(f"GPU busy: {used} MiB used vs idle {idle_mib} MiB; waiting {step_s} s", flush=True)
        sleep(step_s)
        waited += step_s


def compose_env(model_key: str, cfg: ModelsConfig, base_env: dict | None = None) -> dict:
    m = cfg.models[model_key]
    if m.provider != "llamacpp":
        raise ServerError(f"{model_key} is not a local model")
    env = dict(os.environ if base_env is None else base_env)
    if not env.get("AIB_MODELS"):
        env["AIB_MODELS"] = str(Path.home() / "aib-models")
    flags = list(cfg.server.flags)
    if m.chat_template_file:
        flags += ["--chat-template-file", f"/models/{m.chat_template_file}"]
    env.update({
        "AIB_MODEL_PATH": f"{model_key}/{m.files[0].name}",
        "AIB_CTX_SIZE": str(m.ctx_size),
        "AIB_PARALLEL": str(m.parallel),
        "AIB_SERVER_FLAGS": " ".join(flags),
    })
    return env


def _compose(cfg: ModelsConfig) -> list[str]:
    return ["docker", "compose", "-p", cfg.server.compose_project, "-f", str(COMPOSE_FILE)]


def compose_up(model_key: str, cfg: ModelsConfig, run: Callable = _run) -> None:
    env = compose_env(model_key, cfg)
    run([*_compose(cfg), "up", "-d", "--force-recreate", cfg.server.service], env=env)


def compose_down(cfg: ModelsConfig, run: Callable = _run, log_dir: Path | None = None) -> Path | None:
    """Save the service log (if any) and stop the project's containers."""
    env = dict(os.environ)
    env.setdefault("AIB_MODELS", str(Path.home() / "aib-models"))
    # the compose file requires these variables even for `logs` and `down`
    env.update({k: env.get(k) or "unset" for k in ("AIB_MODEL_PATH", "AIB_CTX_SIZE", "AIB_PARALLEL", "AIB_SERVER_FLAGS")})
    saved = None
    logs = run([*_compose(cfg), "logs", "--no-color", cfg.server.service], env=env, check=False)
    if getattr(logs, "stdout", ""):
        log_dir = log_dir or Path(os.environ.get("AIB_RUNS", str(Path.home() / "aib-runs"))) / "server-logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        saved = log_dir / f"llm-{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.log"
        saved.write_text(logs.stdout, encoding="utf-8")
    run([*_compose(cfg), "down"], env=env)
    return saved


def http_get_json(url: str, timeout: float = 5.0) -> tuple[int, dict]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            body = resp.read().decode("utf-8") or "{}"
            return resp.status, json.loads(body)
    except urllib.error.HTTPError as exc:
        return exc.code, {}
    except (urllib.error.URLError, ConnectionError, TimeoutError, OSError, json.JSONDecodeError):
        return 0, {}


def wait_health(base_url: str, timeout_s: float = 900, get: Callable = http_get_json,
                sleep: Callable[[float], None] = time.sleep, clock: Callable[[], float] = time.monotonic,
                interval_s: float = 2.0) -> float:
    """Poll `/health` until it returns 200; return the seconds waited or raise ServerTimeoutError."""
    start = clock()
    while True:
        status, _ = get(f"{base_url}/health")
        if status == 200:
            return clock() - start
        if clock() - start >= timeout_s:
            raise ServerTimeoutError(f"{base_url}/health did not return 200 within {timeout_s:.0f} s (last status {status})")
        sleep(interval_s)


def server_props(base_url: str, get: Callable = http_get_json) -> dict:
    status, props = get(f"{base_url}/props")
    if status != 200:
        raise ServerError(f"{base_url}/props returned {status}")
    return props


def slot_context(props: dict) -> int | None:
    gen = props.get("default_generation_settings") or {}
    n_ctx = gen.get("n_ctx") if isinstance(gen, dict) else None
    return int(n_ctx) if n_ctx is not None else None


def props_summary(props: dict) -> dict:
    return {
        "build_info": props.get("build_info"),
        "total_slots": props.get("total_slots"),
        "n_ctx_per_slot": slot_context(props),
        "model_path": props.get("model_path"),
    }


def up(model_key: str, cfg: ModelsConfig, run: Callable = _run, get: Callable = http_get_json,
       sleep: Callable[[float], None] = time.sleep, reader: Callable[[], int] | None = None) -> dict:
    idle = read_idle_mib()
    if idle is not None:
        wait_gpu_idle(idle, reader=reader or (lambda: gpu_memory_used_mib(run)), sleep=sleep)
    compose_up(model_key, cfg, run)
    waited = wait_health(cfg.server.base_url, cfg.server.health_timeout_s, get=get, sleep=sleep)
    summary = props_summary(server_props(cfg.server.base_url, get))
    summary["health_wait_s"] = round(waited, 1)
    per_slot = summary["n_ctx_per_slot"]
    if per_slot is not None and per_slot < cfg.server.min_ctx_per_slot:
        raise ServerError(f"per-slot context {per_slot} is below {cfg.server.min_ctx_per_slot} tokens")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Start or stop the aib llama.cpp server.")
    sub = parser.add_subparsers(dest="action", required=True)
    p_up = sub.add_parser("up")
    p_up.add_argument("model")
    sub.add_parser("down")
    args = parser.parse_args(argv)
    cfg = load_models()
    try:
        if args.action == "up":
            summary = up(args.model, cfg)
            print(json.dumps({"model": args.model, **summary}), flush=True)
        else:
            saved = compose_down(cfg)
            print(f"stopped aib/{cfg.server.service}" + (f"; log saved to {saved}" if saved else ""), flush=True)
    except (ServerError, subprocess.CalledProcessError) as exc:
        detail = getattr(exc, "stderr", "") or ""
        print(f"serve {args.action} failed: {exc} {detail}".strip(), file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
