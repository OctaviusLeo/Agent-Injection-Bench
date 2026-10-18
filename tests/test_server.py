"""Unit tests for aib.server with fakes for docker, nvidia-smi, and HTTP."""

from __future__ import annotations

import subprocess
from types import SimpleNamespace

import pytest

from aib import server as S
from aib.config import load_models
from tests.fake_openai import FakeOpenAI


class FakeRun:
    def __init__(self, nvidia=("3000",), logs="log line\n"):
        self.calls = []
        self.nvidia = list(nvidia)
        self.logs = logs

    def __call__(self, cmd, env=None, check=True):
        self.calls.append((cmd, env))
        if cmd[0] == "nvidia-smi":
            value = self.nvidia.pop(0) if len(self.nvidia) > 1 else self.nvidia[0]
            return SimpleNamespace(stdout=f"{value}\n", returncode=0)
        if "logs" in cmd:
            return SimpleNamespace(stdout=self.logs, returncode=0)
        return SimpleNamespace(stdout="", returncode=0)


def test_gpu_readings_parse():
    run = FakeRun(nvidia=("2854",))
    assert S.gpu_memory_used_mib(run) == 2854
    assert S.gpu_memory_total_mib(run) == 2854


def test_idle_rule_waits_then_raises():
    readings = iter([5000, 5000, 5000, 5000, 5000, 5000])
    sleeps = []
    with pytest.raises(S.GpuBusyError, match="idle reading 2800"):
        S.wait_gpu_idle(2800, reader=lambda: next(readings), sleep=sleeps.append)
    assert sleeps == [300, 300, 300, 300]


def test_idle_rule_returns_once_memory_drops():
    readings = iter([5000, 3500])
    sleeps = []
    assert S.wait_gpu_idle(2800, reader=lambda: next(readings), sleep=sleeps.append) == 3500
    assert sleeps == [300]


def test_compose_env_passes_model_settings():
    cfg = load_models()
    env = S.compose_env("qwen3-4b", cfg, base_env={"AIB_MODELS": "/m"})
    assert env["AIB_MODEL_PATH"] == "qwen3-4b/Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
    assert env["AIB_CTX_SIZE"] == "65536" and env["AIB_PARALLEL"] == "2"
    assert "--jinja" in env["AIB_SERVER_FLAGS"] and "--chat-template-file" not in env["AIB_SERVER_FLAGS"]
    llama = cfg.models["llama3.1-8b"].model_copy(update={"chat_template_file": "templates/llama.jinja"})
    cfg2 = cfg.model_copy(update={"models": {**cfg.models, "llama3.1-8b": llama}})
    env2 = S.compose_env("llama3.1-8b", cfg2, base_env={})
    assert env2["AIB_PARALLEL"] == "1" and env2["AIB_MODELS"].endswith("aib-models")
    assert env2["AIB_SERVER_FLAGS"].endswith("--chat-template-file /models/templates/llama.jinja")
    with pytest.raises(S.ServerError):
        S.compose_env("gpt-6-luna", cfg)


def test_up_passes_settings_and_checks_props(monkeypatch, tmp_path):
    cfg = load_models()
    monkeypatch.setattr(S, "IDLE_FILE", tmp_path / "idle.txt")
    (tmp_path / "idle.txt").write_text("2800\n")
    props = {"default_generation_settings": {"n_ctx": 32768}, "total_slots": 2, "build_info": "b11580-abc",
             "model_path": "/models/x.gguf"}
    with FakeOpenAI([], routes={"/health": (200, {"status": "ok"}), "/props": (200, props)}) as srv:
        cfg = cfg.model_copy(update={"server": cfg.server.model_copy(update={"base_url": srv.root_url})})
        run = FakeRun(nvidia=("2900",))
        summary = S.up("qwen3-4b", cfg, run=run, sleep=lambda s: None)
    assert summary["n_ctx_per_slot"] == 32768 and summary["build_info"] == "b11580-abc"
    up_cmd, env = [c for c in run.calls if "up" in c[0]][0]
    assert up_cmd[:4] == ["docker", "compose", "-p", "aib"] and up_cmd[-1] == "llm"
    assert env["AIB_CTX_SIZE"] == "65536"


def test_up_rejects_small_slot_context(monkeypatch, tmp_path):
    cfg = load_models()
    monkeypatch.setattr(S, "IDLE_FILE", tmp_path / "missing.txt")
    with FakeOpenAI([], routes={"/health": (200, {}), "/props": (200, {"default_generation_settings": {"n_ctx": 4096}})}) as srv:
        cfg = cfg.model_copy(update={"server": cfg.server.model_copy(update={"base_url": srv.root_url})})
        with pytest.raises(S.ServerError, match="below 32768"):
            S.up("qwen3-4b", cfg, run=FakeRun(), sleep=lambda s: None)


def test_health_polling_times_out_cleanly():
    t = iter(range(0, 10_000, 100))
    with pytest.raises(S.ServerTimeoutError, match="within 900 s"):
        S.wait_health("http://x", 900, get=lambda url: (503, {}), sleep=lambda s: None, clock=lambda: next(t))
    calls = []
    waited = S.wait_health("http://x", 900, get=lambda url: calls.append(url) or (200, {}), sleep=lambda s: None)
    assert calls == ["http://x/health"] and waited >= 0


def test_http_get_json_handles_errors():
    with FakeOpenAI([], routes={"/ok": (200, {"a": 1}), "/bad": (500, {})}) as srv:
        assert S.http_get_json(srv.root_url + "/ok") == (200, {"a": 1})
        assert S.http_get_json(srv.root_url + "/bad")[0] == 500
    assert S.http_get_json("http://127.0.0.1:9/none", timeout=0.5) == (0, {})
    with pytest.raises(S.ServerError):
        S.server_props("http://x", get=lambda url: (0, {}))


def test_down_saves_log_and_stops(tmp_path):
    run = FakeRun(logs="llama log\n")
    saved = S.compose_down(load_models(), run=run, log_dir=tmp_path)
    assert saved.read_text() == "llama log\n"
    assert run.calls[-1][0][-1] == "down"
    assert S.compose_down(load_models(), run=FakeRun(logs=""), log_dir=tmp_path) is None


def test_read_idle(tmp_path):
    p = tmp_path / "i.txt"
    assert S.read_idle_mib(p) is None
    p.write_text("2614\n")
    assert S.read_idle_mib(p) == 2614


def test_main_reports_failures(monkeypatch, capsys):
    monkeypatch.setattr(S, "compose_down", lambda cfg: (_ for _ in ()).throw(subprocess.CalledProcessError(1, "docker", stderr="no docker")))
    assert S.main(["down"]) == 1
    assert "no docker" in capsys.readouterr().err
    monkeypatch.setattr(S, "up", lambda key, cfg: {"build_info": "b"})
    assert S.main(["up", "qwen3-4b"]) == 0
    monkeypatch.setattr(S, "compose_down", lambda cfg: None)
    assert S.main(["down"]) == 0


def test_compose_port_matches_config():
    cfg = load_models()
    compose = S.COMPOSE_FILE.read_text()
    assert f'"127.0.0.1:{cfg.server.host_port}:8080"' in compose
    assert cfg.server.base_url == f"http://127.0.0.1:{cfg.server.host_port}"
