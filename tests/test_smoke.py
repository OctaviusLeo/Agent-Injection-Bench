"""Unit tests for aib.smoke with fakes for nvidia-smi, Docker, the server script, and HTTP."""

from __future__ import annotations

import io
import json
import shutil
import subprocess
import urllib.error
from types import SimpleNamespace

import pytest

from aib import smoke as SM
from aib.llm import ChatLLM
from tests.fake_openai import FakeOpenAI, completion


def _repo(tmp_path):
    root = tmp_path / "repo"
    shutil.copytree(SM.REPO_ROOT / "config", root / "config")
    for cmd in (["init", "-q", "-b", "main"], ["config", "user.email", "t@e.com"], ["config", "user.name", "T"],
                ["add", "-A"], ["commit", "-q", "-m", "init"]):
        subprocess.run(["git", *cmd], cwd=root, check=True, capture_output=True)
    return root


class FakeRun:
    def __init__(self, fail=None, used="3100", total="10240", serve_out=None):
        self.fail = fail or set()
        self.used, self.total = used, total
        self.serve_out = serve_out if serve_out is not None else json.dumps(
            {"model": "qwen3-4b", "build_info": "b11580-4413a19", "n_ctx_per_slot": 32768, "total_slots": 2})
        self.calls = []

    def __call__(self, cmd, timeout=300):
        self.calls.append(cmd)
        key = " ".join(cmd)
        ok = lambda out="": SimpleNamespace(returncode=0, stdout=out, stderr="")  # noqa: E731
        bad = lambda err: SimpleNamespace(returncode=1, stdout="", stderr=err)  # noqa: E731
        if cmd[:2] == ["docker", "version"]:
            return bad("Cannot connect to the Docker daemon") if "docker" in self.fail else ok("29.9.1\n")
        if cmd[:2] == ["docker", "run"]:
            return bad("could not select device driver") if "gpu" in self.fail else ok("GPU table\n")
        if key == "nvidia-smi":
            return ok("NVIDIA-SMI 617.42 | RTX 3080 | 10240MiB\n")
        if "memory.total" in key:
            return ok(f"{self.total}\n")
        if "memory.used" in key:
            return ok(f"{self.used}\n")
        if cmd[:2] == ["bash", "scripts/serve.sh"]:
            return bad("compose up failed") if "serve" in self.fail else ok("pulling\n" + self.serve_out + "\n")
        raise AssertionError(f"unexpected command {cmd}")


class FakeSampler:
    def __init__(self, samples):
        self.samples = samples

    def start(self):
        pass

    def stop(self):
        return list(self.samples)


def _post_ok(url, payload, timeout):
    assert payload["messages"][0]["content"] == SM.SMOKE_PROMPT and timeout == 120
    assert payload["temperature"] == 0.7 and payload["top_k"] == 0
    return {"choices": [{"message": {"content": "A unit test checks one small piece of code."}}],
            "timings": {"predicted_per_second": 101.5, "prompt_per_second": 900.0}, "usage": {"completion_tokens": 12}}


def test_smoke_passes_and_records_everything(tmp_path):
    root = _repo(tmp_path)
    code, res = SM.run_smoke("qwen3-4b", root=root, run=FakeRun(), post=_post_ok,
                             sampler=FakeSampler([3100, 7400, 7350]), version=(3, 12, 3))
    assert code == 0 and res["passed"]
    assert res["total_mib"] == 10240 and res["baseline_mib"] == 3100 and res["peak_mib"] == 7400
    assert res["predicted_per_second"] == 101.5 and res["n_ctx_per_slot"] == 32768
    assert res["python"] == "3.12.3" and res["docker_ok"] and res["docker_gpu_ok"]
    saved = json.loads((root / "results" / "m0_smoke_qwen3-4b.json").read_text())
    assert len(saved["stamp"]["commit"]) == 40 and saved["response"].startswith("A unit test")
    assert (root / "docs" / "evidence" / "nvidia_smi_qwen3-4b.txt").read_text().startswith("NVIDIA-SMI")


@pytest.mark.parametrize(
    "kwargs,condition",
    [
        (dict(run=FakeRun(fail={"docker"})), "docker available"),
        (dict(run=FakeRun(fail={"gpu"})), "docker GPU access"),
        (dict(run=FakeRun(fail={"serve"})), "model server start"),
        (dict(run=FakeRun(serve_out="not json")), "model server start"),
        (dict(version=(3, 10, 12)), "Python >= 3.11"),
        (dict(sampler=FakeSampler([12000])), "peak GPU memory <= total"),
        (dict(post=lambda u, p, t: {"choices": [{"message": {"content": "  "}}]}), "non-empty response"),
        (dict(post=lambda u, p, t: (_ for _ in ()).throw(TimeoutError("timed out"))), "valid response within 120 s"),
        (dict(post=lambda u, p, t: (_ for _ in ()).throw(urllib.error.URLError("refused"))), "valid response within 120 s"),
    ],
)
def test_smoke_failures_record_condition_and_exit_nonzero(tmp_path, kwargs, condition):
    root = _repo(tmp_path)
    args = dict(run=FakeRun(), post=_post_ok, sampler=FakeSampler([3100]), version=(3, 12, 3)) | kwargs
    code, res = SM.run_smoke("qwen3-4b", root=root, **args)
    assert code == 1 and res["passed"] is False
    assert res["failed_condition"].startswith(condition) and res["cause"]
    assert json.loads((root / "results" / "m0_smoke_qwen3-4b.json").read_text())["failed_condition"] == res["failed_condition"]


def test_send_prompt_rejects_slow_response():
    t = iter([0.0, 130.0])
    with pytest.raises(SM.SmokeFailure, match="130.0 s"):
        SM.send_prompt("http://x", "m", {}, post=lambda u, p, tm: {"choices": [{"message": {"content": "hi"}}]},
                       clock=lambda: next(t))


def test_memory_sampler_reads_streamed_lines():
    class Proc:
        def __init__(self, *a, **k):
            self.stdout = io.StringIO("3000\n3200\nbad\n3100\n")

        def terminate(self):
            pass

        def wait(self, timeout=None):
            return 0

    s = SM.MemorySampler(popen=Proc)
    s.start()
    assert s.stop() == [3000, 3200, 3100]


def test_http_post_json_round_trip():
    with FakeOpenAI([(200, completion(content="ok"), 0)]) as srv:
        body = SM.http_post_json(srv.base_url + "/chat/completions", {"model": "m", "messages": []}, 5)
    assert body["choices"][0]["message"]["content"] == "ok"


def _factory(srv):
    def make(name, base_url, model, key_env, sampling, timeout):
        return ChatLLM(name, srv.base_url, model, key_env, sampling, timeout, max_attempts=1, backoff_s=0)
    return make


def test_probe_passes_on_structured_tool_call(tmp_path):
    root = _repo(tmp_path)
    with FakeOpenAI([(200, completion(tool_calls=[("c1", "get_current_day", "{}")]), 0)]) as srv:
        code, res = SM.run_probe("qwen3-4b", root=root, llm_factory=_factory(srv))
        sent = srv.requests[0]
    assert code == 0 and res["passed"] and res["tool_calls"][0]["function"] == "get_current_day"
    assert [t["function"]["name"] for t in sent["tools"]] == ["get_current_day"]
    assert sent["messages"] == [{"role": "user", "content": SM.PROBE_PROMPT}]
    assert json.loads((root / "results" / "m0_tool_probe_qwen3-4b.json").read_text())["passed"] is True


@pytest.mark.parametrize(
    "reply,cause",
    [
        ((200, completion(content="It is Monday."), 0), "no such tool call"),
        ((200, completion(tool_calls=[("c1", "get_current_day", "{bad")]), 0), "did not parse"),
        ((500, {"error": {"message": "down"}}, 0), "server"),
    ],
)
def test_probe_failures(tmp_path, reply, cause):
    root = _repo(tmp_path)
    with FakeOpenAI([reply]) as srv:
        code, res = SM.run_probe("llama3.1-8b", root=root, attempt=2, llm_factory=_factory(srv))
    assert code == 1 and not res["passed"] and cause in res["cause"]
    assert res["stamp"]["seed"] == 2


def test_cli_entry_points(monkeypatch, capsys):
    monkeypatch.setattr(SM, "require_clean", lambda: "0" * 40)
    monkeypatch.setattr(SM, "run_smoke", lambda key: (0, {"model": key, "passed": True}))
    monkeypatch.setattr(SM, "run_probe", lambda key, attempt: (1, {"model": key, "attempt": attempt, "passed": False}))
    assert SM.main_smoke(["--model", "qwen3-4b"]) == 0
    assert SM.main_probe(["--model", "qwen3-4b", "--attempt", "3"]) == 1
    assert '"attempt": 3' in capsys.readouterr().out


def test_check_python_threshold():
    assert SM.check_python((3, 11, 0))["python_ok"] and not SM.check_python((3, 10, 9))["python_ok"]
