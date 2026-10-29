"""Unit tests for aib.experiments with StubLLM and fakes."""

from __future__ import annotations

import json
import shutil
import subprocess

import pytest

from aib import config as CFGMOD
from aib import experiments as E
from aib.classifier import KeywordClassifier
from aib.config import load_all, load_experiment
from aib.ledger import SpendCapExceeded
from aib.llm import LLMConnectionError
from aib.runner import RunContext
from aib.stub_model import StubLLM

EXP = load_experiment()
CTX = RunContext(sampling={"temperature": 0.7}, stamp={"command": "c", "seed": 1, "commit": "a" * 40})


def test_grid_sizes_and_orders():
    local = E.build_grid("qwen3-4b", EXP)
    assert len(local) == 2880 and len({s.case_id for s in local}) == 2880
    assert [(s.defense, s.condition, s.user_task_id, s.rep) for s in local[:4]] == [
        ("none", "clean", "user_task_0", 0), ("none", "clean", "user_task_0", 1), ("none", "clean", "user_task_0", 2),
        ("none", "clean", "user_task_1", 0)]
    assert local[-1].defense == "filter" and local[-1].condition == "hidden_field" and local[-1].rep == 2
    hosted = E.build_grid("gpt-6-luna", EXP, hosted=True, hosted_conditions=["clean", "direct", "important_instructions"])
    assert len(hosted) == 1440
    assert all(s.rep == 0 for s in hosted[:480]) and all(s.rep == 2 for s in hosted[960:])
    assert [(s.defense, s.condition, s.user_task_id) for s in hosted[:2]] == [("none", "clean", "user_task_0"),
                                                                              ("none", "clean", "user_task_1")]
    pilot = E.build_grid("qwen3-4b", EXP, pilot=True)
    assert len(pilot) == 48 and {s.user_task_id for s in pilot} == {"user_task_0", "user_task_1"}
    assert {s.rep for s in pilot} == {0}
    assert all((s.injection_task_id is None) == (s.condition == "clean") for s in local)


def test_seeds_are_stable_and_distinct():
    a = E.build_grid("qwen3-4b", EXP)
    b = E.build_grid("qwen3-4b", EXP)
    assert [s.seed for s in a] == [s.seed for s in b]
    assert len({s.seed for s in a}) == len(a)
    assert E.case_seed(20261017, "k") == E.case_seed(20261017, "k") != E.case_seed(20261018, "k")
    assert 0 <= E.case_seed(1, "x") < 2**31


def _small(n=6):
    return [s for s in E.build_grid("stub", EXP, pilot=True) if s.defense in ("none", "filter")][:n]


def _store(tmp_path, name="s.jsonl"):
    store = E.ResultsStore(tmp_path / name)
    store.open(CTX.stamp, {})
    return store


def test_one_record_per_case_and_resume(tmp_path):
    store = E.ResultsStore(tmp_path / "s.jsonl")
    assert store.open(CTX.stamp, {"x": 1}) is False
    specs = _small()
    s1 = E.run_grid("stub", specs, 2, store, lambda spec: StubLLM("follow_user"), CTX, KeywordClassifier)
    assert s1.recorded == 6 and not s1.stopped and len(store.records()) == 6
    assert store.completed_ids() == {s.case_id for s in specs}
    assert store.open(CTX.stamp, {"x": 1}) is True
    s2 = E.run_grid("stub", specs, 2, store, lambda spec: StubLLM("follow_user"), CTX, KeywordClassifier)
    assert s2.skipped == 6 and s2.recorded == 0 and len(store.records()) == 6
    with pytest.raises(E.StoreError, match="started on commit"):
        store.open({**CTX.stamp, "commit": "b" * 40}, {})


def test_store_without_header_is_refused_and_torn_lines_ignored(tmp_path):
    p = tmp_path / "s.jsonl"
    p.write_text(json.dumps({"case_id": "x"}) + "\n")
    with pytest.raises(E.StoreError, match="no header"):
        E.ResultsStore(p).open(CTX.stamp, {})
    store = _store(tmp_path, "t.jsonl")
    with (tmp_path / "t.jsonl").open("a") as f:
        f.write(json.dumps({"case_id": "half"})[:9])
    assert store.records() == [] and store.header()["header"] is True


class Down:
    name = "down"

    def query(self, *a, **k):
        raise LLMConnectionError("connection failed: refused")


class Crash:
    name = "crash"

    def query(self, *a, **k):
        raise ValueError("unparseable server response")


class Broke:
    name = "broke"

    def query(self, *a, **k):
        raise SpendCapExceeded(2.99, 0.0, 0.02, 3.0)


def test_infrastructure_stop_records_nothing(tmp_path):
    store = _store(tmp_path)
    specs = E.build_grid("stub", EXP, pilot=True)[:12]
    summary = E.run_grid("stub", specs, 1, store, lambda spec: Down(), CTX)
    assert summary.stopped and "5 consecutive" in summary.stop_reason
    assert summary.infra_failures == 5 and store.records() == []


def test_isolated_infra_failure_is_not_recorded_and_flags_the_run(tmp_path):
    store = _store(tmp_path)
    specs = _small(4)
    summary = E.run_grid("stub", specs, 1, store, lambda spec: Down() if spec is specs[1] else StubLLM("follow_user"),
                         CTX, KeywordClassifier)
    assert summary.recorded == 3 and summary.infra_failures == 1 and summary.stopped
    assert specs[1].case_id not in store.completed_ids()


def test_per_case_error_is_recorded_and_sweep_continues(tmp_path):
    store = _store(tmp_path)
    specs = _small(4)
    summary = E.run_grid("stub", specs, 2, store, lambda spec: Crash() if spec is specs[0] else StubLLM("follow_user"),
                         CTX, KeywordClassifier)
    assert summary.recorded == 4 and summary.errored == 1 and not summary.stopped
    errored = [r for r in store.records() if r["status"] == "errored"]
    assert errored[0]["case_id"] == specs[0].case_id and "unparseable" in errored[0]["error_cause"]


def test_spend_cap_refusal_stops_the_run(tmp_path):
    store = _store(tmp_path)
    summary = E.run_grid("stub", _small(4), 1, store, lambda spec: Broke(), CTX)
    assert summary.stopped and summary.stopped_at_cap and store.records() == []


def test_same_configuration_and_seed_reproduce_ids_seeds_and_configuration(tmp_path):
    specs = _small()
    keep = ("case_id", "seed", "model", "defense", "condition", "user_task_id", "injection_task_id", "rep", "sampling",
            "stamp", "status", "task_success", "injection_success")
    rows = []
    for name in ("a.jsonl", "b.jsonl"):
        store = _store(tmp_path, name)
        E.run_grid("stub", specs, 2, store, lambda spec: StubLLM("follow_user"), CTX, KeywordClassifier)
        rows.append(sorted(({k: r[k] for k in keep} for r in store.records()), key=lambda r: r["case_id"]))
    assert rows[0] == rows[1]


def _repo(tmp_path):
    root = tmp_path / "repo"
    shutil.copytree(E.REPO_ROOT / "config", root / "config")
    for cmd in (["init", "-q", "-b", "main"], ["config", "user.email", "t@e.com"], ["config", "user.name", "T"],
                ["add", "-A"], ["commit", "-q", "-m", "init"]):
        subprocess.run(["git", *cmd], cwd=root, check=True, capture_output=True)
    return root


def test_run_cli_local_and_hosted_with_stubs(tmp_path, monkeypatch, capsys):
    import aib.classifier
    import aib.hosted
    import aib.llm

    root = _repo(tmp_path)
    monkeypatch.setenv("AIB_RUNS", str(tmp_path / "runs"))
    monkeypatch.setattr(aib.llm, "ChatLLM", lambda *a, **k: StubLLM("follow_user"))
    monkeypatch.setattr(aib.hosted, "make_hosted_llm", lambda *a, **k: StubLLM("follow_injection"))
    monkeypatch.setattr(aib.classifier, "InjectionClassifier", lambda **k: KeywordClassifier())
    real = E.build_grid
    monkeypatch.setattr(E, "build_grid", lambda *a, **k: real(*a, **k)[:3])
    assert E.run_cli("qwen3-4b", pilot=True, workers=1, root=root) == 0
    header = E.ResultsStore(root / "results/pilot/qwen3-4b.jsonl").header()
    assert header["config"]["pilot"] is True and header["config"]["sampling"]["top_k"] == 0
    assert E.run_cli("gpt-6-luna", pilot=True, workers=1, root=root) == 0
    assert not (root / "results" / "hosted_run.json").exists()  # only the full hosted run writes its summary
    assert E.run_cli("gpt-6-luna", pilot=False, workers=1, root=root) == 0
    summary = json.loads((root / "results" / "hosted_run.json").read_text())
    assert summary["completed_cases"] == 3 and summary["stopped_at_cap"] is False and summary["spend_cap_usd"] == 3.0
    assert E.run_cli("llama3.1-8b", pilot=False, workers=1, root=root) == 2  # disabled: replaced by qwen2.5-7b
    assert "starting 3 cases" in capsys.readouterr().out


def test_run_cli_claude_uses_its_grid_ledger_and_summary(tmp_path, monkeypatch):
    import aib.claude
    import aib.classifier

    root = _repo(tmp_path)
    monkeypatch.setenv("AIB_RUNS", str(tmp_path / "runs"))
    built = {}

    def fake_llm(cfg, model, ledger, purpose="", client=None):
        built.update(model=model, run_id=ledger.run_id, cap=ledger.cap_usd, purpose=purpose)
        return StubLLM("follow_injection")

    monkeypatch.setattr(aib.claude, "make_claude_llm", fake_llm)
    monkeypatch.setattr(aib.classifier, "InjectionClassifier", lambda **k: KeywordClassifier())
    real = E.build_grid
    monkeypatch.setattr(E, "build_grid", lambda *a, **k: real(*a, **k)[:3])
    assert E.run_cli("claude-haiku-5-5", pilot=False, workers=1, root=root) == 0
    assert built["cap"] == 5.0 and built["purpose"] == "benchmark:claude-haiku-5-5"
    summary = json.loads((root / "results" / "claude_run.json").read_text())
    assert summary["completed_cases"] == 3 and summary["cap_scope"] == "run" and summary["run_id"] == built["run_id"]
    assert summary["spend_cap_usd"] == 5.0 and not (root / "results" / "hosted_run.json").exists()
    header = E.ResultsStore(root / "results/raw/claude-haiku-5-5.jsonl").header()
    assert header["config"]["sampling"]["temperature"] is None and header["config"]["claude"]["run_cap_usd"] == 5.0
    assert "hosted" not in header["config"]


def test_grid_for_uses_each_providers_conditions():
    cfg = load_all()
    claude = E.grid_for(cfg, "claude-haiku-5-5")
    assert len(claude) == 3 * 4 * 40 * 3 and {s.condition for s in claude} == {"clean", "direct", "important_instructions"}
    assert claude[0].rep == 0 and claude[-1].rep == 2  # Repetition-major, like the OpenAI hosted grid
    assert len(E.grid_for(cfg, "gpt-6-luna")) == len(claude)
    assert len(E.grid_for(cfg, "qwen3-4b")) == 4 * 6 * 40 * 3
    assert E.model_version(cfg, "claude-haiku-5-5").startswith("claude-haiku-5-5")


def test_main_dispatches(monkeypatch):
    monkeypatch.setattr(E, "run_cli", lambda model, pilot, workers: 7 if pilot and workers == 3 else 0)
    assert E.main(["run", "--model", "qwen3-4b", "--pilot", "--workers", "3"]) == 7


def test_model_version_strings_and_provider_cli(capsys):
    cfg = load_all()
    assert E.model_version(cfg, "qwen3-4b").startswith("unsloth/Qwen3-4B-Instruct-2507-GGUF@a06e946")
    assert E.model_version(cfg, "gpt-6-luna").startswith("gpt-6-luna")
    assert CFGMOD.main(["provider", "qwen3-4b"]) == 0 and CFGMOD.main(["provider", "gpt-6-luna"]) == 0
    assert capsys.readouterr().out.split() == ["llamacpp", "openai"]
    assert CFGMOD.main(["provider", "nope"]) == 2 and CFGMOD.main([]) == 2
