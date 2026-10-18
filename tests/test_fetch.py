"""Unit tests for aib.fetch with a fake downloader (no network)."""

from __future__ import annotations

import hashlib
import json
import subprocess

import pytest

from aib import fetch as F
from aib.config import load_models


def _fake_download(contents):
    def download(repo, filename, revision, local_dir):
        path = local_dir / filename
        path.write_bytes(contents[filename])
        return str(path)

    return download


def test_plan_covers_models_and_classifier(tmp_path):
    cfg = load_models()
    items = F.plan(cfg, F.DEFAULT_KEYS, tmp_path)
    names = [i.file.name for i in items]
    assert "Qwen3-4B-Instruct-2507-Q4_K_M.gguf" in names and "model.safetensors" in names
    assert "training_args.bin" not in names and not any(n.startswith("onnx") for n in names)
    assert {i.dest_dir.name for i in items} == {"qwen3-4b", "llama3.1-8b", "classifier"}
    with pytest.raises(KeyError):
        F.plan(cfg, ("nope",), tmp_path)
    with pytest.raises(ValueError):
        F.plan(cfg, ("gpt-6-luna",), tmp_path)


def test_fetch_item_checks_size_and_sha(tmp_path):
    cfg = load_models()
    item = F.plan(cfg, ("classifier",), tmp_path)[0]  # config.json, 994 bytes, no sha in config
    good = F.fetch_item(item, _fake_download({"config.json": b"x" * 994}))
    assert good["size_ok"] and good["sha256_ok"] and good["sha256"] == hashlib.sha256(b"x" * 994).hexdigest()
    bad = F.fetch_item(item, _fake_download({"config.json": b"x" * 10}))
    assert not bad["size_ok"]
    gguf = F.plan(cfg, ("qwen3-4b",), tmp_path)[0]
    wrong = F.FetchItem(gguf.key, gguf.repo, gguf.revision, gguf.file.model_copy(update={"size": 3}), gguf.dest_dir)
    res = F.fetch_item(wrong, _fake_download({gguf.file.name: b"abc"}))
    assert res["size_ok"] and not res["sha256_ok"]


def _repo(tmp_path):
    root = tmp_path / "repo"
    (root / "config").mkdir(parents=True)
    (root / "config" / "models.yaml").write_text((F.REPO_ROOT / "config" / "models.yaml").read_text())
    for cmd in (["init", "-q", "-b", "main"], ["config", "user.email", "t@e.com"], ["config", "user.name", "T"],
                ["add", "-A"], ["commit", "-q", "-m", "init"]):
        subprocess.run(["git", *cmd], cwd=root, check=True, capture_output=True)
    return root


def test_main_writes_stamped_results_and_merges(tmp_path, monkeypatch):
    root = _repo(tmp_path)
    monkeypatch.setenv("AIB_MODELS", str(tmp_path / "models"))
    cfg = load_models()
    # large files are deliberately one byte long, so a mismatch is reported
    contents = {f.name: b"x" * (f.size if f.size < 10_000 else 1) for f in cfg.classifier.files}
    code = F.main(["--model", "classifier"], download=_fake_download(contents), root=root)
    assert code == 1
    data = json.loads((root / "results" / "m0_models.json").read_text())
    assert len(data["stamp"]["commit"]) == 40 and data["stamp"]["command"].endswith("--model classifier")
    first = cfg.classifier.files[0]
    monkeypatch.setattr(F, "plan", lambda c, k, b: [F.FetchItem("extra", "r", "0" * 40, first, b / "extra")])
    assert F.main(["--model", "classifier"], download=_fake_download({first.name: b"x" * first.size}), root=root) == 0
    merged = json.loads((root / "results" / "m0_models.json").read_text())["entries"]
    assert set(merged) == {"classifier", "extra"}
