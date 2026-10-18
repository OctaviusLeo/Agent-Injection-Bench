"""Unit tests for aib.repro (Req 0.1, 0.2)."""

from __future__ import annotations

import json
import re
import subprocess

import pytest

from aib import repro
from aib.repro import ReproError, Stamp, commit_hash, make_stamp, require_clean

_HEX40 = re.compile(r"\A[0-9a-f]{40}\Z")


def _git(cwd, *args):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


@pytest.fixture
def repo(tmp_path):
    _git(tmp_path, "init", "-q", "-b", "main")
    _git(tmp_path, "config", "user.email", "test@example.com")
    _git(tmp_path, "config", "user.name", "Test")
    (tmp_path / "aib").mkdir()
    (tmp_path / "aib" / "m.py").write_text("x = 1\n")
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "r.json").write_text("{}\n")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "init")
    return tmp_path


def test_commit_hash_of_this_repository_is_40_hex():
    assert _HEX40.match(commit_hash())


def test_commit_hash_raises_without_a_commit(tmp_path):
    _git(tmp_path, "init", "-q")
    with pytest.raises(ReproError):
        commit_hash(tmp_path)


def test_commit_hash_raises_on_non_hex_output(monkeypatch):
    monkeypatch.setattr(repro, "_git", lambda args, cwd: "0" * 39 + "\n")
    with pytest.raises(ReproError, match="40-character"):
        commit_hash()


def test_missing_git_raises_repro_error(monkeypatch):
    def boom(*a, **k):
        raise FileNotFoundError("git")

    monkeypatch.setattr(repro.subprocess, "run", boom)
    with pytest.raises(ReproError, match="not installed"):
        commit_hash()


def test_require_clean_accepts_results_change(repo):
    (repo / "results" / "r.json").write_text('{"a": 1}\n')
    (repo / "results" / "new.json").write_text("{}\n")
    assert require_clean(cwd=repo) == commit_hash(repo)


def test_require_clean_refuses_code_change_and_lists_it(repo):
    (repo / "aib" / "m.py").write_text("x = 2\n")
    (repo / "aib" / "extra.py").write_text("y = 1\n")
    with pytest.raises(ReproError) as info:
        require_clean(cwd=repo)
    assert "aib/m.py" in str(info.value)
    assert "aib/extra.py" in str(info.value)


def test_require_clean_checks_both_sides_of_a_rename(repo):
    _git(repo, "mv", "results/r.json", "aib/r.json")
    with pytest.raises(ReproError, match="aib/r.json"):
        require_clean(cwd=repo)


def test_stamp_round_trips_through_json(repo):
    stamp = make_stamp("python -m aib.metrics --seed 7", 7, cwd=repo)
    restored = Stamp.from_dict(json.loads(json.dumps(stamp.to_dict())))
    assert restored == stamp
    assert _HEX40.match(restored.commit)
    assert restored.seed == 7 and isinstance(restored.seed, int)
