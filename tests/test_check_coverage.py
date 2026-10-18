"""Unit tests for scripts/check_coverage.py."""

from __future__ import annotations

import json

from scripts import check_coverage as cc


def _write(root, rel, text):
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_counts_public_functions_and_methods():
    text = "def a():\n    pass\n\ndef _b():\n    pass\n\nclass C:\n    def m(self):\n        pass\n    def _n(self):\n        pass\n\nclass _D:\n    def x(self):\n        pass\n"
    assert cc.count_public_functions(text) == 2


def test_source_lines_skip_blank_and_comment_lines():
    assert cc.count_source_lines("# c\n\nx = 1\n  # d\ny = 2\n") == 2


def _report(files):
    return {"files": {k: {"summary": {"percent_covered": v}} for k, v in files.items()}}


def test_flags_non_trivial_module_under_minimum(tmp_path):
    _write(tmp_path, "aib/big.py", "def a():\n    pass\n\ndef b():\n    pass\n")
    _write(tmp_path, "aib/small.py", "def only():\n    pass\n")
    _write(tmp_path, "aib/__init__.py", "")
    results = cc.assess(_report({"aib/big.py": 50.0, "aib/small.py": 10.0}), tmp_path)
    failed = cc.failures(results)
    assert [r.path for r in failed] == ["aib/big.py"]


def test_long_module_is_non_trivial_and_missing_counts_as_zero(tmp_path):
    _write(tmp_path, "scripts/long.py", "".join(f"x{i} = {i}\n" for i in range(60)))
    results = cc.assess(_report({}), tmp_path)
    assert results[0].non_trivial and results[0].percent == 0.0
    assert [r.path for r in cc.failures(results)] == ["scripts/long.py"]


def test_absolute_paths_are_mapped_to_repository_paths(tmp_path):
    _write(tmp_path, "aib/m.py", "def a():\n    pass\n\ndef b():\n    pass\n")
    report = _report({str((tmp_path / "aib/m.py").resolve()): 90.0, "/elsewhere/x.py": 1.0})
    assert cc.load_percentages(report, tmp_path) == {"aib/m.py": 90.0}
    assert cc.failures(cc.assess(report, tmp_path)) == []


def test_main_exit_codes(tmp_path, capsys):
    _write(tmp_path, "aib/m.py", "def a():\n    pass\n\ndef b():\n    pass\n")
    good = tmp_path / "good.json"
    good.write_text(json.dumps(_report({"aib/m.py": 85.0})), encoding="utf-8")
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(_report({"aib/m.py": 79.9})), encoding="utf-8")
    assert cc.main([str(good), "--root", str(tmp_path)]) == 0
    assert cc.main([str(bad), "--root", str(tmp_path)]) == 1
    assert "FAIL aib/m.py: 79.9%" in capsys.readouterr().out
