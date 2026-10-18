"""Unit tests for scripts/check_traceability.py (the traceability CLI; Req 9.3, 9.4).

Covers command-stripping, the exit codes (0 when clean, non-zero on an
untraceable number), and that the committed README skeleton passes the check.
"""

from __future__ import annotations

import importlib.util
import os
import pathlib
import sys

# scripts/ is not a package; load the CLI by path (mirrors test_smoke_test.py).
_ROOT = pathlib.Path(__file__).resolve().parent.parent
_PATH = _ROOT / "scripts" / "check_traceability.py"
_spec = importlib.util.spec_from_file_location("aib_check_traceability", _PATH)
check_traceability = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
sys.modules["aib_check_traceability"] = check_traceability
_spec.loader.exec_module(check_traceability)


def test_strip_commands_removes_code_and_comments():
    text = "run `python -m pytest 2.0` and ```\nnumbers 9.9\n``` <!-- TODO 7 --> prose 1.5"
    stripped = check_traceability.strip_commands(text)
    assert "2.0" not in stripped
    assert "9.9" not in stripped
    assert "7" not in stripped
    assert "1.5" in stripped


def test_main_exit_zero_when_clean(tmp_path, capsys):
    doc = os.path.join(str(tmp_path), "README.md")
    src = os.path.join(str(tmp_path), "summary.md")
    with open(src, "w", encoding="utf-8") as fh:
        fh.write("| ASR | 0.500 |")
    with open(doc, "w", encoding="utf-8") as fh:
        fh.write("The ASR was 0.5. Run `pytest 1.0`.")
    rc = check_traceability.main(["--doc", doc, "--source", src])
    assert rc == 0
    assert "traceability OK" in capsys.readouterr().out


def test_main_exit_nonzero_on_untraceable(tmp_path, capsys):
    doc = os.path.join(str(tmp_path), "README.md")
    src = os.path.join(str(tmp_path), "summary.md")
    with open(src, "w", encoding="utf-8") as fh:
        fh.write("| ASR | 0.500 |")
    with open(doc, "w", encoding="utf-8") as fh:
        fh.write("The headline figure was 0.999.")
    rc = check_traceability.main(["--doc", doc, "--source", src])
    assert rc == 1
    assert "0.999" in capsys.readouterr().err


def test_committed_readme_skeleton_passes():
    """The committed README skeleton must have no untraceable numbers (default args)."""
    rc = check_traceability.main([])
    assert rc == 0
