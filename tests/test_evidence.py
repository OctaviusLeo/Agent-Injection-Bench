"""Unit tests for aib.evidence with fake distributions and a fake fetch (no network)."""

from __future__ import annotations

import datetime as dt
import importlib.metadata as md
import json
from email.message import Message

import pytest

from aib import evidence as ev

MIT_TEXT = """MIT License

Copyright (c) 2024 Example Authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction.
"""
BSD3 = "Redistribution and use in source and binary forms, with or without modification. Neither the name of the copyright holder"
BSD2 = "Redistribution and use in source and binary forms, with or without modification, are permitted."


class _File:
    def __init__(self, path, text):
        self._path, self._text = path, text
        self.name = path.rsplit("/", 1)[-1]

    def __str__(self):
        return self._path

    def read_text(self, encoding="utf-8"):
        return self._text


class _Dist:
    def __init__(self, name, version, expression=None, license=None, classifiers=(), files=()):
        msg = Message()
        msg["Name"] = name
        if expression:
            msg["License-Expression"] = expression
        if license:
            msg["License"] = license
        for c in classifiers:
            msg["Classifier"] = c
        self.metadata = msg
        self.version = version
        self.files = [_File(f"{name}-{version}.dist-info/{p}", t) for p, t in files]


def _registry(dists):
    def distribution(name):
        if name not in dists:
            raise md.PackageNotFoundError(name)
        return dists[name]

    return distribution


def test_is_mit_and_copyright():
    assert ev.is_mit(MIT_TEXT)
    assert not ev.is_mit("Apache License Version 2.0")
    assert ev.copyright_lines(MIT_TEXT) == ["Copyright (c) 2024 Example Authors"]


@pytest.mark.parametrize(
    "expr,field,classifiers,files,expected",
    [
        ("Apache-2.0 OR MIT", None, [], [], ("Apache-2.0 OR MIT", "License-Expression")),
        (None, "MIT", [], [], ("MIT", "License field")),
        (None, "Apache 2.0", [], [], ("Apache-2.0", "License field")),
        (None, "Modified BSD License", [], [], ("BSD-3-Clause", "License field")),
        (None, "long text\nmore", ["License :: OSI Approved :: MIT License"], [], ("MIT", "Classifier")),
        (None, None, ["License :: OSI Approved :: BSD License"], [BSD3], ("BSD-3-Clause", "Classifier + license file")),
        (None, None, ["License :: OSI Approved :: BSD License"], [BSD2], ("BSD-2-Clause", "Classifier + license file")),
        (None, "Dual License", ["License :: OSI Approved :: BSD License", "License :: OSI Approved :: Apache Software License"],
         [BSD3], ("BSD-3-Clause AND Apache-2.0", "Classifier + license file")),
        (None, None, ["License :: OSI Approved :: BSD License"], [],
         ("BSD-3-Clause", "Classifier (BSD variant not stated; 3-clause assumed)")),
        (None, None, [], [MIT_TEXT], ("MIT", "license file")),
        (None, None, [], [], ("UNKNOWN", "none")),
    ],
)
def test_classify_license(expr, field, classifiers, files, expected):
    assert ev.classify_license(expr, field, classifiers, files) == expected


def test_locked_packages_skips_project_and_duplicate_local_versions(tmp_path):
    lock = tmp_path / "uv.lock"
    lock.write_text(
        'version = 1\n'
        '[[package]]\nname = "aib"\nversion = "0.1.0"\nsource = { virtual = "." }\n'
        '[[package]]\nname = "torch"\nversion = "2.14.1"\nsource = { registry = "x" }\n'
        '[[package]]\nname = "torch"\nversion = "2.14.1+cpu"\nsource = { registry = "x" }\n'
        '[[package]]\nname = "colorama"\nversion = "0.4.6"\nsource = { registry = "x" }\n',
        encoding="utf-8",
    )
    assert [p["name"] for p in ev.locked_packages(lock)] == ["colorama", "torch"]
    rows = ev.package_inventory(lock, _registry({"torch": _Dist("torch", "2.14.1+cpu", expression="BSD-3-Clause")}))
    assert rows[0]["license"] == "not installed on this platform"
    assert rows[1] == {"name": "torch", "version": "2.14.1+cpu", "license": "BSD-3-Clause", "source": "License-Expression"}


def test_replace_section_inserts_then_replaces():
    text = "# T\n\nintro\n"
    once = ev.replace_section(text, "<!-- a -->", "<!-- b -->", "one")
    assert once.endswith("<!-- a -->\none\n<!-- b -->\n")
    twice = ev.replace_section(once, "<!-- a -->", "<!-- b -->", "two")
    assert "one" not in twice and twice.count("<!-- a -->") == 1 and "two" in twice
    md_table = ev.packages_markdown([{"name": "x", "version": "1", "license": "MIT", "source": "s"}])
    assert md_table.splitlines()[-1] == "| x | 1 | MIT | s |"


def _fixture_root(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "THIRD_PARTY.md").write_text("# Third\n", encoding="utf-8")
    (tmp_path / "uv.lock").write_text(
        'version = 1\n[[package]]\nname = "agentdojo"\nversion = "0.1.35"\nsource = { registry = "x" }\n', encoding="utf-8"
    )
    return tmp_path


def _fake_run(cmd):
    return {"--version": "Python 3.12.3\n" if "python" in cmd[0] else "uv 0.13.1\n", "tree": "aib v0.1.0\n"}[cmd[-1] if cmd[-1] != "--frozen" else "tree"]


def test_main_writes_evidence_and_passes_for_mit(tmp_path):
    root = _fixture_root(tmp_path)
    dists = _registry({"agentdojo": _Dist("agentdojo", "0.1.35", classifiers=["License :: OSI Approved :: MIT License"],
                                          files=[("licenses/LICENSE", MIT_TEXT)])})
    code = ev.main(root, dt.date(2026, 10, 17), fetch=lambda url: MIT_TEXT, distribution=dists, run=_fake_run)
    assert code == 0
    out = root / "docs" / "evidence"
    meta = json.loads((out / "agentdojo_license.json").read_text())
    assert meta["is_mit"] and meta["version"] == "0.1.35" and meta["accessed"] == "2026-10-17"
    assert json.loads((out / "injecagent_license.json").read_text())["url"] == ev.INJECAGENT_LICENSE_URL
    assert (out / "uv_tree.txt").read_text() == "aib v0.1.0\n"
    assert (out / "python_version.txt").read_text() == "Python 3.12.3\n"
    assert "| agentdojo | 0.1.35 | MIT | Classifier |" in (root / "THIRD_PARTY.md").read_text()


def test_main_fails_when_a_license_is_not_mit(tmp_path, capsys):
    root = _fixture_root(tmp_path)
    dists = _registry({"agentdojo": _Dist("agentdojo", "0.1.35", classifiers=["License :: OSI Approved :: MIT License"],
                                          files=[("licenses/LICENSE", MIT_TEXT)])})
    code = ev.main(root, dt.date(2026, 10, 17), fetch=lambda url: "GPL-3.0", distribution=dists, run=_fake_run)
    assert code == 1
    assert "BLOCKER" in capsys.readouterr().out


def test_fetch_text_uses_opener():
    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b"hello"

    assert ev.fetch_text("http://example.invalid", opener=lambda url, timeout: _Resp()) == "hello"


def test_tool_versions_default_runner_is_used(monkeypatch):
    calls = []
    assert ev.tool_versions(lambda cmd: calls.append(cmd) or "x\n")["uv"] == "x"
    assert calls[-1] == ["uv", "tree", "--frozen"]


def test_off_platform_package_falls_back_to_pypi_metadata(tmp_path):
    lock = tmp_path / "uv.lock"
    lock.write_text('version = 1\n[[package]]\nname = "colorama"\nversion = "0.4.6"\nsource = { registry = "x" }\n',
                    encoding="utf-8")
    payload = json.dumps({"info": {"license": None, "classifiers": ["License :: OSI Approved :: MIT License"]}})
    urls = []
    rows = ev.package_inventory(lock, _registry({}),
                                lambda n, v: ev.pypi_metadata(n, v, lambda url: urls.append(url) or payload))
    assert rows == [{"name": "colorama", "version": "0.4.6", "license": "MIT",
                     "source": "PyPI JSON API, Classifier (not installed on Linux: platform marker)"}]
    assert urls == ["https://pypi.org/pypi/colorama/0.4.6/json"]
