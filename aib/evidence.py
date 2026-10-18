"""Prior-work license evidence and the Python package license inventory (Req 0.5-0.8, 1.1-1.3).

Reads AgentDojo's installed distribution metadata and bundled LICENSE file, fetches InjecAgent's
LICENCE from its repository, and reads the license of every package pinned in uv.lock from its
installed metadata. Everything is saved under docs/evidence/ with the access date.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import importlib.metadata as md
import json
import re
import tomllib
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

INJECAGENT_LICENSE_URL = "https://raw.githubusercontent.com/uiuc-kang-lab/InjecAgent/main/LICENCE"

_SPDX_TOKEN = re.compile(r"^[A-Za-z0-9.+-]+(?: (?:AND|OR|WITH) [A-Za-z0-9.+-]+)*$")
_KNOWN_SPDX = {
    "MIT", "MIT-0", "MIT-CMU", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause", "0BSD", "ISC", "MPL-2.0",
    "PSF-2.0", "Unlicense", "Zlib", "CC0-1.0", "BSL-1.0", "CNRI-Python", "LLVM-exception",
}
_ALIASES = {
    "apache 2.0": "Apache-2.0",
    "apache 2.0 license": "Apache-2.0",
    "apache license, version 2.0": "Apache-2.0",
    "apache software license": "Apache-2.0",
    "isc license": "ISC",
    "modified bsd license": "BSD-3-Clause",
    "mit license": "MIT",
}
_CLASSIFIERS = {
    "License :: OSI Approved :: MIT License": "MIT",
    "License :: OSI Approved :: Apache Software License": "Apache-2.0",
    "License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)": "MPL-2.0",
    "License :: OSI Approved :: The Unlicense (Unlicense)": "Unlicense",
    "License :: OSI Approved :: ISC License (ISCL)": "ISC",
    "License :: OSI Approved :: Python Software Foundation License": "PSF-2.0",
}
_BSD_CLASSIFIER = "License :: OSI Approved :: BSD License"


@dataclass(frozen=True)
class PackageLicense:
    name: str
    version: str
    license: str
    source: str  # where the identifier came from


def is_mit(text: str) -> bool:
    """True when `text` is the MIT license (its title and its grant sentence)."""
    flat = " ".join(text.split())
    return "MIT License" in flat and "Permission is hereby granted, free of charge, to any person" in flat


def copyright_lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip().lower().startswith("copyright")]


def _license_file_texts(dist: md.Distribution) -> list[str]:
    texts = []
    for f in dist.files or []:
        name = f.name.upper()
        if name.startswith(("LICENSE", "LICENCE", "COPYING")) and ".dist-info" in str(f):
            try:
                texts.append(f.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError):
                continue
    return texts


def _bsd_variant(texts: list[str]) -> str | None:
    """BSD-2-Clause or BSD-3-Clause from a license file's text; None when no BSD text is available."""
    for text in texts:
        flat = " ".join(text.split())
        if "Redistribution and use in source and binary forms" in flat:
            return "BSD-3-Clause" if "Neither the name" in flat or "names of its contributors" in flat else "BSD-2-Clause"
    return None


def classify_license(
    expression: str | None, license_field: str | None, classifiers: list[str], file_texts: list[str]
) -> tuple[str, str]:
    """Return (SPDX identifier or expression, source) from a distribution's metadata."""
    if expression:
        return expression.strip(), "License-Expression"
    field = (license_field or "").strip()
    first = field.splitlines()[0].strip() if field else ""
    if first and _SPDX_TOKEN.match(first) and all(
        t in _KNOWN_SPDX for t in re.split(r" (?:AND|OR|WITH) ", first)
    ):
        return first, "License field"
    if first.lower() in _ALIASES:
        return _ALIASES[first.lower()], "License field"
    ids, note = [], ""
    for c in classifiers:
        if c in _CLASSIFIERS:
            ids.append(_CLASSIFIERS[c])
        elif c == _BSD_CLASSIFIER:
            variant = _bsd_variant(file_texts)
            ids.append(variant or "BSD-3-Clause")
            note = " + license file" if variant else " (BSD variant not stated; 3-clause assumed)"
    if ids:
        return " AND ".join(dict.fromkeys(ids)), "Classifier" + note
    for text in file_texts:
        if is_mit(text):
            return "MIT", "license file"
    return "UNKNOWN", "none"


def package_license(dist: md.Distribution) -> PackageLicense:
    meta = dist.metadata
    ident, source = classify_license(
        meta.get("License-Expression"),
        meta.get("License"),
        [c for c in (meta.get_all("Classifier") or []) if c.startswith("License ::")],
        _license_file_texts(dist),
    )
    return PackageLicense(meta["Name"], dist.version, ident, source)


def locked_packages(lock_path: Path) -> list[dict]:
    """Every package in uv.lock except the project itself, with its resolution markers."""
    lock = tomllib.loads(lock_path.read_text(encoding="utf-8"))
    out = {}
    for p in lock["package"]:
        if p.get("source", {}).get("virtual") == ".":
            continue
        key = (p["name"], p["version"].split("+")[0])
        out.setdefault(key, {"name": p["name"], "version": p["version"]})
    return sorted(out.values(), key=lambda p: p["name"])


def pypi_metadata(name: str, version: str, fetch: Callable[[str], str] | None = None) -> dict:
    """The `info` block of PyPI's JSON API for one release."""
    return json.loads((fetch or fetch_text)(f"https://pypi.org/pypi/{name}/{version}/json"))["info"]


def package_inventory(lock_path: Path, distribution: Callable[[str], md.Distribution] = md.distribution,
                      pypi: Callable[[str, str], dict] | None = None) -> list[dict]:
    """License of every locked package from its installed metadata; packages that a platform marker
    keeps off Linux (for example Windows-only ones) fall back to PyPI's metadata for that release."""
    rows = []
    for p in locked_packages(lock_path):
        try:
            lic = package_license(distribution(p["name"]))
            rows.append({"name": p["name"], "version": lic.version, "license": lic.license, "source": lic.source})
        except md.PackageNotFoundError:
            row = {"name": p["name"], "version": p["version"], "license": "not installed on this platform",
                   "source": "uv.lock only (platform marker)"}
            if pypi is not None:
                info = pypi(p["name"], p["version"])
                ident, source = classify_license(
                    info.get("license_expression"), info.get("license"),
                    [c for c in info.get("classifiers") or [] if c.startswith("License ::")], [],
                )
                row |= {"license": ident, "source": f"PyPI JSON API, {source} (not installed on Linux: platform marker)"}
            rows.append(row)
    return rows


def agentdojo_evidence(distribution: Callable[[str], md.Distribution] = md.distribution) -> dict:
    dist = distribution("agentdojo")
    texts = _license_file_texts(dist)
    text = texts[0] if texts else ""
    classifiers = [c for c in (dist.metadata.get_all("Classifier") or []) if c.startswith("License ::")]
    return {
        "package": "agentdojo",
        "version": dist.version,
        "license_classifiers": classifiers,
        "license_expression": dist.metadata.get("License-Expression"),
        "license_file_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "copyright": copyright_lines(text),
        "is_mit": "License :: OSI Approved :: MIT License" in classifiers and is_mit(text),
        "license_text": text,
    }


def fetch_text(url: str, opener: Callable = urllib.request.urlopen, timeout: float = 30.0) -> str:
    with opener(url, timeout=timeout) as resp:
        return resp.read().decode("utf-8")


def injecagent_evidence(text: str) -> dict:
    return {
        "url": INJECAGENT_LICENSE_URL,
        "license_file_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "copyright": copyright_lines(text),
        "is_mit": is_mit(text),
        "license_text": text,
    }


def write_evidence(out_dir: Path, lock_path: Path, accessed: dt.date, fetch: Callable[[str], str] = fetch_text,
                   distribution: Callable[[str], md.Distribution] = md.distribution) -> dict:
    """Write the evidence files and return a summary; raises nothing on a non-MIT result (callers decide)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ad = agentdojo_evidence(distribution)
    ia = injecagent_evidence(fetch(INJECAGENT_LICENSE_URL))
    for name, ev in (("agentdojo", ad), ("injecagent", ia)):
        (out_dir / f"{name}_LICENSE.txt").write_text(ev["license_text"], encoding="utf-8")
        meta = {k: v for k, v in ev.items() if k != "license_text"} | {"accessed": accessed.isoformat()}
        (out_dir / f"{name}_license.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    rows = package_inventory(lock_path, distribution, lambda n, v: pypi_metadata(n, v, fetch))
    (out_dir / "python_licenses.json").write_text(
        json.dumps({"accessed": accessed.isoformat(), "packages": rows}, indent=2) + "\n", encoding="utf-8"
    )
    return {"agentdojo": {k: v for k, v in ad.items() if k != "license_text"},
            "injecagent": {k: v for k, v in ia.items() if k != "license_text"},
            "packages": rows}


def packages_markdown(rows: list[dict]) -> str:
    lines = ["| Package | Version | License | Source of the identifier |", "| --- | --- | --- | --- |"]
    lines += [f"| {r['name']} | {r['version']} | {r['license']} | {r['source']} |" for r in rows]
    return "\n".join(lines)


def replace_section(text: str, start: str, end: str, body: str) -> str:
    """Replace the text between two marker lines (inclusive markers kept); append the block if missing."""
    block = f"{start}\n{body}\n{end}"
    pattern = re.compile(re.escape(start) + r".*?" + re.escape(end), re.S)
    if pattern.search(text):
        return pattern.sub(lambda _: block, text)
    return text.rstrip("\n") + "\n\n" + block + "\n"


def today() -> dt.date:
    return dt.datetime.now(dt.timezone.utc).date()



def tool_versions(run: Callable = None) -> dict[str, str]:
    """Python version, uv version, and `uv tree` output, each from a command on this machine."""
    import subprocess
    import sys

    run = run or (lambda cmd: subprocess.run(cmd, capture_output=True, text=True, check=True).stdout)
    return {
        "python": run([sys.executable, "--version"]).strip(),
        "uv": run(["uv", "--version"]).strip(),
        "uv_tree": run(["uv", "tree", "--frozen"]),
    }


PACKAGES_START = "<!-- python-packages:start (generated by scripts/license_evidence.py) -->"
PACKAGES_END = "<!-- python-packages:end -->"


def main(root: Path, accessed: dt.date | None = None, fetch: Callable[[str], str] = fetch_text,
         distribution: Callable[[str], md.Distribution] = md.distribution, run: Callable | None = None) -> int:
    accessed = accessed or today()
    out = root / "docs" / "evidence"
    summary = write_evidence(out, root / "uv.lock", accessed, fetch, distribution)
    versions = tool_versions(run)
    (out / "python_version.txt").write_text(versions["python"] + "\n", encoding="utf-8")
    (out / "uv_version.txt").write_text(versions["uv"] + "\n", encoding="utf-8")
    (out / "uv_tree.txt").write_text(versions["uv_tree"], encoding="utf-8")
    third_party = root / "THIRD_PARTY.md"
    third_party.write_text(
        replace_section(third_party.read_text(encoding="utf-8"), PACKAGES_START, PACKAGES_END,
                        packages_markdown(summary["packages"])),
        encoding="utf-8",
    )
    ok = summary["agentdojo"]["is_mit"] and summary["injecagent"]["is_mit"]
    print(f"agentdojo {summary['agentdojo']['version']}: MIT={summary['agentdojo']['is_mit']} "
          f"{summary['agentdojo']['copyright']}")
    print(f"InjecAgent: MIT={summary['injecagent']['is_mit']} {summary['injecagent']['copyright']}")
    print(f"{len(summary['packages'])} packages from uv.lock; {versions['python']}; {versions['uv']}")
    unknown = [r["name"] for r in summary["packages"] if r["license"] == "UNKNOWN"]
    if unknown:
        print(f"packages with no license identifier: {unknown}")
    if not ok:
        print("BLOCKER: a prior-work license is not MIT; the PriorWork_Decision depends on it")
    return 0 if ok and not unknown else 1
