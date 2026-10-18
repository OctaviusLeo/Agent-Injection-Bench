"""Automated milestone gates (replace the owner review stops). Each check returns a list of failures.

Usage: python -m aib.gates <m0|m1|m2|m3|m4|m5>   (scripts/verify_mN.sh runs the unit tests first)
"""

from __future__ import annotations

import json
import re
import sys
from collections.abc import Callable
from pathlib import Path

from aib.repro import REPO_ROOT

_HEX40 = re.compile(r"^[0-9a-f]{40}$")


def load_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def stamp_ok(data: dict | None) -> bool:
    stamp = (data or {}).get("stamp") or {}
    return bool(_HEX40.match(str(stamp.get("commit", "")))) and "command" in stamp and "seed" in stamp


def without_stamp(data: dict) -> dict:
    return {k: v for k, v in data.items() if k != "stamp"}


# ---------------------------------------------------------------- M1
def check_m1(root: Path = REPO_ROOT, regenerate: Callable[[], dict] | None = None) -> list[str]:
    failures: list[str] = []
    path = root / "results" / "m1_catalog.json"
    cat = load_json(path)
    if cat is None:
        return [f"{path.relative_to(root)} is missing or not JSON"]
    tasks = cat.get("user_tasks", [])
    if len(tasks) != 40 or cat.get("user_task_count") != 40:
        failures.append(f"m1_catalog: expected 40 user tasks, found {len(tasks)}")
    for t in tasks:
        if not t.get("ground_truth_passes"):
            failures.append(f"m1_catalog: {t.get('id')} ground truth does not pass its check")
        if not t.get("injection_placeholders"):
            failures.append(f"m1_catalog: {t.get('id')} reads no injection placeholder")
    if int(cat.get("tool_count", 0)) < 4:
        failures.append(f"m1_catalog: fewer than 4 tools ({cat.get('tool_count')})")
    if (cat.get("validate_catalog") or {}).get("failures"):
        failures.append(f"m1_catalog: validate_catalog failures {cat['validate_catalog']['failures']}")
    if not stamp_ok(cat):
        failures.append("m1_catalog: missing stamp with a 40-character commit")
    if regenerate is None:
        from aib.catalog import build_catalog as regenerate
    regen = regenerate()
    out = root / "runs" / "m1_catalog_regen.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(regen, indent=2) + "\n", encoding="utf-8")
    if regen != without_stamp(cat):
        failures.append("m1_catalog: regenerating the catalog into runs/ gives different content")
    return failures


# ---------------------------------------------------------------- M2
def check_m2(root: Path = REPO_ROOT) -> list[str]:
    path = root / "results" / "m2_stub_validation.json"
    data = load_json(path)
    if data is None:
        return [f"{path.relative_to(root)} is missing or not JSON"]
    failures: list[str] = []
    cases = data.get("cases", [])
    if len(cases) != 400:
        failures.append(f"m2_stub_validation: expected 400 cases, found {len(cases)}")
    fu = [c for c in cases if c.get("policy") == "follow_user"]
    fi = [c for c in cases if c.get("policy") == "follow_injection"]
    if len(fu) != 200 or not all(c.get("exposed") is True and c.get("task_success") is True
                                 and c.get("injection_success") is False for c in fu):
        bad = [f"{c.get('style')}/{c.get('user_task_id')}" for c in fu
               if not (c.get("exposed") is True and c.get("task_success") is True and c.get("injection_success") is False)]
        failures.append(f"m2_stub_validation: follow_user must be 200 exposed cases with task success and no injection "
                        f"success (found {len(fu)}, failing {bad[:10]})")
    if len(fi) != 200 or not all(c.get("injection_success") is True for c in fi):
        bad = [f"{c.get('style')}/{c.get('user_task_id')}" for c in fi if c.get("injection_success") is not True]
        failures.append(f"m2_stub_validation: follow_injection must be 200 cases with injection success "
                        f"(found {len(fi)}, failing {bad[:10]})")
    styles = {c.get("style") for c in cases}
    if len(styles) != 5 or not {"direct", "fake_system", "hidden_field"} <= styles:
        failures.append(f"m2_stub_validation: expected 5 styles incl. direct, fake_system, hidden_field; found {sorted(map(str, styles))}")
    goals = {c.get("injection_task_id") for c in cases}
    if goals != {f"injection_task_{i}" for i in range(6)}:
        failures.append(f"m2_stub_validation: goals must be exactly injection tasks 0-5; found {sorted(map(str, goals))}")
    if not stamp_ok(data):
        failures.append("m2_stub_validation: missing stamp with a 40-character commit")
    return failures


# ---------------------------------------------------------------- M0
_CITED_PATH = re.compile(r"`((?:docs|results|config|scripts|tests|aib|docker|\.github)/[^`\s<>*]+)`")
GITIGNORE_ENTRIES = (".env", "*.env", "secrets.*", "*.key", "*.pem", "*.token", "runs/", ".venv/", "__pycache__/",
                     ".pytest_cache/", ".hypothesis/", ".coverage", "htmlcov/", "*.gguf", "*.safetensors", "*.onnx", "*.bin")
GITATTRIBUTES_ENTRIES = ("* text=auto eol=lf", "*.png binary")


def cited_paths(text: str) -> list[str]:
    """Repository paths cited in backticks (a `::test` suffix is dropped)."""
    return sorted({m.group(1).split("::")[0] for m in _CITED_PATH.finditer(text)})


def m0_outcome_block(design: str) -> str:
    m = re.search(r"^## M0 outcome\n(.*?)(?=^## )", design, re.S | re.M)
    return m.group(1) if m else ""


def check_m0(root: Path = REPO_ROOT) -> list[str]:
    from aib.config import load_all

    failures: list[str] = []
    cfg = load_all(root / "config")
    in_use = cfg.models.local_models()
    models = load_json(root / "results" / "m0_models.json")
    if models is None:
        failures.append("results/m0_models.json is missing")
    else:
        if not stamp_ok(models):
            failures.append("m0_models: missing stamp with a 40-character commit")
        entries = models.get("entries", {})
        wanted = {k: cfg.models.models[k].files for k in in_use} | {"classifier": cfg.models.classifier.files}
        for key, files in wanted.items():
            got = {f["name"]: f for f in (entries.get(key) or {}).get("files", [])}
            for f in files:
                g = got.get(f.name)
                if g is None or not g.get("size_ok") or g.get("size") != f.size or not re.fullmatch(r"[0-9a-f]{64}", str(g.get("sha256"))):
                    failures.append(f"m0_models: {key}/{f.name} missing, wrong size, or without SHA-256")
    for key in in_use:
        smoke = load_json(root / "results" / f"m0_smoke_{key}.json")
        if smoke is None:
            failures.append(f"results/m0_smoke_{key}.json is missing")
        elif not (smoke.get("python_ok") and smoke.get("docker_ok") and str(smoke.get("response", "")).strip()
                  and smoke.get("peak_mib") is not None and smoke.get("total_mib") is not None
                  and smoke["peak_mib"] <= smoke["total_mib"] and stamp_ok(smoke)):
            failures.append(f"m0_smoke_{key}: needs Python >= 3.11, Docker working, a non-empty response, peak <= total MiB, "
                            f"and a stamp (failed condition: {smoke.get('failed_condition')})")
        probe = load_json(root / "results" / f"m0_tool_probe_{key}.json")
        if probe is None or not probe.get("passed") or not any(c.get("function") == "get_current_day" for c in probe.get("tool_calls", [])):
            failures.append(f"m0_tool_probe_{key}: no parsed tool call to get_current_day")
    hosted = load_json(root / "results" / "m0_hosted_check.json")
    if hosted is None:
        failures.append("results/m0_hosted_check.json is missing")
    else:
        if not str((hosted.get("tool_free") or {}).get("content") or "").strip():
            failures.append("m0_hosted_check: empty tool-free response")
        if not any(c.get("function") == "get_current_day" for c in (hosted.get("tool_request") or {}).get("tool_calls", [])):
            failures.append("m0_hosted_check: no tool call")
    from aib.hosted import make_ledger

    total = make_ledger(cfg.hosted, root).total_usd
    if total > cfg.hosted.spend_cap_usd:
        failures.append(f"hosted ledger total ${total:.4f} exceeds the Spend_Cap ${cfg.hosted.spend_cap_usd:.2f}")
    env = (root / "docs" / "ENVIRONMENT.md").read_text(encoding="utf-8")
    if "TBD" in env:
        failures.append("docs/ENVIRONMENT.md still contains TBD")
    for path in cited_paths(env):
        if not (root / path).exists():
            failures.append(f"docs/ENVIRONMENT.md cites a missing path: {path}")
    block = m0_outcome_block((root / ".kiro" / "specs" / "agent-injection-bench" / "design.md").read_text(encoding="utf-8"))
    if not block.strip() or re.search(r"^- .*TBD", block, re.M):
        failures.append("design.md M0 outcome block is missing or still has TBD lines")
    third = (root / "THIRD_PARTY.md").read_text(encoding="utf-8")
    needed = {"AgentDojo": "MIT", "InjecAgent": "MIT", "llama.cpp": "MIT", cfg.models.classifier.repo: "Apache-2.0"}
    needed |= {cfg.models.models[k].repo: None for k in in_use}
    for name, lic in needed.items():
        lines = [line for line in third.splitlines() if name in line]
        if not lines or (lic and not any(lic in line for line in lines)) or (lic is None and not any("|" in line for line in lines)):
            failures.append(f"THIRD_PARTY.md does not list {name}" + (f" with license {lic}" if lic else " with a license"))
    compose = (root / "docker" / "compose.yaml").read_text(encoding="utf-8")
    if not re.search(r"image: \S+@sha256:[0-9a-f]{64}", compose):
        failures.append("docker/compose.yaml does not pin the image by digest")
    gi = (root / ".gitignore").read_text(encoding="utf-8").splitlines()
    missing = [e for e in GITIGNORE_ENTRIES if e not in gi]
    if missing:
        failures.append(f".gitignore lacks {missing}")
    ga = (root / ".gitattributes").read_text(encoding="utf-8").splitlines()
    if [e for e in GITATTRIBUTES_ENTRIES if e not in ga]:
        failures.append(".gitattributes lacks the design's entries")
    return failures


# ---------------------------------------------------------------- M3
VALID_STATUSES = {"errored", "clean", "unclassified_not_exposed", "classified"}
MAX_ERRORED_SHARE = 0.05


def check_store(path: Path, expected_ids: set[str]) -> list[str]:
    from aib.metrics import load_store

    name = f"results/raw/{path.name}"
    if not path.exists():
        return [f"{name} is missing"]
    header, records = load_store(path)
    failures = []
    if not header or not _HEX40.match(str((header.get("stamp") or {}).get("commit", ""))):
        failures.append(f"{name}: header with a 40-character commit missing")
    ids = [r["case_id"] for r in records]
    if len(ids) != len(set(ids)):
        failures.append(f"{name}: {len(ids) - len(set(ids))} duplicate case ids")
    if set(ids) != expected_ids:
        failures.append(f"{name}: {len(expected_ids - set(ids))} grid cases missing, {len(set(ids) - expected_ids)} unexpected")
    bad = [r["case_id"] for r in records if r.get("status") not in VALID_STATUSES]
    if bad:
        failures.append(f"{name}: invalid statuses in {bad[:5]}")
    errored = sum(r.get("status") == "errored" for r in records)
    if records and errored / len(records) > MAX_ERRORED_SHARE:
        failures.append(f"{name}: {errored} errored cases ({errored / len(records):.1%}) exceed 5%")
    filt = [r["case_id"] for r in records if r.get("defense") == "filter" and r.get("status") != "errored"
            and len(r.get("filter_decisions") or []) != r.get("n_tool_outputs", -1)]
    if filt:
        failures.append(f"{name}: filter cases without a decision for every screened output: {filt[:5]}")
    return failures


def check_m3(root: Path = REPO_ROOT) -> list[str]:
    from aib.config import load_all
    from aib.experiments import build_grid

    cfg = load_all(root / "config")
    failures: list[str] = []
    for key in cfg.models.local_models():
        ids = {s.case_id for s in build_grid(key, cfg.experiment)}
        failures += check_store(root / "results" / "raw" / f"{key}.jsonl", ids)
    for rel in ("results/m3_filter_calibration.json", "results/pilot.json"):
        if not stamp_ok(load_json(root / rel)):
            failures.append(f"{rel} is missing or has no stamp")
    return failures


# ---------------------------------------------------------------- M4
def _rate_ok(entry: dict | None) -> bool:
    if entry is None:
        return True
    if entry.get("not_computed"):
        return bool(entry.get("cause"))
    lo, hi = (-1.0, 1.0) if "paired" in str(entry.get("method")) else (0.0, 1.0)
    return lo <= entry["lower"] <= entry["point"] <= entry["upper"] <= hi


def strip_stamp_lines(text: str) -> str:
    return "\n".join(line for line in text.splitlines()
                     if not re.match(r"^- (Table|Metrics|Report|README|Results_Table): command `", line))


def check_m4(root: Path = REPO_ROOT, regenerate: Callable[[], dict] | None = None) -> list[str]:
    from aib.config import load_all
    from aib.experiments import build_grid
    from aib.hosted import make_ledger
    from aib.metrics import compute_metrics, load_store
    from aib.results_table import render_summary

    cfg = load_all(root / "config")
    failures: list[str] = []
    hosted_path = root / "results" / "raw" / f"{cfg.hosted.model}.jsonl"
    if not hosted_path.exists():
        failures.append(f"{hosted_path.relative_to(root)} is missing")
    else:
        specs = build_grid(cfg.hosted.model, cfg.experiment, hosted=True, hosted_conditions=cfg.hosted.conditions,
                           hosted_defenses=cfg.hosted.defenses)
        grid = {s.case_id for s in specs}
        _, records = load_store(hosted_path)
        ids = [r["case_id"] for r in records]
        if set(ids) - grid:
            failures.append(f"hosted store holds {len(set(ids) - grid)} cases outside the hosted grid")
        if len(ids) != len(set(ids)):
            failures.append("hosted store repeats a case id")
        summary = load_json(root / "results" / "hosted_run.json") or {}
        if set(ids) != grid and not summary.get("stopped_at_cap"):
            failures.append("hosted store is incomplete and results/hosted_run.json records no Spend_Cap stop")
        if not stamp_ok(summary):
            failures.append("results/hosted_run.json is missing or has no stamp")
    total = make_ledger(cfg.hosted, root).total_usd
    if total > cfg.hosted.spend_cap_usd:
        failures.append(f"hosted ledger total ${total:.4f} exceeds the Spend_Cap ${cfg.hosted.spend_cap_usd:.2f}")
    metrics = load_json(root / "results" / "metrics.json")
    if metrics is None:
        return failures + ["results/metrics.json is missing"]
    present = set()
    for path in sorted((root / "results" / "raw").glob("*.jsonl")):
        _, records = load_store(path)
        present |= {(r["model"], r["defense"], r["condition"]) for r in records}
    covered = {(c["model"], c["defense"], c["condition"]) for c in metrics.get("cells", [])}
    if present - covered:
        failures.append(f"metrics.json misses {len(present - covered)} Cells present in the stores")
    entries = [e for c in metrics.get("cells", []) for e in (c["asr"], c["tsr"], c["exposure"])]
    entries += [e for p in metrics.get("pooled", []) for e in (p["asr"], p["tsr"], p["exposure"])]
    entries += [e for d in metrics.get("defense_cost", []) for e in (d["defense_cost"], d["asr_reduction"], d["tsr_none"], d["tsr_defense"])]
    entries += [e for f in metrics.get("filter", []) for e in (f.get("precision"), f.get("recall"), f["false_positive_rate"])]
    if not all(_rate_ok(e) for e in entries):
        failures.append("metrics.json has a rate whose interval does not bracket it, or a not-computed rate without a cause")
    if not stamp_ok(metrics):
        failures.append("results/metrics.json has no stamp")
    settings = metrics.get("settings", {})
    regen = (regenerate or (lambda: compute_metrics(root / "results" / "raw", settings.get("seed"), settings.get("resamples"))))()
    (root / "runs").mkdir(exist_ok=True)
    (root / "runs" / "metrics_regen.json").write_text(json.dumps(regen, indent=2) + "\n", encoding="utf-8")
    if json.dumps(without_stamp(metrics) | {"settings": {**settings, "store": None}}, sort_keys=True) != json.dumps(
            regen | {"settings": {**regen.get("settings", {}), "store": None}}, sort_keys=True):
        failures.append("regenerating results/metrics.json gives different content")
    summary_path = root / "results" / "summary.md"
    if not summary_path.exists():
        failures.append("results/summary.md is missing")
    else:
        regen_summary = render_summary({**regen, "stamp": metrics.get("stamp", {})}, {"command": "", "seed": 0, "commit": ""})
        if strip_stamp_lines(regen_summary) != strip_stamp_lines(summary_path.read_text(encoding="utf-8")):
            failures.append("regenerating results/summary.md gives different content")
    for plot in ("asr_by_defense.png", "defense_cost.png"):
        p = root / "results" / "plots" / plot
        if not p.exists() or p.stat().st_size == 0:
            failures.append(f"results/plots/{plot} is missing or empty")
    return failures


# ---------------------------------------------------------------- M5
REPORT_SECTIONS = ("## Question", "## Setup", "## Results", "## Limitations", "## Threats to validity")
README_SECTIONS = ("```mermaid", "## Run it", "[docs/REPORT.md](docs/REPORT.md)", "`bash scripts/test_unit.sh`",
                   "`uv run --frozen python scripts/ci_smoke.py`", "`bash scripts/run_benchmark.sh <model key>`",
                   "`bash scripts/run_all_local.sh`", "`bash scripts/build_results.sh`", "## Known limitations",
                   "## Safety and scope", "All tools are simulated", "This repository is private.")


def check_m5(root: Path = REPO_ROOT) -> list[str]:
    failures: list[str] = []
    before = root / "runs" / "m5_before"
    for name in ("results/summary.md", "docs/REPORT.md", "README.md"):
        a, b = root / name, before / name
        if not a.exists():
            failures.append(f"{name} is missing")
        elif not b.exists() or strip_stamp_lines(a.read_text(encoding="utf-8")) != strip_stamp_lines(b.read_text(encoding="utf-8")):
            failures.append(f"running scripts/build_results.sh again changed {name} (or no earlier copy in runs/m5_before)")
    report = (root / "docs" / "REPORT.md").read_text(encoding="utf-8") if (root / "docs" / "REPORT.md").exists() else ""
    for s in REPORT_SECTIONS:
        if s not in report:
            failures.append(f"docs/REPORT.md lacks {s}")
    if not re.search(r"temperature \S+ and top-p \S+", report) or "Repetitions" not in report:
        failures.append("docs/REPORT.md does not state the temperature, top-p, and Repetitions")
    readme = (root / "README.md").read_text(encoding="utf-8") if (root / "README.md").exists() else ""
    for s in README_SECTIONS:
        if s not in readme:
            failures.append(f"README.md lacks {s}")
    head = (root / "runs" / "head_commit.txt").read_text(encoding="utf-8").strip() if (root / "runs" / "head_commit.txt").exists() else ""
    ci = load_json(root / "runs" / "ci_latest.json") or {}
    if not head or ci.get("headSha") != head or ci.get("conclusion") != "success":
        failures.append("the latest CI run on main is not a success for HEAD (runs/ci_latest.json, written from PowerShell)")
    red = load_json(root / "docs" / "evidence" / "ci_red_check.json") or {}
    if (red.get("red") or {}).get("conclusion") != "failure":
        failures.append("docs/evidence/ci_red_check.json does not show a failed red run")
    env = (root / "docs" / "ENVIRONMENT.md").read_text(encoding="utf-8")
    third = _norm((root / "THIRD_PARTY.md").read_text(encoding="utf-8"))
    for name in third_party_names(env):
        if _norm(name) not in third:
            failures.append(f"THIRD_PARTY.md does not list {name}, named in docs/ENVIRONMENT.md")
    design = (root / ".kiro" / "specs" / "agent-injection-bench" / "design.md").read_text(encoding="utf-8")
    if "Responsible disclosure" not in design or "responsible-disclosure process" not in design:
        failures.append("design.md does not state the responsible-disclosure rule")
    return failures


def _norm(text: str) -> str:
    return text.lower().replace("_", "-")


def third_party_names(env: str) -> list[str]:
    """Packages, models, and images named in docs/ENVIRONMENT.md."""
    names = set()
    m = re.search(r"Direct dependencies[^:]*:(.*)", env)
    if m:
        names |= {tok.split()[0] for tok in m.group(1).split(",") if tok.strip()}
    names |= set(re.findall(r"`([\w.-]+/[\w.-]+)` at revision", env))
    names |= set(re.findall(r"ghcr\.io/[\w.-]+/[\w.-]+", env))
    return sorted(n for n in names if n)


GATES: dict[str, Callable[..., list[str]]] = {"m0": check_m0, "m1": check_m1, "m2": check_m2, "m3": check_m3,
                                              "m4": check_m4, "m5": check_m5}


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1 or argv[0] not in GATES:
        print(f"usage: python -m aib.gates <{'|'.join(GATES)}>")
        return 2
    failures = GATES[argv[0]]()
    for f in failures:
        print(f"GATE FAIL: {f}")
    print(f"gate {argv[0]}: {'PASS' if not failures else f'FAIL ({len(failures)} problems)'}")
    return 0 if not failures else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
