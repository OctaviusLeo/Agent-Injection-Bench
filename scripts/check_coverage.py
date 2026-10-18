"""Fail when a non-trivial module in aib/ or scripts/ is under 80% statement coverage (Req 0.11).

A module is non-trivial when it has more than one public function (module-level functions and
methods of public classes whose names do not start with an underscore) or more than 50 source
lines (lines that are neither blank nor comment-only). A module missing from the coverage report
counts as 0% covered.

Usage: python scripts/check_coverage.py runs/coverage.json [--root .] [--min 80]
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from dataclasses import dataclass
from pathlib import Path

PACKAGES = ("aib", "scripts")
MIN_PERCENT = 80.0
MAX_TRIVIAL_LINES = 50


@dataclass(frozen=True)
class ModuleCoverage:
    path: str
    public_functions: int
    source_lines: int
    percent: float

    @property
    def non_trivial(self) -> bool:
        return self.public_functions > 1 or self.source_lines > MAX_TRIVIAL_LINES


def count_source_lines(text: str) -> int:
    return sum(1 for line in text.splitlines() if line.strip() and not line.strip().startswith("#"))


def count_public_functions(text: str) -> int:
    tree = ast.parse(text)
    count = 0
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith("_"):
            count += 1
        elif isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
            count += sum(
                1
                for item in node.body
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and not item.name.startswith("_")
            )
    return count


def load_percentages(report: dict, root: Path) -> dict[str, float]:
    """Map repository-relative POSIX paths to percent covered."""
    out: dict[str, float] = {}
    for name, data in report.get("files", {}).items():
        path = Path(name)
        if path.is_absolute():
            try:
                path = path.resolve().relative_to(root.resolve())
            except ValueError:
                continue
        out[path.as_posix()] = float(data["summary"]["percent_covered"])
    return out


def find_modules(root: Path) -> list[Path]:
    modules: list[Path] = []
    for package in PACKAGES:
        base = root / package
        if base.is_dir():
            modules.extend(p for p in sorted(base.rglob("*.py")) if p.name != "__init__.py" and "__pycache__" not in p.parts)
    return modules


def assess(report: dict, root: Path) -> list[ModuleCoverage]:
    percentages = load_percentages(report, root)
    results = []
    for module in find_modules(root):
        rel = module.relative_to(root).as_posix()
        text = module.read_text(encoding="utf-8")
        results.append(
            ModuleCoverage(rel, count_public_functions(text), count_source_lines(text), percentages.get(rel, 0.0))
        )
    return results


def failures(results: list[ModuleCoverage], minimum: float = MIN_PERCENT) -> list[ModuleCoverage]:
    return [r for r in results if r.non_trivial and r.percent < minimum]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("report", help="coverage JSON written by pytest-cov")
    parser.add_argument("--root", default=".", help="repository root")
    parser.add_argument("--min", type=float, default=MIN_PERCENT, help="minimum percent for non-trivial modules")
    args = parser.parse_args(argv)
    root = Path(args.root)
    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    results = assess(report, root)
    failed = failures(results, args.min)
    checked = [r for r in results if r.non_trivial]
    print(f"coverage check: {len(checked)} non-trivial modules, minimum {args.min:.0f}%")
    for r in failed:
        print(f"FAIL {r.path}: {r.percent:.1f}% ({r.public_functions} public functions, {r.source_lines} source lines)")
    if failed:
        print(f"{len(failed)} module(s) under {args.min:.0f}% statement coverage")
        return 1
    print("all non-trivial modules meet the minimum")
    return 0


if __name__ == "__main__":
    sys.exit(main())
