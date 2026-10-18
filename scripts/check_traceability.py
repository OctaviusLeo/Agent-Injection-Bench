"""Fail when a number in README.md or docs/REPORT.md cannot be traced to a file generated under results/.

Usage: python scripts/check_traceability.py [--doc README.md --doc docs/REPORT.md] [--results results]
Prints each untraceable number and exits 1 when any exist (Req 9.4).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from aib.traceability import check_files  # noqa: E402


def main(argv: list[str] | None = None, root: Path = ROOT) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--doc", action="append", help="document to check (repeatable)")
    parser.add_argument("--results", default="results")
    args = parser.parse_args(argv)
    docs = [root / d for d in (args.doc or ["README.md", "docs/REPORT.md"])]
    results = check_files(docs, root / args.results)
    failed = False
    for doc, res in results.items():
        name = Path(doc).relative_to(root).as_posix() if Path(doc).is_relative_to(root) else doc
        if res is None:
            print(f"{name}: not found (skipped)")
            continue
        if res.ok:
            print(f"{name}: every number traces to a file under {args.results}/")
            continue
        failed = True
        for number in res.untraceable:
            print(f"{name}: UNTRACEABLE number {number}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
