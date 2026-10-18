"""CLI wrapper for the README/Report number-traceability check (Req 7.4, 9.3,
9.4; Property 11).

Reports each untraceable number in a README/Report and exits non-zero when any
exist, so the build fails if a reported number cannot be traced to a generated
source (Req 9.3, 9.4). For the committed README skeleton - which carries no
reported Measured_Values, only documented commands and marked ``[ME]``
placeholders - the check exits ``0``.

What counts as prose vs. a command
----------------------------------
Numbers inside fenced code blocks and inline-code spans are **literal commands**
the reader runs (for example ``python -m pytest --cov=aib``), not reported
Measured_Values, so they are stripped before the check. Every remaining number -
the prose a reader would take as a result - must trace to a generated source:
``results/summary.md`` (the Results_Table), the Results_Store, or the generated
Report, whichever exist on disk.

Single command: ``python scripts/check_traceability.py``.
"""

from __future__ import annotations

import argparse
import re
import sys
from typing import Sequence

from aib.traceability import check_text

#: Default README path (worktree root), resolved relative to this script.
_HERE = __import__("os").path.dirname(__import__("os").path.dirname(__file__))


def _default(path: str) -> str:
    import os

    return os.path.join(_HERE, path)


#: Fenced code block (```...```) and inline code span (`...`) matchers; their
#: contents are documented commands, not reported numbers, so they are stripped.
_FENCED_RE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_RE = re.compile(r"`[^`]*`")
#: HTML comments carry the ``<!-- TODO [ME]: ... -->`` owner placeholders.
_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)


def strip_commands(text: str) -> str:
    """Remove code blocks, inline code, and HTML comments before checking.

    Those regions hold literal commands and owner-placeholder markers rather than
    reported Measured_Values; the traceability contract applies to the prose that
    states results (Req 7.4).
    """
    text = _FENCED_RE.sub(" ", text)
    text = _INLINE_RE.sub(" ", text)
    text = _COMMENT_RE.sub(" ", text)
    return text


def _read(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return fh.read()
    except FileNotFoundError:
        return ""


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python scripts/check_traceability.py",
        description=(
            "Fail if any number reported in the README/Report cannot be traced "
            "to a generated source (results/summary.md, the Results_Store, or the "
            "generated Report)."
        ),
    )
    parser.add_argument(
        "--doc",
        default=_default("README.md"),
        help="README/Report file to check (default: README.md at the worktree root)",
    )
    parser.add_argument(
        "--source",
        action="append",
        default=None,
        help="a generated source file numbers may trace to (repeatable)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the traceability check and exit non-zero on any untraceable number."""
    args = _build_parser().parse_args(argv)

    sources = args.source or [
        _default("results/summary.md"),
        _default("results/store.jsonl"),
        _default("results/report.md"),
    ]
    doc_text = strip_commands(_read(args.doc))
    source_texts = [_read(p) for p in sources]

    result = check_text(doc_text, source_texts)
    if result.ok:
        print(f"traceability OK: no untraceable numbers in {args.doc}")
        return 0

    print(
        f"traceability FAILED: {len(result.untraceable)} untraceable number(s) in "
        f"{args.doc}:",
        file=sys.stderr,
    )
    for number in result.untraceable:
        print(f"  - {number}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
