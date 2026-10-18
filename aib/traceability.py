"""README / Report number-traceability check (design.md 'Fabricated-number
prevention'; Req 7.4, 9.3, 9.4; Property 11).

Every number a human reads in the README or the Report must be a Measured_Value
that came out of a repository script - not a figure typed in by hand. This module
enforces that: it extracts the numeric tokens from a README/Report text and
confirms each one also appears in at least one **generated source** (the
Results_Table ``results/summary.md``, the Results_Store, or any other
script-generated file passed in). A number that cannot be traced to a generated
source is reported, and the overall check fails when any untraceable number
exists (Req 9.3, 9.4).

Number model
------------
A "number" is a maximal run matched by :data:`NUMBER_RE` (an optional sign, then
digits with optional decimals / grouping commas). Each extracted token is
normalized to a canonical string via :func:`normalize_number` so that cosmetic
differences never change the verdict: ``+0.5``, ``0.50``, and ``0.500`` all
normalize equal, and a value present in a generated source traces regardless of
how many trailing zeros the prose used. The traceable set is the union of the
normalized numbers found across every generated source.

The check returns the list of untraceable numbers (as they appeared in the text)
and a boolean status; the CLI wrapper :mod:`scripts.check_traceability` prints
each untraceable number and exits non-zero when any exist.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Sequence

#: Matches a number: optional sign, digits with optional grouping commas, and an
#: optional decimal fraction. Anchored so a run like ``12,345.60`` is one token.
NUMBER_RE = re.compile(r"[+-]?\d[\d,]*(?:\.\d+)?")


@dataclass(frozen=True)
class TraceResult:
    """Outcome of a traceability check (Req 9.3, 9.4).

    ``ok`` is ``True`` only when no untraceable number was found.
    ``untraceable`` holds each offending number as it appeared in the text, in
    first-seen order (duplicates reported once).
    """

    ok: bool
    untraceable: tuple[str, ...]


def normalize_number(token: str) -> str | None:
    """Return a canonical form of ``token`` so cosmetic variants compare equal.

    Strips a leading ``+`` and any grouping commas, parses the value as a float,
    and renders it with a fixed precision so ``0.5`` / ``0.50`` / ``+0.500`` all
    canonicalize to the same string. ``-0`` folds to ``0``. Returns ``None`` when
    the token is not a parseable number.
    """
    cleaned = token.strip().replace(",", "")
    if cleaned in ("", "+", "-", "."):
        return None
    try:
        value = float(cleaned)
    except ValueError:
        return None
    if value == 0:  # fold -0.0 and 0.0 together
        value = 0.0
    # Fixed-precision canonical form; enough to separate the 3-decimal rates the
    # Results_Table emits while treating trailing zeros as insignificant.
    return f"{value:.6f}"


def extract_numbers(text: str) -> list[str]:
    """Return every numeric token in ``text`` in order (raw, un-normalized)."""
    return NUMBER_RE.findall(text or "")


def _normalized_set(texts: Iterable[str]) -> set[str]:
    """Collect the normalized numbers appearing across ``texts``."""
    found: set[str] = set()
    for text in texts:
        for token in extract_numbers(text):
            norm = normalize_number(token)
            if norm is not None:
                found.add(norm)
    return found


def check_text(text: str, generated_sources: Sequence[str]) -> TraceResult:
    """Check that every number in ``text`` traces to a generated source.

    ``generated_sources`` is the text content of the generated files that are
    allowed to originate numbers (the Results_Table, the Results_Store, and any
    other script-generated file). A number in ``text`` is traceable iff its
    normalized form appears among the numbers of some generated source.
    """
    traceable = _normalized_set(generated_sources)

    untraceable: list[str] = []
    seen: set[str] = set()
    for token in extract_numbers(text):
        norm = normalize_number(token)
        if norm is None:
            continue
        if norm not in traceable and norm not in seen:
            untraceable.append(token)
            seen.add(norm)

    return TraceResult(ok=not untraceable, untraceable=tuple(untraceable))


def _read_text(path: str) -> str:
    """Read a file as UTF-8 text (missing file => empty string)."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return fh.read()
    except FileNotFoundError:
        return ""


def check_files(doc_path: str, source_paths: Sequence[str]) -> TraceResult:
    """Check a README/Report file against generated source files on disk.

    Convenience wrapper over :func:`check_text`: reads ``doc_path`` and each of
    ``source_paths`` and runs the traceability check. A missing generated source
    contributes no traceable numbers (so a number relying on it stays untraceable
    and the check fails loudly rather than passing by accident).
    """
    doc_text = _read_text(doc_path)
    sources = [_read_text(p) for p in source_paths]
    return check_text(doc_text, sources)
