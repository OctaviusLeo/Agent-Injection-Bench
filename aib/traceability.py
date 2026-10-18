"""README/Report number traceability (task 6.2; Req 7.4, 9.3, 9.4; Property 11).

Every number a reader sees in README.md or docs/REPORT.md must appear verbatim in a file generated
under results/. Ignored, because they are not reported values: numbers inside words (`qwen3-4b`,
`M0`), dates (`2026-10-17`), version strings (`0.1.35`, `v7.0.1`), URLs, file paths, fenced code,
inline code, HTML comments, and ordered-list markers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

# A number is a token not glued to a word, a path, a dot, or a hyphenated word on either side.
NUMBER_RE = re.compile(r"(?<![\w./\\-])(-?\d+(?:\.\d+)?)(?![\w/\\]|-\w|\.\d)")
# Any number-looking run in a generated source (permissive: JSON, tables, prose).
SOURCE_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")

_FENCED = re.compile(r"```.*?```", re.S)
_INLINE = re.compile(r"`[^`\n]*`")
_COMMENT = re.compile(r"<!--.*?-->", re.S)
_LINK_TARGET = re.compile(r"\]\([^)]*\)")
_URL = re.compile(r"https?://\S+")
_PATH = re.compile(r"\S*[/\\]\S*")
_LIST_MARKER = re.compile(r"(?m)^\s*\d+\.\s")


@dataclass(frozen=True)
class TraceResult:
    ok: bool
    untraceable: tuple[str, ...]  # each untraceable number once, in first-seen order


def strip_ignored(text: str) -> str:
    for pattern in (_FENCED, _COMMENT, _INLINE, _LINK_TARGET, _URL, _PATH, _LIST_MARKER):
        text = pattern.sub(" ", text)
    return text


def extract_numbers(text: str) -> list[str]:
    """The numbers a reader would take as reported values, in order."""
    return NUMBER_RE.findall(strip_ignored(text))


def source_numbers(texts) -> set[str]:
    found: set[str] = set()
    for text in texts:
        for token in SOURCE_NUMBER_RE.findall(text):
            found.add(token)
            found.add(token.lstrip("-"))
    return found


def check_text(text: str, generated_sources) -> TraceResult:
    """A number traces when the same token appears verbatim in one of the generated sources."""
    known = source_numbers(generated_sources)
    missing: list[str] = []
    for token in extract_numbers(text):
        if token not in known and token not in missing:
            missing.append(token)
    return TraceResult(ok=not missing, untraceable=tuple(missing))


def generated_sources(results_dir: Path) -> list[str]:
    """The text of every file generated under results/ (JSON, JSONL, Markdown)."""
    texts = []
    for path in sorted(Path(results_dir).rglob("*")):
        if path.is_file() and path.suffix in (".json", ".jsonl", ".md", ".txt", ".csv"):
            texts.append(path.read_text(encoding="utf-8", errors="replace"))
    return texts


def check_files(docs: list[Path], results_dir: Path) -> dict[str, TraceResult | None]:
    """{doc path: TraceResult, or None when the doc does not exist}."""
    sources = generated_sources(results_dir)
    out: dict[str, TraceResult | None] = {}
    for doc in docs:
        doc = Path(doc)
        out[str(doc)] = check_text(doc.read_text(encoding="utf-8"), sources) if doc.exists() else None
    return out
