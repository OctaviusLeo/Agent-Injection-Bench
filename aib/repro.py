"""Reproducibility helper: every reported number carries its command, seed, and commit (Req 0.1, 0.2).

`commit_hash()` returns the full 40-character hash of HEAD and raises when there is none, so a
result can never be stamped with a placeholder. `require_clean()` refuses to run when the working
tree has changes outside the generated-output paths, so every stamped number comes from committed
code. Both use only the two read-only Git commands the execution environment allows.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

ALLOWED_DIRTY: tuple[str, ...] = (
    "results/",
    "runs/",
    "docs/ENVIRONMENT.md",
    "docs/evidence/",
    "docs/BLOCKERS.md",
    "docs/REPORT.md",
    "README.md",
    ".kiro/",
)

_HEX40 = re.compile(r"\A[0-9a-f]{40}\Z")


class ReproError(RuntimeError):
    """Raised when there is no real commit to stamp or the tree has uncommitted code changes."""


def _git(args: list[str], cwd: Path | str | None) -> str:
    try:
        out = subprocess.run(
            ["git", *args], cwd=str(cwd or REPO_ROOT), capture_output=True, text=True, check=True
        )
    except FileNotFoundError as exc:
        raise ReproError("git is not installed") from exc
    except subprocess.CalledProcessError as exc:
        raise ReproError(f"git {' '.join(args)} failed: {exc.stderr.strip()}") from exc
    return out.stdout


def commit_hash(cwd: Path | str | None = None) -> str:
    """Return `git rev-parse HEAD`; raise ReproError unless it is 40 lowercase hex characters."""
    value = _git(["rev-parse", "HEAD"], cwd).strip()
    if not _HEX40.match(value):
        raise ReproError(f"git rev-parse HEAD did not return a 40-character hash: {value!r}")
    return value


def _porcelain_paths(output: str) -> list[str]:
    paths = []
    for line in output.splitlines():
        if len(line) < 4:
            continue
        path = line[3:]
        if " -> " in path:  # rename: check both sides
            old, new = path.split(" -> ", 1)
            paths.extend([old.strip('"'), new.strip('"')])
        else:
            paths.append(path.strip('"'))
    return paths


def _is_allowed(path: str, allowed: tuple[str, ...]) -> bool:
    return any(path == a or (a.endswith("/") and path.startswith(a)) for a in allowed)


def require_clean(allowed: tuple[str, ...] = ALLOWED_DIRTY, cwd: Path | str | None = None) -> str:
    """Raise ReproError listing every changed path outside `allowed`; otherwise return commit_hash()."""
    status = _git(["--no-optional-locks", "status", "--porcelain"], cwd)
    offending = sorted({p for p in _porcelain_paths(status) if not _is_allowed(p, allowed)})
    if offending:
        raise ReproError("uncommitted changes outside generated paths (commit first): " + ", ".join(offending))
    return commit_hash(cwd)


@dataclass(frozen=True)
class Stamp:
    """Provenance of a Measured_Value: the generating command, its seed, and the full commit hash."""

    command: str
    seed: int
    commit: str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Stamp":
        return cls(command=str(data["command"]), seed=int(data["seed"]), commit=str(data["commit"]))


def make_stamp(command: str, seed: int, cwd: Path | str | None = None) -> Stamp:
    """Stamp `command` and `seed` with the current commit (raises when there is no commit)."""
    return Stamp(command=command, seed=int(seed), commit=commit_hash(cwd))
