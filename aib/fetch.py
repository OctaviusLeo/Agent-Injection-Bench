"""Download the pinned model files into ~/aib-models and verify them (task 1.3; Req 0.5, 0.6, 1.8).

Each file is fetched with huggingface_hub at the revision pinned in config/models.yaml, its size is
checked against the config, its SHA-256 is computed (and compared with the LFS object id when the
config records one), and the result is written to a stamped results/m0_models.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from aib.config import ModelFile, ModelsConfig, load_models
from aib.repro import REPO_ROOT, make_stamp, require_clean

CLASSIFIER_KEY = "classifier"
DEFAULT_KEYS = ("qwen3-4b", "llama3.1-8b", CLASSIFIER_KEY)


@dataclass(frozen=True)
class FetchItem:
    key: str
    repo: str
    revision: str
    file: ModelFile
    dest_dir: Path


def models_dir() -> Path:
    return Path(os.environ.get("AIB_MODELS", str(Path.home() / "aib-models")))


def classifier_dir(base: Path | None = None) -> Path:
    return (base or models_dir()) / CLASSIFIER_KEY


def model_file_path(key: str, filename: str, base: Path | None = None) -> Path:
    return (base or models_dir()) / key / filename


def plan(cfg: ModelsConfig, keys: tuple[str, ...], base: Path) -> list[FetchItem]:
    items = []
    for key in keys:
        if key == CLASSIFIER_KEY:
            c = cfg.classifier
            items += [FetchItem(key, c.repo, c.revision, f, base / CLASSIFIER_KEY) for f in c.files]
            continue
        if key not in cfg.models:
            raise KeyError(f"unknown model key {key!r}")
        m = cfg.models[key]
        if m.provider != "llamacpp":
            raise ValueError(f"{key} is a hosted model; nothing to download")
        items += [FetchItem(key, m.repo, m.revision, f, base / key) for f in m.files]
    return items


def sha256_file(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def _hf_download(repo: str, filename: str, revision: str, local_dir: Path) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=repo, filename=filename, revision=revision, local_dir=str(local_dir))


def fetch_item(item: FetchItem, download: Callable[..., str] = _hf_download) -> dict:
    item.dest_dir.mkdir(parents=True, exist_ok=True)
    path = Path(download(item.repo, item.file.name, item.revision, item.dest_dir))
    size = path.stat().st_size
    digest = sha256_file(path)
    return {
        "name": item.file.name,
        "path": f"~/aib-models/{item.dest_dir.name}/{item.file.name}",
        "size": size,
        "expected_size": item.file.size,
        "size_ok": size == item.file.size,
        "sha256": digest,
        "expected_sha256": item.file.sha256,
        "sha256_ok": item.file.sha256 is None or digest == item.file.sha256,
    }


def fetch(cfg: ModelsConfig, keys: tuple[str, ...], base: Path, download: Callable[..., str] = _hf_download) -> dict:
    entries: dict[str, dict] = {}
    for item in plan(cfg, keys, base):
        entry = entries.setdefault(item.key, {"repo": item.repo, "revision": item.revision, "files": []})
        result = fetch_item(item, download)
        entry["files"].append(result)
        status = "ok" if result["size_ok"] and result["sha256_ok"] else "MISMATCH"
        print(f"{item.key}: {item.file.name} {result['size']} bytes sha256 {result['sha256'][:12]}... {status}", flush=True)
    return entries


def main(argv: list[str] | None = None, download: Callable[..., str] = _hf_download, root: Path = REPO_ROOT) -> int:
    parser = argparse.ArgumentParser(description="Download and verify the pinned model files.")
    parser.add_argument("--model", help="fetch one entry (a model key or 'classifier'); default: the two local models and the classifier")
    parser.add_argument("--out", default="results/m0_models.json")
    args = parser.parse_args(argv)
    keys = (args.model,) if args.model else DEFAULT_KEYS
    require_clean(cwd=root)
    command = "python scripts/fetch_models.py" + (f" --model {args.model}" if args.model else "")
    stamp = make_stamp(command, 0, cwd=root)
    entries = fetch(load_models(root / "config" / "models.yaml"), keys, models_dir(), download)
    out = root / args.out
    previous = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    merged = {**previous.get("entries", {}), **entries}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"stamp": stamp.to_dict(), "models_dir": "~/aib-models", "entries": merged}, indent=2) + "\n",
                   encoding="utf-8")
    bad = [f["name"] for e in entries.values() for f in e["files"] if not (f["size_ok"] and f["sha256_ok"])]
    if bad:
        print(f"size or SHA-256 mismatch: {bad}")
        return 1
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
