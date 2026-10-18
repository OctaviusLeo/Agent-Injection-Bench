"""Start the model server for a model with an optional chat-template override, without changing config/models.yaml.

Used only by scripts/multi_tool_probes.sh to compare candidate setups for the 8B slot (design.md fallback order:
the GGUF's own template, llama.cpp's bundled template from the same build tag, then qwen2.5-7b). The template path
is relative to ~/aib-models (fetched there from the llama.cpp repository at tag b11580).

Usage: python -m aib.server_variant <model key> [template path]
"""

from __future__ import annotations

import json
import sys

from aib import server as S
from aib.config import ModelsConfig, load_models


def variant_config(cfg: ModelsConfig, key: str, template: str | None) -> ModelsConfig:
    m = cfg.models[key].model_copy(update={"chat_template_file": template or None, "enabled": True})
    return cfg.model_copy(update={"models": {**cfg.models, key: m}})


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        print("usage: python -m aib.server_variant <model key> [template path]", file=sys.stderr)
        return 2
    cfg = variant_config(load_models(), argv[0], argv[1] if len(argv) > 1 else None)
    S.compose_down(cfg)
    print(json.dumps({"model": argv[0], "chat_template_file": cfg.models[argv[0]].chat_template_file, **S.up(argv[0], cfg)}))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
