"""Unit tests for aib.server_variant."""

from __future__ import annotations

from aib import server_variant as SV
from aib.config import load_models


def test_variant_config_overrides_only_the_template_and_enables_the_model():
    cfg = load_models()
    v = SV.variant_config(cfg, "qwen2.5-7b", "templates/x.jinja")
    assert v.models["qwen2.5-7b"].chat_template_file == "templates/x.jinja" and v.models["qwen2.5-7b"].enabled
    assert cfg.models["qwen2.5-7b"].chat_template_file is None  # the loaded config is not modified
    assert SV.variant_config(cfg, "llama3.1-8b", "").models["llama3.1-8b"].chat_template_file is None


def test_main_starts_the_variant(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(SV.S, "compose_down", lambda cfg: calls.append("down"))
    monkeypatch.setattr(SV.S, "up", lambda key, cfg: calls.append(("up", key, cfg.models[key].chat_template_file)) or {"total_slots": 1})
    assert SV.main(["llama3.1-8b", "templates/t.jinja"]) == 0
    assert calls == ["down", ("up", "llama3.1-8b", "templates/t.jinja")]
    assert '"chat_template_file": "templates/t.jinja"' in capsys.readouterr().out
    assert SV.main([]) == 2
