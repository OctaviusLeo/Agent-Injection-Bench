"""Unit tests for aib.config: every file loads, and bad values are rejected naming the field."""

from __future__ import annotations

import shutil

import pytest
import yaml

from aib import config as C


@pytest.fixture
def cfg_dir(tmp_path):
    d = tmp_path / "config"
    shutil.copytree(C.CONFIG_DIR, d)
    return d


def _edit(path, fn):
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    fn(data)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def test_repository_files_load_and_agree():
    cfg = C.load_all()
    assert cfg.experiment.repetitions == 3
    assert len(cfg.experiment.user_tasks) == 40
    assert cfg.experiment.styles == ["direct", "fake_system", "ignore_previous", "important_instructions", "hidden_field"]
    assert cfg.hosted.spend_cap_usd == 3.00
    assert cfg.models.local_models() == ["qwen3-4b", "qwen2.5-7b"]  # qwen2.5-7b replaced llama3.1-8b (design.md M0 outcome)
    assert cfg.models.models["llama3.1-8b"].enabled is False
    assert cfg.models.models["qwen3-4b"].ctx_per_slot == 32768
    assert cfg.defenses.threshold == 0.5
    assert cfg.models.models["gpt-6-luna"].ctx_per_slot is None


@pytest.mark.parametrize(
    "file,edit,field",
    [
        ("defenses.yaml", lambda d: d["defenses"].append("tool_limit"), "defenses"),
        ("experiment.yaml", lambda d: d["defenses"].append("bogus"), "defenses"),
        ("experiment.yaml", lambda d: d["conditions"].append("rot13"), "conditions"),
        ("hosted.yaml", lambda d: d["conditions"].append("rot13"), "conditions"),
        ("experiment.yaml", lambda d: d["goals"].append("injection_task_9"), "goals"),
        ("models.yaml", lambda d: d["models"].__setitem__("gpt-9", d["models"]["gpt-6-luna"]), "models"),
        ("experiment.yaml", lambda d: d.__setitem__("repetitions", 2), "repetitions"),
        ("hosted.yaml", lambda d: d.__setitem__("spend_cap_usd", -1.0), "spend_cap_usd"),
        ("defenses.yaml", lambda d: d.__setitem__("threshold", 1.5), "threshold"),
        ("defenses.yaml", lambda d: d.__setitem__("threshold", -0.1), "threshold"),
        ("experiment.yaml", lambda d: d["sampling"].__setitem__("top_p", 0.0), "sampling.top_p"),
        ("hosted.yaml", lambda d: d.__setitem__("accepted_params", ["logprobs"]), "accepted_params"),
        ("experiment.yaml", lambda d: d.__setitem__("unknown_key", 1), "unknown_key"),
    ],
)
def test_invalid_values_are_rejected_naming_the_field(cfg_dir, file, edit, field):
    _edit(cfg_dir / file, edit)
    loader = {
        "experiment.yaml": C.load_experiment,
        "models.yaml": C.load_models,
        "hosted.yaml": C.load_hosted,
        "defenses.yaml": C.load_defenses,
    }[file]
    with pytest.raises(C.ConfigError) as info:
        loader(cfg_dir / file)
    assert field in str(info.value)
    assert file in str(info.value)


def test_slot_context_below_minimum_is_rejected(cfg_dir):
    _edit(cfg_dir / "models.yaml", lambda d: d["models"]["qwen3-4b"].__setitem__("ctx_size", 32768))
    with pytest.raises(C.ConfigError, match="qwen3-4b"):
        C.load_models(cfg_dir / "models.yaml")


def test_workers_must_equal_slots(cfg_dir):
    _edit(cfg_dir / "models.yaml", lambda d: d["models"]["llama3.1-8b"].__setitem__("workers", 2))
    with pytest.raises(C.ConfigError, match="workers"):
        C.load_models(cfg_dir / "models.yaml")


def test_llamacpp_model_needs_files(cfg_dir):
    _edit(cfg_dir / "models.yaml", lambda d: d["models"]["qwen3-4b"].__setitem__("files", []))
    with pytest.raises(C.ConfigError, match="files"):
        C.load_models(cfg_dir / "models.yaml")


def test_openai_model_needs_key_env(cfg_dir):
    _edit(cfg_dir / "models.yaml", lambda d: d["models"]["gpt-6-luna"].pop("api_key_env"))
    with pytest.raises(C.ConfigError, match="api_key_env"):
        C.load_models(cfg_dir / "models.yaml")


def test_cross_file_disagreement_is_rejected(cfg_dir):
    _edit(cfg_dir / "defenses.yaml", lambda d: d.__setitem__("classifier_revision", "0" * 40))
    with pytest.raises(C.ConfigError, match="classifier"):
        C.load_all(cfg_dir)


def test_order_and_controls_are_required(cfg_dir):
    _edit(cfg_dir / "experiment.yaml", lambda d: d.__setitem__("local_order", ["defense", "task", "rep"]))
    with pytest.raises(C.ConfigError, match="local_order"):
        C.load_experiment(cfg_dir / "experiment.yaml")
    _edit(cfg_dir / "experiment.yaml", lambda d: (d.__setitem__("local_order", ["defense", "condition", "task", "rep"]), d["conditions"].remove("clean")))
    with pytest.raises(C.ConfigError, match="clean"):
        C.load_experiment(cfg_dir / "experiment.yaml")


def test_unreadable_and_invalid_yaml(tmp_path):
    with pytest.raises(C.ConfigError, match="cannot read"):
        C.load_hosted(tmp_path / "missing.yaml")
    bad = tmp_path / "bad.yaml"
    bad.write_text("a: [1, 2\n", encoding="utf-8")
    with pytest.raises(C.ConfigError, match="invalid YAML"):
        C.load_hosted(bad)


def test_pilot_tasks_must_be_in_the_grid(cfg_dir):
    _edit(cfg_dir / "experiment.yaml", lambda d: d["pilot"].__setitem__("user_tasks", ["user_task_99"]))
    with pytest.raises(C.ConfigError, match="pilot"):
        C.load_experiment(cfg_dir / "experiment.yaml")


def test_claude_file_loads_and_agrees_with_models():
    cfg = C.load_all()
    assert cfg.claude.model == "claude-haiku-5-5" and cfg.claude.run_cap_usd == 5.00
    assert cfg.models.models["claude-haiku-5-5"].provider == "anthropic"
    assert cfg.models.local_models() == ["qwen3-4b", "qwen2.5-7b"]  # the Claude entry is not a local model
    haiku = cfg.claude.prices_per_million_usd["claude-haiku-5-5"]
    assert haiku.long_context_tokens == 100_000 and haiku.long_context.output == 2.50
    assert cfg.claude.judge.model == "claude-sonnet-5-5" and cfg.claude.judge.model in cfg.claude.prices_per_million_usd


@pytest.mark.parametrize(
    "edit,field",
    [
        (lambda d: d.__setitem__("run_cap_usd", -1), "run_cap_usd"),
        (lambda d: d.__setitem__("conditions", ["clean", "nope"]), "conditions"),
        (lambda d: d.__setitem__("effort", "extreme"), "effort"),
        (lambda d: d["prices_per_million_usd"].pop("claude-sonnet-5-5"), "prices_per_million_usd"),
        (lambda d: d["prices_per_million_usd"]["claude-haiku-5-5"].pop("long_context"), "long_context"),
    ],
)
def test_invalid_claude_values_are_rejected_naming_the_field(cfg_dir, edit, field):
    _edit(cfg_dir / "claude.yaml", edit)
    with pytest.raises(C.ConfigError, match=field):
        C.load_claude(cfg_dir / "claude.yaml")


def test_anthropic_model_needs_key_env(cfg_dir):
    _edit(cfg_dir / "models.yaml", lambda d: d["models"]["claude-haiku-5-5"].pop("api_key_env"))
    with pytest.raises(C.ConfigError, match="anthropic model needs api_key_env"):
        C.load_models(cfg_dir / "models.yaml")


def test_claude_cross_file_checks(cfg_dir):
    def openai_target(d):
        d["model"] = "gpt-6-luna"
        d["prices_per_million_usd"]["gpt-6-luna"] = d["prices_per_million_usd"].pop("claude-haiku-5-5")
    _edit(cfg_dir / "claude.yaml", openai_target)
    _edit(cfg_dir / "experiment.yaml", lambda d: d["conditions"].remove("important_instructions"))
    with pytest.raises(C.ConfigError, match="claude.model: gpt-6-luna is not an anthropic model") as err:
        C.load_all(cfg_dir)
    assert "claude.conditions: not a subset" in str(err.value)
    assert "claude.prices_per_million_usd: no prices for ['claude-haiku-5-5']" in str(err.value)


def test_claude_file_is_needed_only_for_an_enabled_claude_model(cfg_dir):
    (cfg_dir / "claude.yaml").unlink()
    with pytest.raises(C.ConfigError, match="need config/claude.yaml"):
        C.load_all(cfg_dir)
    _edit(cfg_dir / "models.yaml", lambda d: d["models"]["claude-haiku-5-5"].__setitem__("enabled", False))
    assert C.load_all(cfg_dir).claude is None
