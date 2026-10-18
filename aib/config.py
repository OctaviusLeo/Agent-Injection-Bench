"""Configuration loader: pydantic models that validate every value in config/*.yaml on load (Req 0.2, 4.5).

Every load error names the file and the offending field. Known ids (models, defenses, conditions,
goals) are fixed here, so a typo in a YAML file fails loudly instead of silently changing the grid.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "config"

KNOWN_MODELS = ("qwen3-4b", "llama3.1-8b", "qwen2.5-7b", "gpt-6-luna")
KNOWN_DEFENSES = ("none", "delimit", "hierarchy", "filter")
KNOWN_STYLES = ("direct", "fake_system", "ignore_previous", "important_instructions", "hidden_field")
KNOWN_CONDITIONS = ("clean", *KNOWN_STYLES)
KNOWN_GOALS = tuple(f"injection_task_{i}" for i in range(6))
KNOWN_HOSTED_PARAMS = ("temperature", "top_p", "seed")
_USER_TASK = re.compile(r"\Auser_task_\d+\Z")


class ConfigError(ValueError):
    """A configuration file is missing, unreadable, or holds an invalid value (names the field)."""


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _check_ids(values: list[str], known: tuple[str, ...], what: str) -> list[str]:
    unknown = [v for v in values if v not in known]
    if unknown:
        raise ValueError(f"unknown {what} {unknown}; known: {list(known)}")
    if len(set(values)) != len(values):
        raise ValueError(f"duplicate {what} in {values}")
    return values


# ---------------------------------------------------------------- experiment.yaml
class LocalExtra(_Strict):
    top_k: int = Field(ge=0)
    min_p: float = Field(ge=0.0, le=1.0)


class Sampling(_Strict):
    temperature: float = Field(ge=0.0, le=2.0)
    top_p: float = Field(gt=0.0, le=1.0)
    local_extra: LocalExtra


class Pilot(_Strict):
    user_tasks: list[str] = Field(min_length=1)
    repetitions: int = Field(ge=1)


class ExperimentConfig(_Strict):
    base_seed: int = Field(ge=0)
    repetitions: int = Field(ge=3)
    suite: Literal["workspace"]
    benchmark_version: tuple[int, int, int]
    register_v1_1_2: bool
    user_tasks: list[str] = Field(min_length=40)
    goals: list[str]
    goal_assignment: Literal["mod6"]
    conditions: list[str]
    defenses: list[str]
    max_iters: int = Field(ge=1)
    max_tokens: int = Field(ge=1)
    sampling: Sampling
    local_order: list[Literal["defense", "condition", "task", "rep"]]
    hosted_order: list[Literal["rep", "defense", "condition", "task"]]
    pilot: Pilot

    @field_validator("user_tasks")
    @classmethod
    def _user_tasks(cls, v: list[str]) -> list[str]:
        bad = [t for t in v if not _USER_TASK.match(t)]
        if bad:
            raise ValueError(f"user task ids must look like user_task_<n>: {bad}")
        if len(set(v)) != len(v):
            raise ValueError("duplicate user task ids")
        return v

    @field_validator("goals")
    @classmethod
    def _goals(cls, v: list[str]) -> list[str]:
        return _check_ids(v, KNOWN_GOALS, "goal")

    @field_validator("conditions")
    @classmethod
    def _conditions(cls, v: list[str]) -> list[str]:
        _check_ids(v, KNOWN_CONDITIONS, "condition")
        if "clean" not in v:
            raise ValueError("conditions must include the clean control")
        return v

    @field_validator("defenses")
    @classmethod
    def _defenses(cls, v: list[str]) -> list[str]:
        _check_ids(v, KNOWN_DEFENSES, "defense")
        if "none" not in v:
            raise ValueError("defenses must include the none control")
        return v

    @field_validator("local_order", "hosted_order")
    @classmethod
    def _order(cls, v: list[str]) -> list[str]:
        if sorted(v) != sorted({"defense", "condition", "task", "rep"}) or len(v) != 4:
            raise ValueError("order must list defense, condition, task, and rep exactly once")
        return v

    @model_validator(mode="after")
    def _pilot_tasks(self) -> "ExperimentConfig":
        missing = [t for t in self.pilot.user_tasks if t not in self.user_tasks]
        if missing:
            raise ValueError(f"pilot.user_tasks not in user_tasks: {missing}")
        return self

    @property
    def styles(self) -> list[str]:
        return [c for c in self.conditions if c != "clean"]


# ---------------------------------------------------------------- models.yaml
class ModelFile(_Strict):
    name: str
    size: int = Field(gt=0)
    sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")


class ServerConfig(_Strict):
    image: str
    compose_project: str
    service: str
    host_port: int = Field(ge=1, le=65535)
    base_url: str
    flags: list[str]
    min_ctx_per_slot: int = Field(ge=1)
    health_timeout_s: float = Field(gt=0)


class ModelEntry(_Strict):
    provider: Literal["llamacpp", "openai"]
    enabled: bool
    name: str
    license: str
    workers: int = Field(ge=1)
    timeout_s: float = Field(gt=0)
    repo: str | None = None
    revision: str | None = Field(default=None, pattern=r"^[0-9a-f]{40}$")
    files: list[ModelFile] = Field(default_factory=list)
    quantization: str | None = None
    base_model: str | None = None
    ctx_size: int | None = Field(default=None, ge=1)
    parallel: int | None = Field(default=None, ge=1)
    chat_template_file: str | None = None
    base_url: str | None = None
    api_key_env: str | None = None

    @model_validator(mode="after")
    def _provider_fields(self) -> "ModelEntry":
        if self.provider == "llamacpp":
            missing = [f for f in ("repo", "revision", "quantization", "ctx_size", "parallel") if getattr(self, f) is None]
            if missing or not self.files:
                raise ValueError(f"llamacpp model needs {missing + ([] if self.files else ['files'])}")
            if self.workers != self.parallel:
                raise ValueError("workers must equal parallel (one worker per server slot)")
        else:
            if not self.base_url or not self.api_key_env:
                raise ValueError("openai model needs base_url and api_key_env")
        return self

    @property
    def ctx_per_slot(self) -> int | None:
        if self.ctx_size is None or self.parallel is None:
            return None
        return self.ctx_size // self.parallel


class ClassifierEntry(_Strict):
    repo: str
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    license: str
    base_model: str
    files: list[ModelFile] = Field(min_length=1)


class ModelsConfig(_Strict):
    server: ServerConfig
    models: dict[str, ModelEntry]
    classifier: ClassifierEntry

    @field_validator("models")
    @classmethod
    def _keys(cls, v: dict[str, ModelEntry]) -> dict[str, ModelEntry]:
        _check_ids(list(v), KNOWN_MODELS, "model")
        return v

    @model_validator(mode="after")
    def _slots(self) -> "ModelsConfig":
        for key, m in self.models.items():
            if m.provider == "llamacpp" and (m.ctx_per_slot or 0) < self.server.min_ctx_per_slot:
                raise ValueError(f"models.{key}: ctx_size / parallel is below {self.server.min_ctx_per_slot} tokens per slot")
        return self

    def local_models(self) -> list[str]:
        return [k for k, m in self.models.items() if m.provider == "llamacpp" and m.enabled]


# ---------------------------------------------------------------- hosted.yaml
class Prices(_Strict):
    input: float = Field(ge=0)
    cached_input: float = Field(ge=0)
    cache_write: float = Field(ge=0)
    output: float = Field(ge=0)


class HostedConfig(_Strict):
    model: str
    spend_cap_usd: float = Field(ge=0)
    prices_per_million_usd: Prices
    worst_case_chars_per_token: float = Field(gt=0)
    reasoning_effort: Literal["none", "minimal", "low", "medium", "high"]
    max_completion_tokens: int = Field(ge=1)
    conditions: list[str]
    defenses: list[str]
    workers: int = Field(ge=1)
    ledger: str
    accepted_params: list[str] | None = None

    @field_validator("model")
    @classmethod
    def _model(cls, v: str) -> str:
        _check_ids([v], KNOWN_MODELS, "model")
        return v

    @field_validator("conditions")
    @classmethod
    def _conditions(cls, v: list[str]) -> list[str]:
        return _check_ids(v, KNOWN_CONDITIONS, "condition")

    @field_validator("defenses")
    @classmethod
    def _defenses(cls, v: list[str]) -> list[str]:
        return _check_ids(v, KNOWN_DEFENSES, "defense")

    @field_validator("accepted_params")
    @classmethod
    def _params(cls, v: list[str] | None) -> list[str] | None:
        return None if v is None else _check_ids(v, KNOWN_HOSTED_PARAMS, "hosted parameter")


# ---------------------------------------------------------------- defenses.yaml
class DefenseConfig(_Strict):
    defenses: list[str]
    delimit_open: str = Field(min_length=1)
    delimit_close: str = Field(min_length=1)
    delimit_text: str = Field(min_length=1)
    hierarchy_text: str = Field(min_length=1)
    classifier_repo: str
    classifier_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    injection_label: str
    threshold: float = Field(ge=0.0, le=1.0)
    window_tokens: int = Field(ge=8)
    window_overlap_tokens: int = Field(ge=0)
    replacement_text: str = Field(min_length=1)

    @field_validator("defenses")
    @classmethod
    def _defenses(cls, v: list[str]) -> list[str]:
        return _check_ids(v, KNOWN_DEFENSES, "defense")

    @model_validator(mode="after")
    def _windows(self) -> "DefenseConfig":
        if self.window_overlap_tokens >= self.window_tokens:
            raise ValueError("window_overlap_tokens must be smaller than window_tokens")
        return self


# ---------------------------------------------------------------- loading
def _load(model: type[BaseModel], path: Path):
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigError(f"{path}: cannot read: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path}: invalid YAML: {exc}") from exc
    try:
        return model.model_validate(data)
    except ValidationError as exc:
        parts = []
        for err in exc.errors():
            loc = ".".join(str(p) for p in err["loc"]) or "<root>"
            parts.append(f"{loc}: {err['msg']}")
        raise ConfigError(f"{path}: " + "; ".join(parts)) from exc


def load_experiment(path: Path | str | None = None) -> ExperimentConfig:
    return _load(ExperimentConfig, Path(path) if path else CONFIG_DIR / "experiment.yaml")


def load_models(path: Path | str | None = None) -> ModelsConfig:
    return _load(ModelsConfig, Path(path) if path else CONFIG_DIR / "models.yaml")


def load_hosted(path: Path | str | None = None) -> HostedConfig:
    return _load(HostedConfig, Path(path) if path else CONFIG_DIR / "hosted.yaml")


def load_defenses(path: Path | str | None = None) -> DefenseConfig:
    return _load(DefenseConfig, Path(path) if path else CONFIG_DIR / "defenses.yaml")


class AllConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    experiment: ExperimentConfig
    models: ModelsConfig
    hosted: HostedConfig
    defenses: DefenseConfig


def load_all(config_dir: Path | str | None = None) -> AllConfig:
    """Load the four files and check that they agree with each other."""
    d = Path(config_dir) if config_dir else CONFIG_DIR
    cfg = AllConfig(
        experiment=load_experiment(d / "experiment.yaml"),
        models=load_models(d / "models.yaml"),
        hosted=load_hosted(d / "hosted.yaml"),
        defenses=load_defenses(d / "defenses.yaml"),
    )
    problems = []
    hosted_model = cfg.models.models.get(cfg.hosted.model)
    if hosted_model is None or hosted_model.provider != "openai":
        problems.append(f"hosted.model: {cfg.hosted.model} is not an openai model in models.yaml")
    if not set(cfg.hosted.conditions) <= set(cfg.experiment.conditions):
        problems.append("hosted.conditions: not a subset of experiment.conditions")
    if not set(cfg.hosted.defenses) <= set(cfg.experiment.defenses):
        problems.append("hosted.defenses: not a subset of experiment.defenses")
    if cfg.defenses.defenses != cfg.experiment.defenses:
        problems.append("defenses.defenses: differs from experiment.defenses")
    if (cfg.defenses.classifier_repo, cfg.defenses.classifier_revision) != (
        cfg.models.classifier.repo,
        cfg.models.classifier.revision,
    ):
        problems.append("defenses.classifier_repo/classifier_revision: differ from models.classifier")
    if problems:
        raise ConfigError(f"{d}: " + "; ".join(problems))
    return cfg
