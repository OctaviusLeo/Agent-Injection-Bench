"""Defense_Conditions and the pipelines that apply them (tasks 3.4 and 4.1; Req 4).

Every pipeline is AgentDojo's shape - [SystemMessage, InitQuery, LLM, ToolsExecutionLoop([executor,
(detector), LLM], max_iters=15)] - and only the chosen mitigation differs (Req 4.2):
- `none`: nothing (Req 4.3);
- `delimit`: AgentDojo's spotlighting with delimiting: each Tool_Output wrapped in << >> and its sentence
  added to the system message;
- `hierarchy`: this project's instruction-hierarchy text added to the system message;
- `filter`: every Tool_Output screened by the classifier; flagged outputs replaced by AgentDojo's
  "<Data omitted because a prompt injection was detected>", with one decision recorded per output.
The texts and classifier settings come from config/defenses.yaml (Req 4.5). A tool-permission-limit
defense is not implemented (design.md "Defense_Conditions").

The recording executor stores each new Tool_Output's raw text, before any defense changes it, in
`extra_args["aib_tool_outputs"]`; exposure and the filter's ground truth come from it.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import asdict, dataclass

from agentdojo.agent_pipeline.agent_pipeline import AgentPipeline, load_system_message
from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.agent_pipeline.basic_elements import InitQuery, SystemMessage
from agentdojo.agent_pipeline.pi_detector import PromptInjectionDetector
from agentdojo.agent_pipeline.tool_execution import ToolsExecutionLoop, ToolsExecutor, tool_result_to_str
from agentdojo.functions_runtime import EmptyEnv, Env, FunctionsRuntime
from agentdojo.types import ChatMessage, get_text_content_as_str

from aib.attacks import contains_injection
from aib.config import DefenseConfig, load_defenses

DEFENSES = ("none", "delimit", "hierarchy", "filter")
MAX_ITERS = 15


class RecordingToolsExecutor(ToolsExecutor):
    """AgentDojo's ToolsExecutor with optional delimiters, recording each new output's raw text."""

    def __init__(self, open_mark: str = "", close_mark: str = "") -> None:
        self.open_mark, self.close_mark = open_mark, close_mark
        super().__init__(tool_output_formatter=self._format)

    def _format(self, result) -> str:
        return f"{self.open_mark}{tool_result_to_str(result)}{self.close_mark}"

    def _raw(self, text: str) -> str:
        if self.open_mark and text.startswith(self.open_mark) and text.endswith(self.close_mark):
            return text[len(self.open_mark): len(text) - len(self.close_mark)]
        return text

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = [],
        extra_args: dict = {},
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict]:
        n_before = len(messages)
        query, runtime, env, messages, extra_args = super().query(query, runtime, env, messages, extra_args)
        new = [m for m in messages[n_before:] if m["role"] == "tool"]
        if not new:
            return query, runtime, env, messages, extra_args
        recorded = list(extra_args.get("aib_tool_outputs", []))
        for m in new:
            recorded.append({
                "tool": m["tool_call"].function,
                "call_id": m["tool_call_id"],
                "raw": self._raw(get_text_content_as_str(m["content"])),
                "error": m["error"],
            })
        return query, runtime, env, messages, {**extra_args, "aib_tool_outputs": recorded}


@dataclass(frozen=True)
class FilterDecision:
    output_sha256: str
    injected: bool  # ground truth by the exposure rule
    score: float  # highest INJECTION probability over windows
    flagged: bool
    tool: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class RecordingPIDetector(PromptInjectionDetector):
    """AgentDojo's detector in "message" mode with this project's classifier, keeping a FilterDecision for
    every screened Tool_Output in `extra_args["aib_filter"]` (Req 4.4). Replacement uses AgentDojo's
    `transform` (the "<Data omitted ...>" text)."""

    def __init__(self, classifier, threshold: float) -> None:
        super().__init__(mode="message", raise_on_injection=False)
        self.classifier = classifier
        self.threshold = threshold

    def detect(self, tool_output: str) -> tuple[bool, float]:
        score = float(self.classifier.score(tool_output))
        return score >= self.threshold, score

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = [],
        extra_args: dict = {},
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict]:
        if not messages or messages[-1]["role"] != "tool":
            return query, runtime, env, messages, extra_args
        trailing = []
        for m in reversed(messages):
            if m["role"] != "tool":
                break
            trailing.append(m)
        injections = extra_args.get("aib_injections") or {}
        decisions = list(extra_args.get("aib_filter", []))
        for m in reversed(trailing):  # screen in the order the outputs arrived
            text = get_text_content_as_str(m["content"]) or ""
            flagged, score = self.detect(text)
            decisions.append(FilterDecision(hashlib.sha256(text.encode("utf-8")).hexdigest(),
                                            contains_injection(text, injections), score, flagged,
                                            m["tool_call"].function).to_dict())
            if flagged:
                m["content"] = self.transform(m["content"] or [])
        return query, runtime, env, messages, {**extra_args, "aib_filter": decisions}


def system_message(defense: str, cfg: DefenseConfig) -> str:
    base = load_system_message(None)
    if defense == "delimit":
        return f"{base} {cfg.delimit_text}"
    if defense == "hierarchy":
        return f"{base} {cfg.hierarchy_text}"
    return base


def build_pipeline(defense: str, llm: BasePipelineElement, cfg: DefenseConfig | None = None, classifier=None,
                   max_iters: int = MAX_ITERS) -> AgentPipeline:
    """[SystemMessage(text), InitQuery(), llm, ToolsExecutionLoop([RecordingToolsExecutor, (RecordingPIDetector),
    llm], max_iters)]; only the chosen mitigation differs (Req 4.2)."""
    if defense not in DEFENSES:
        raise ValueError(f"unknown defense {defense!r}; known: {DEFENSES}")
    cfg = cfg or load_defenses()
    if defense == "delimit":
        executor = RecordingToolsExecutor(cfg.delimit_open, cfg.delimit_close)
    else:
        executor = RecordingToolsExecutor()
    loop_elements: list[BasePipelineElement] = [executor]
    if defense == "filter":
        if classifier is None:
            raise ValueError("the filter defense needs a classifier")
        loop_elements.append(RecordingPIDetector(classifier, cfg.threshold))
    loop_elements.append(llm)
    pipeline = AgentPipeline([SystemMessage(system_message(defense, cfg)), InitQuery(), llm,
                              ToolsExecutionLoop(loop_elements, max_iters=max_iters)])
    pipeline.name = f"{llm.name}-{defense}"
    return pipeline
