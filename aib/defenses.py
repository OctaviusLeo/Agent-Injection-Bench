"""Defense_Conditions and the pipelines that apply them (tasks 3.4 and 4.1; Req 4).

Every pipeline is AgentDojo's shape - [SystemMessage, InitQuery, LLM, ToolsExecutionLoop([executor,
(detector), LLM], max_iters=15)] - and only the chosen mitigation differs (Req 4.2). The recording
executor stores each new Tool_Output's raw text, before any defense changes it, in
`extra_args["aib_tool_outputs"]` (exposure and the filter's ground truth come from it).
"""

from __future__ import annotations

from collections.abc import Sequence

from agentdojo.agent_pipeline.agent_pipeline import AgentPipeline, load_system_message
from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.agent_pipeline.basic_elements import InitQuery, SystemMessage
from agentdojo.agent_pipeline.tool_execution import ToolsExecutionLoop, ToolsExecutor, tool_result_to_str
from agentdojo.functions_runtime import EmptyEnv, Env, FunctionsRuntime
from agentdojo.types import ChatMessage, get_text_content_as_str

DEFENSES = ("none",)
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


def build_pipeline(defense: str, llm: BasePipelineElement, cfg=None, classifier=None,
                   max_iters: int = MAX_ITERS) -> AgentPipeline:
    """The pipeline for one Defense_Condition. Task 3.4 provides `none`; task 4.1 adds the rest."""
    if defense != "none":
        raise ValueError(f"unknown or not yet implemented defense {defense!r}")
    system = SystemMessage(load_system_message(None))
    loop = ToolsExecutionLoop([RecordingToolsExecutor(), llm], max_iters=max_iters)
    pipeline = AgentPipeline([system, InitQuery(), llm, loop])
    pipeline.name = f"{llm.name}-{defense}"
    return pipeline
