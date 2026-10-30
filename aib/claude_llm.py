"""Claude chat model element for AgentDojo pipelines, through the Anthropic Messages API (claude-provider Req C1, C2).

`ClaudeLLM` mirrors `aib.llm.ChatLLM`: it converts AgentDojo messages to Messages API content blocks (the
pipeline's system message becomes `system`; the tool outputs of one assistant turn become `tool_result` blocks in
one user turn), sends the tool schemas, parses `tool_use` blocks back into AgentDojo `FunctionCall`s, retries
transient failures, and appends one `ClaudeCallRecord` per request to `extra_args["aib_calls"]`. Every request
goes through the `SpendLedger`.

Current Claude models accept only their default sampling, so no temperature, top_p, top_k or seed is sent.
AgentDojo gets only the text blocks and the tool calls. Claude's own content blocks, thinking blocks and their
signatures included, stay on the assistant message under `aib_claude_content` and go back verbatim on the next
request: AgentDojo's text helper joins every block, so a thinking block in `content` would leak into the final
answer that the task check reads.
"""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass

import anthropic
from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.functions_runtime import EmptyEnv, Env, Function, FunctionCall, FunctionsRuntime
from agentdojo.types import ChatAssistantMessage, ChatMessage, get_text_content_as_str, text_content_block_from_string

from aib.ledger import SpendLedger
from aib.llm import (
    CallRecord,
    ContextOverflowError,
    LLMBadRequestError,
    LLMConnectionError,
    LLMError,
    LLMResponseError,
    LLMServerError,
    LLMTimeoutError,
    MissingSecretError,
)

CLAUDE_CONTENT = "aib_claude_content"  # extra key on an assistant message: Claude's content blocks, in order
_CONTEXT = re.compile(r"prompt is too long|context (window|length)|too many tokens", re.I)
_NO_CREDIT = re.compile(r"credit balance", re.I)


class AccountError(MissingSecretError):
    """The API refused the key (HTTP 401 or 403) or the account has no credit left. Like a missing key, it stops
    the run instead of erroring every remaining case."""

    def __init__(self, env_var: str, detail: str) -> None:
        RuntimeError.__init__(self, f"the Anthropic API refused the key in {env_var}: {detail}")
        self.env_var = env_var


@dataclass
class ClaudeCallRecord(CallRecord):
    """A `CallRecord` plus the Claude-only fields. `seed` is None: the Messages API has no seed."""

    cache_write_tokens: int = 0
    stop_category: str | None = None  # the refusal category when finish_reason is `refusal`


def tool_schema(f: Function) -> dict:
    return {"name": f.name, "description": f.description, "input_schema": f.parameters.model_json_schema()}


def _text(content) -> str:
    return get_text_content_as_str(content) if content else ""


def _assistant_blocks(m: ChatMessage) -> list[dict]:
    """Claude's own blocks when this element produced the turn; otherwise built from AgentDojo's fields."""
    if m.get(CLAUDE_CONTENT):
        return list(m[CLAUDE_CONTENT])
    blocks: list[dict] = [{"type": "text", "text": c["content"]} for c in (m["content"] or [])
                          if c["type"] == "text" and c["content"]]
    blocks += [{"type": "tool_use", "id": c.id, "name": c.function, "input": dict(c.args)} for c in (m["tool_calls"] or [])]
    return blocks


def to_claude_messages(messages: Sequence[ChatMessage]) -> tuple[str | None, list[dict]]:
    """AgentDojo messages -> (system, Messages API messages). Consecutive user-side messages (the tool results of
    one assistant turn) merge into one user turn, so the results of parallel tool calls go back together."""
    system = None
    out: list[dict] = []
    for i, m in enumerate(messages):
        role = m["role"]
        if role == "system":
            if i != 0:
                raise ValueError("a system message is supported only as the first message")
            system = _text(m["content"])
            continue
        if role == "user":
            blocks = [{"type": "text", "text": _text(m["content"])}]
        elif role == "assistant":
            blocks = _assistant_blocks(m)
            if not blocks:  # an empty turn (a refusal before any output) cannot be sent and holds nothing to replay
                continue
        elif role == "tool":
            tc = m.get("tool_call")
            result: dict = {"type": "tool_result", "tool_use_id": m.get("tool_call_id") or (tc.id if tc is not None else None)}
            text = m["error"] if m.get("error") else _text(m["content"])
            if text:
                result["content"] = text
            if m.get("error"):
                result["is_error"] = True
            blocks = [result]
        else:  # pragma: no cover - AgentDojo defines only these four roles
            raise ValueError(f"unknown message role {role!r}")
        side = "assistant" if role == "assistant" else "user"
        if out and out[-1]["role"] == side == "user":
            out[-1]["content"].extend(blocks)
        else:
            out.append({"role": side, "content": blocks})
    return system, out


def parse_response(resp) -> tuple[ChatAssistantMessage, list[str]]:
    """Messages API response -> (AgentDojo assistant message carrying Claude's blocks, names of tool calls whose
    input is not a JSON object). Tool calls count only when Claude stopped to use them."""
    texts, calls, raw, malformed = [], [], [], []
    for block in resp.content:
        if block.type == "text":
            if block.text:  # an empty text block would be rejected when replayed
                raw.append({"type": "text", "text": block.text})
                texts.append(text_content_block_from_string(block.text))
        elif block.type == "thinking":
            raw.append({"type": "thinking", "thinking": block.thinking, "signature": block.signature})
        elif block.type == "redacted_thinking":
            raw.append({"type": "redacted_thinking", "data": block.data})
        elif block.type == "tool_use":
            raw.append({"type": "tool_use", "id": block.id, "name": block.name, "input": block.input})
            if not isinstance(block.input, dict):
                malformed.append(block.name)
            calls.append(FunctionCall(function=block.name, args=block.input if isinstance(block.input, dict) else {},
                                      id=block.id))
        else:
            raw.append(block.model_dump(mode="json", exclude_none=True))
    if resp.stop_reason != "tool_use":
        calls = []
    message = ChatAssistantMessage(role="assistant", content=texts or None, tool_calls=calls or None)
    message[CLAUDE_CONTENT] = raw  # type: ignore[typeddict-unknown-key] - replayed verbatim by to_claude_messages
    return message, malformed


def ledger_usage(usage) -> dict:
    """Messages API usage -> the ledger's shape: cache writes count as uncached prompt tokens, cache reads as cached."""
    write = usage.cache_creation_input_tokens or 0
    read = usage.cache_read_input_tokens or 0
    return {"prompt_tokens": (usage.input_tokens or 0) + write + read, "completion_tokens": usage.output_tokens or 0,
            "prompt_tokens_details": {"cached_tokens": read}}


class ClaudeLLM(BasePipelineElement):
    def __init__(
        self,
        name: str,
        model: str,
        api_key_env: str,
        max_tokens: int,
        effort: str,
        timeout_s: float,
        ledger: SpendLedger,
        *,
        output_format: dict | None = None,
        cache: bool = True,
        max_attempts: int = 5,
        backoff_s: float = 1.0,
        purpose: str = "",
        client=None,
    ) -> None:
        if client is None:
            api_key = os.environ.get(api_key_env, "")
            if not api_key.strip():
                raise MissingSecretError(api_key_env)
            client = anthropic.Anthropic(api_key=api_key, timeout=timeout_s, max_retries=0)
        self.name = name
        self.model = model
        self.api_key_env = api_key_env
        self.max_tokens = max_tokens
        self.effort = effort
        self.ledger = ledger
        self.output_format = output_format
        self.cache = cache
        self.max_attempts = max_attempts
        self.backoff_s = backoff_s
        self.purpose = purpose
        self.client = client

    # -- request construction -------------------------------------------------------------
    def request_params(self, system: str | None, messages: list[dict], tools: list[dict]) -> dict:
        output_config: dict = {"effort": self.effort}
        if self.output_format is not None:
            output_config["format"] = self.output_format
        params: dict = {"model": self.model, "max_tokens": self.max_tokens, "messages": messages,
                        "output_config": output_config}
        if self.cache:  # the conversation prefix repeats on every turn of a case
            params["cache_control"] = {"type": "ephemeral"}
        if system:
            params["system"] = system
        if tools:
            params["tools"] = tools
        return params

    def _send(self, params: dict, messages_in, extra_args) -> tuple[object, int]:
        attempt = 0
        while True:
            attempt += 1
            try:
                return self.client.messages.create(**params), attempt
            except anthropic.APITimeoutError as exc:  # before APIConnectionError, its base class
                err: LLMError = LLMTimeoutError(f"request timed out: {exc}", messages_in, extra_args)
            except anthropic.APIConnectionError as exc:
                err = LLMConnectionError(f"connection failed: {exc}", messages_in, extra_args)
            except anthropic.APIStatusError as exc:
                text = str(exc.message)
                if exc.status_code in (401, 403) or _NO_CREDIT.search(text):
                    raise AccountError(self.api_key_env, f"HTTP {exc.status_code}: {text}") from exc
                if exc.status_code == 429 or exc.status_code >= 500:  # 529 is "overloaded"
                    err = LLMServerError(f"HTTP {exc.status_code}: {text}", messages_in, extra_args)
                elif _CONTEXT.search(text):
                    raise ContextOverflowError(f"HTTP {exc.status_code} context overflow: {text}", messages_in,
                                               extra_args) from exc
                else:
                    raise LLMBadRequestError(f"HTTP {exc.status_code}: {text}", messages_in, extra_args) from exc
            except anthropic.APIError as exc:  # for example a response the SDK could not validate
                raise LLMResponseError(f"unusable response: {exc}", messages_in, extra_args) from exc
            if attempt >= self.max_attempts:
                raise err
            time.sleep(min(self.backoff_s * 2 ** (attempt - 1), 30.0))

    # -- pipeline element ------------------------------------------------------------------
    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = [],
        extra_args: dict = {},
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict]:
        extra_args = dict(extra_args)
        calls = list(extra_args.get("aib_calls", []))
        system, claude_messages = to_claude_messages(messages)
        tools = [tool_schema(f) for f in runtime.functions.values()]
        params = self.request_params(system, claude_messages, tools)
        record = ClaudeCallRecord(turn=len(calls), seed=None, latency_s=0.0, attempts=0)
        prompt_chars = len(system or "") + len(json.dumps(claude_messages)) + len(json.dumps(tools))
        reservation = self.ledger.reserve(prompt_chars, self.max_tokens, self.purpose)

        start = time.monotonic()
        try:
            resp, record.attempts = self._send(params, messages, {**extra_args, "aib_calls": calls})
        except Exception as exc:  # any failure frees the reservation and leaves a `failed` ledger line
            record.latency_s = time.monotonic() - start
            kind = getattr(exc, "kind", None) or ("account" if isinstance(exc, AccountError) else type(exc).__name__)
            record.error = f"{kind}: {exc}"[:500]
            self.ledger.release(reservation, record.error, self.purpose)
            if isinstance(exc, LLMError):
                exc.extra_args = {**extra_args, "aib_calls": [*calls, record.to_dict()]}
            raise
        record.latency_s = time.monotonic() - start

        usage = ledger_usage(resp.usage)
        record.prompt_tokens = usage["prompt_tokens"]
        record.cached_tokens = usage["prompt_tokens_details"]["cached_tokens"]
        record.cache_write_tokens = resp.usage.cache_creation_input_tokens or 0
        record.completion_tokens = usage["completion_tokens"]
        record.response_model = resp.model
        record.cost_usd = self.ledger.record(reservation, usage, record.response_model, self.purpose)
        record.finish_reason = resp.stop_reason
        if resp.stop_reason == "refusal" and resp.stop_details is not None:
            record.stop_category = resp.stop_details.category
        assistant, malformed = parse_response(resp)
        record.n_tool_calls = len(assistant["tool_calls"] or [])
        record.malformed_args = malformed
        extra_args["aib_calls"] = [*calls, record.to_dict()]
        return query, runtime, env, [*messages, assistant], extra_args
