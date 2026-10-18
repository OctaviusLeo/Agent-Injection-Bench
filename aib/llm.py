"""OpenAI-compatible chat model element for AgentDojo pipelines: llama-server and the hosted API (Req 1.5, 5.2, 5.3).

`ChatLLM` converts AgentDojo messages to Chat Completions messages (system role `system`, never
`developer`), sends the tool schemas and the sampling settings with a per-request seed, parses tool
calls back into AgentDojo `FunctionCall`s, retries transient failures, and appends one `CallRecord`
per request to `extra_args["aib_calls"]`. Hosted requests go through the `SpendLedger`.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field

import openai
from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.functions_runtime import EmptyEnv, Env, Function, FunctionCall, FunctionsRuntime
from agentdojo.types import ChatAssistantMessage, ChatMessage, get_text_content_as_str, text_content_block_from_string

from aib.ledger import SpendLedger

LOCAL_API_KEY = "sk-no-key-required"
HOSTED_SAMPLING_PARAMS = ("temperature", "top_p", "seed")
_CONTEXT_PATTERNS = re.compile(r"context (size|length|window)|exceed_context|context_length_exceeded|too many tokens", re.I)
_UNSUPPORTED = re.compile(r"unsupported (parameter|value)|not supported|does not support", re.I)


class MissingSecretError(RuntimeError):
    """A required secret environment variable is unset or empty (Req 0.13)."""

    def __init__(self, env_var: str) -> None:
        super().__init__(f"required environment variable {env_var} is unset or empty")
        self.env_var = env_var


class LLMError(RuntimeError):
    """A request failed. Carries the conversation and extra_args reached, so the runner can record them."""

    kind = "error"

    def __init__(self, message: str, messages: Sequence[ChatMessage] = (), extra_args: dict | None = None) -> None:
        super().__init__(message)
        self.messages = list(messages)
        self.extra_args = dict(extra_args or {})


class LLMTimeoutError(LLMError):
    kind = "timeout"


class LLMConnectionError(LLMError):
    kind = "connection"


class LLMServerError(LLMError):
    """429 or 5xx still failing after the retries."""

    kind = "server"


class LLMBadRequestError(LLMError):
    kind = "bad_request"


class ContextOverflowError(LLMBadRequestError):
    """The conversation no longer fits the server's context (llama-server answers HTTP 400)."""

    kind = "context_overflow"


class UnsupportedParameterError(LLMBadRequestError):
    kind = "unsupported_parameter"

    def __init__(self, message: str, param: str | None, **kw) -> None:
        super().__init__(message, **kw)
        self.param = param


class LLMResponseError(LLMError):
    """The server answered, but the response cannot be used (no choices)."""

    kind = "bad_response"


@dataclass(frozen=True)
class Sampling:
    temperature: float
    top_p: float
    max_tokens: int
    extra_body: dict = field(default_factory=dict)  # local only: top_k, min_p


@dataclass
class CallRecord:
    turn: int
    seed: int
    latency_s: float
    attempts: int
    prompt_tokens: int = 0
    cached_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str | None = None
    response_model: str | None = None
    n_tool_calls: int = 0
    malformed_args: list[str] = field(default_factory=list)
    cost_usd: float | None = None
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def request_seed(case_seed: int, turn: int) -> int:
    """Per-request seed from the case seed and the turn number (first 4 bytes of SHA-256, < 2**31)."""
    digest = hashlib.sha256(f"{case_seed}|{turn}".encode()).digest()
    return int.from_bytes(digest[:4], "big") % (2**31)


def tool_schema(f: Function) -> dict:
    return {
        "type": "function",
        "function": {"name": f.name, "description": f.description, "parameters": f.parameters.model_json_schema()},
    }


def to_openai_messages(messages: Sequence[ChatMessage]) -> list[dict]:
    """AgentDojo messages -> Chat Completions messages with string content."""
    out: list[dict] = []
    for m in messages:
        role = m["role"]
        text = get_text_content_as_str(m["content"] or []) if m.get("content") is not None else None
        if role in ("system", "user"):
            out.append({"role": role, "content": text or ""})
        elif role == "assistant":
            calls = m.get("tool_calls") or []
            msg: dict = {"role": "assistant", "content": text if text else (None if calls else "")}
            if calls:
                msg["tool_calls"] = [
                    {"id": c.id, "type": "function", "function": {"name": c.function, "arguments": json.dumps(dict(c.args))}}
                    for c in calls
                ]
            out.append(msg)
        elif role == "tool":
            tc = m.get("tool_call")
            call_id = m.get("tool_call_id") or (tc.id if tc is not None else None)
            content = m["error"] if m.get("error") else (text or "")
            out.append({"role": "tool", "tool_call_id": call_id, "content": content})
        else:  # pragma: no cover - AgentDojo defines only these four roles
            raise ValueError(f"unknown message role {role!r}")
    return out


def parse_tool_calls(raw_calls, turn: int) -> tuple[list[FunctionCall], list[str]]:
    """Server tool calls -> FunctionCalls. Arguments that are not a JSON object become {} and are flagged."""
    calls, malformed = [], []
    for i, rc in enumerate(raw_calls or []):
        name = rc.function.name or ""
        try:
            args = json.loads(rc.function.arguments or "{}")
            if not isinstance(args, dict):
                raise ValueError("arguments are not a JSON object")
        except (ValueError, TypeError):
            args = {}
            malformed.append(name)
        calls.append(FunctionCall(function=name, args=args, id=rc.id or f"call_{turn}_{i}"))
    return calls, malformed


def _usage(resp) -> dict:
    u = getattr(resp, "usage", None)
    if u is None:
        return {}
    d = u.model_dump() if hasattr(u, "model_dump") else dict(u)
    return {
        "prompt_tokens": d.get("prompt_tokens") or 0,
        "completion_tokens": d.get("completion_tokens") or 0,
        "prompt_tokens_details": {"cached_tokens": ((d.get("prompt_tokens_details") or {}).get("cached_tokens")) or 0},
    }


def _error_text(exc: openai.APIStatusError) -> tuple[str, str | None]:
    body = exc.body if isinstance(exc.body, dict) else {}
    err = body.get("error", body) if isinstance(body, dict) else {}
    if not isinstance(err, dict):
        err = {"message": str(err)}
    return str(err.get("message") or exc.message or exc), err.get("param")


class ChatLLM(BasePipelineElement):
    def __init__(
        self,
        name: str,
        base_url: str,
        model: str,
        api_key_env: str | None,
        sampling: Sampling,
        timeout_s: float,
        reasoning_effort: str | None = None,
        ledger: SpendLedger | None = None,
        *,
        provider: str = "llamacpp",
        accepted_params: Sequence[str] | None = None,
        max_attempts: int = 5,
        backoff_s: float = 1.0,
        purpose: str = "",
    ) -> None:
        if provider not in ("llamacpp", "openai"):
            raise ValueError(f"unknown provider {provider!r}")
        api_key = LOCAL_API_KEY
        if api_key_env is not None:
            api_key = os.environ.get(api_key_env, "")
            if not api_key.strip():
                raise MissingSecretError(api_key_env)
        self.name = name
        self.model = model
        self.provider = provider
        self.sampling = sampling
        self.reasoning_effort = reasoning_effort
        self.ledger = ledger
        self.accepted_params = tuple(accepted_params) if accepted_params is not None else HOSTED_SAMPLING_PARAMS
        self.max_attempts = max_attempts
        self.backoff_s = backoff_s
        self.purpose = purpose
        self.client = openai.OpenAI(base_url=base_url, api_key=api_key, timeout=timeout_s, max_retries=0)

    # -- request construction -------------------------------------------------------------
    def request_params(self, messages: list[dict], tools: list[dict], seed: int) -> dict:
        params: dict = {"model": self.model, "messages": messages}
        if tools:
            params["tools"] = tools
            params["tool_choice"] = "auto"
        sampling = {"temperature": self.sampling.temperature, "top_p": self.sampling.top_p, "seed": seed}
        if self.provider == "openai":
            params.update({k: v for k, v in sampling.items() if k in self.accepted_params})
            params["max_completion_tokens"] = self.sampling.max_tokens
            if self.reasoning_effort is not None:
                params["reasoning_effort"] = self.reasoning_effort
        else:
            params.update(sampling)
            params["max_tokens"] = self.sampling.max_tokens
            if self.sampling.extra_body:
                params["extra_body"] = dict(self.sampling.extra_body)
        return params

    def _send(self, params: dict, messages_in, extra_args) -> tuple[object, int]:
        attempt = 0
        while True:
            attempt += 1
            try:
                return self.client.chat.completions.create(**params), attempt
            except openai.APITimeoutError as exc:
                err: LLMError = LLMTimeoutError(f"request timed out: {exc}", messages_in, extra_args)
            except openai.APIConnectionError as exc:
                err = LLMConnectionError(f"connection failed: {exc}", messages_in, extra_args)
            except openai.APIStatusError as exc:
                text, param = _error_text(exc)
                if exc.status_code == 429 or exc.status_code >= 500:
                    err = LLMServerError(f"HTTP {exc.status_code}: {text}", messages_in, extra_args)
                elif exc.status_code == 400 and _CONTEXT_PATTERNS.search(text):
                    raise ContextOverflowError(f"HTTP 400 context overflow: {text}", messages_in, extra_args) from exc
                elif exc.status_code == 400 and (param or _UNSUPPORTED.search(text)):
                    raise UnsupportedParameterError(f"HTTP 400: {text}", param, messages=messages_in,
                                                    extra_args=extra_args) from exc
                else:
                    raise LLMBadRequestError(f"HTTP {exc.status_code}: {text}", messages_in, extra_args) from exc
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
        turn = len(calls)
        seed = request_seed(int(extra_args.get("aib_seed", 0)), turn)
        oa_messages = to_openai_messages(messages)
        tools = [tool_schema(f) for f in runtime.functions.values()]
        params = self.request_params(oa_messages, tools, seed)
        record = CallRecord(turn=turn, seed=seed, latency_s=0.0, attempts=0)

        reservation = None
        if self.ledger is not None:
            prompt_chars = len(json.dumps(oa_messages)) + len(json.dumps(tools))
            reservation = self.ledger.reserve(prompt_chars, self.sampling.max_tokens, self.purpose)

        start = time.monotonic()
        try:
            resp, record.attempts = self._send(params, messages, {**extra_args, "aib_calls": calls})
        except LLMError as exc:
            record.latency_s = time.monotonic() - start
            record.error = f"{exc.kind}: {exc}"[:500]
            if reservation is not None:
                self.ledger.release(reservation, record.error, self.purpose)
            exc.extra_args = {**extra_args, "aib_calls": [*calls, record.to_dict()]}
            raise
        record.latency_s = time.monotonic() - start

        usage = _usage(resp)
        record.prompt_tokens = usage.get("prompt_tokens", 0)
        record.cached_tokens = usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)
        record.completion_tokens = usage.get("completion_tokens", 0)
        record.response_model = getattr(resp, "model", None)
        if reservation is not None:
            record.cost_usd = self.ledger.record(reservation, usage, record.response_model, self.purpose)

        if not getattr(resp, "choices", None):
            record.error = "bad_response: no choices"
            raise LLMResponseError("response has no choices", messages, {**extra_args, "aib_calls": [*calls, record.to_dict()]})
        choice = resp.choices[0]
        record.finish_reason = choice.finish_reason
        tool_calls, malformed = parse_tool_calls(choice.message.tool_calls, turn)
        record.n_tool_calls = len(tool_calls)
        record.malformed_args = malformed
        content = choice.message.content
        assistant = ChatAssistantMessage(
            role="assistant",
            content=[text_content_block_from_string(content)] if content is not None else None,
            tool_calls=tool_calls or None,
        )
        extra_args["aib_calls"] = [*calls, record.to_dict()]
        return query, runtime, env, [*messages, assistant], extra_args
