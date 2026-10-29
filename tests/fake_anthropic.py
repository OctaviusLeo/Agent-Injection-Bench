"""A scripted stand-in for the Anthropic client in unit tests: real SDK response and error types, no key, no network."""

from __future__ import annotations

import copy

import anthropic
import httpx2

REQUEST = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")


def message(content, stop="end_turn", input_tokens=10, output_tokens=5, cache_write=0, cache_read=0, category=None,
            model="claude-haiku-5-5"):
    """A Messages API response; `content` is a list of block dicts, `category` makes a refusal's stop details."""
    data = {"id": "msg_1", "type": "message", "role": "assistant", "model": model, "content": content,
            "stop_reason": stop, "stop_sequence": None,
            "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens,
                      "cache_creation_input_tokens": cache_write, "cache_read_input_tokens": cache_read}}
    if category:
        data["stop_details"] = {"type": "refusal", "category": category, "explanation": "declined"}
    return anthropic.types.Message.model_validate(data)


def text(value, **kw):
    return message([{"type": "text", "text": value}], **kw)


def status_error(cls, status, text):
    return cls(text, response=httpx2.Response(status, request=REQUEST), body=None)


class FakeClient:
    """`client.messages.create(**params)` returns (or raises) scripted replies in order and repeats the last one."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.requests: list[dict] = []
        self.messages = self

    def create(self, **params):
        self.requests.append(copy.deepcopy(params))
        reply = self.replies[min(len(self.requests) - 1, len(self.replies) - 1)]
        if isinstance(reply, Exception):
            raise reply
        return reply
