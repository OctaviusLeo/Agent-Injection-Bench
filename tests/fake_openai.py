"""A scripted OpenAI-compatible HTTP server on 127.0.0.1 for unit tests (no real network)."""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def completion(content=None, tool_calls=None, finish="stop", prompt=10, completion_tokens=5, cached=0, model="fake-model"):
    """A Chat Completions response body. `tool_calls` is a list of (id, name, arguments-string)."""
    message = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = [
            {"id": cid, "type": "function", "function": {"name": name, "arguments": args}} for cid, name, args in tool_calls
        ]
        finish = "tool_calls"
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 0,
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        "usage": {
            "prompt_tokens": prompt,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt + completion_tokens,
            "prompt_tokens_details": {"cached_tokens": cached},
        },
    }


class FakeOpenAI:
    """Serves scripted (status, body, delay_s) replies in order; repeats the last one when exhausted."""

    def __init__(self, replies, routes=None):
        self.replies = list(replies)
        self.routes = dict(routes or {})  # GET path -> (status, body)
        self.requests: list[dict] = []
        self.paths: list[str] = []
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _reply(self, status, body):
                data = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                try:
                    self.wfile.write(data)
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def do_GET(self):
                fake.paths.append(self.path)
                status, body = fake.routes.get(self.path, (404, {"error": "not found"}))
                self._reply(status, body)

            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length) or b"{}")
                fake.paths.append(self.path)
                fake.requests.append(body)
                idx = min(len(fake.requests) - 1, len(fake.replies) - 1)
                status, reply, delay = fake.replies[idx]
                if delay:
                    time.sleep(delay)
                self._reply(status, reply)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_address[1]}/v1"

    @property
    def root_url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_address[1]}"

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()
        return False
