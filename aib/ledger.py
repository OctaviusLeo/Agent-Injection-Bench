"""Spend ledger for the hosted model: refuses a request before sending it if its worst case could
push total spend past the Spend_Cap, and records the cost of every request it sends (Req 5.2).

The ledger file (results/hosted_ledger.jsonl) is append-only; its total is the sum of `cost_usd` over
every line, including lines written by earlier runs, so the cap holds over the whole project.
Reservations make the precheck safe with several concurrent workers: each worker reserves its worst
case before sending and settles it with the real cost afterwards.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import os
import threading
from dataclasses import dataclass
from pathlib import Path


class SpendCapExceeded(RuntimeError):
    """A request was refused before sending because its worst case could exceed the Spend_Cap."""

    def __init__(self, spent: float, reserved: float, worst: float, cap: float) -> None:
        super().__init__(
            f"Spend_Cap reached: spent ${spent:.6f} (+${reserved:.6f} in flight); next request could cost "
            f"${worst:.6f}; cap ${cap:.2f}"
        )
        self.spent, self.reserved, self.worst, self.cap = spent, reserved, worst, cap


@dataclass(frozen=True)
class Prices:
    """USD per 1M tokens."""

    cache_write: float
    cached_input: float
    output: float


@dataclass(frozen=True)
class Reservation:
    worst_usd: float


def request_cost(prices: Prices, prompt_tokens: int, cached_tokens: int, completion_tokens: int) -> float:
    """Conservative cost: uncached prompt tokens at the cache-write rate, cached at the cached rate."""
    uncached = max(prompt_tokens - cached_tokens, 0)
    return (uncached * prices.cache_write + cached_tokens * prices.cached_input + completion_tokens * prices.output) / 1e6


class SpendLedger:
    def __init__(self, path: Path | str, cap_usd: float, prices: Prices, chars_per_token: float = 3.0) -> None:
        if cap_usd < 0:
            raise ValueError("cap_usd must be non-negative")
        self.path = Path(path)
        self.cap_usd = float(cap_usd)
        self.prices = prices
        self.chars_per_token = float(chars_per_token)
        self._lock = threading.Lock()
        self._reserved = 0.0
        self._spent = self._load_total()

    def _load_total(self) -> float:
        if not self.path.exists():
            return 0.0
        total = 0.0
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                total += float(json.loads(line).get("cost_usd", 0.0))
        return total

    @property
    def total_usd(self) -> float:
        with self._lock:
            return self._spent

    def worst_case(self, prompt_chars: int, max_completion_tokens: int) -> float:
        prompt_tokens = math.ceil(prompt_chars / self.chars_per_token)
        return request_cost(self.prices, prompt_tokens, 0, max_completion_tokens)

    def reserve(self, prompt_chars: int, max_completion_tokens: int, purpose: str = "") -> Reservation:
        """Reserve the worst case or raise SpendCapExceeded (and log the refusal) without sending anything."""
        worst = self.worst_case(prompt_chars, max_completion_tokens)
        with self._lock:
            if self._spent + self._reserved + worst > self.cap_usd:
                self._append({"event": "refused", "purpose": purpose, "worst_usd": worst, "cost_usd": 0.0,
                              "total_usd": self._spent, "cap_usd": self.cap_usd})
                raise SpendCapExceeded(self._spent, self._reserved, worst, self.cap_usd)
            self._reserved += worst
        return Reservation(worst)

    def record(self, reservation: Reservation, usage: dict, response_model: str | None, purpose: str = "") -> float:
        """Settle a reservation with the cost computed from the response `usage`; returns that cost."""
        prompt = int(usage.get("prompt_tokens") or 0)
        cached = int(((usage.get("prompt_tokens_details") or {}).get("cached_tokens")) or 0)
        completion = int(usage.get("completion_tokens") or 0)
        cost = request_cost(self.prices, prompt, cached, completion)
        with self._lock:
            self._reserved = max(self._reserved - reservation.worst_usd, 0.0)
            self._spent += cost
            self._append({"event": "request", "purpose": purpose, "response_model": response_model,
                          "prompt_tokens": prompt, "cached_tokens": cached, "completion_tokens": completion,
                          "cost_usd": cost, "total_usd": self._spent, "cap_usd": self.cap_usd})
        return cost

    def release(self, reservation: Reservation, error: str, purpose: str = "") -> None:
        """Drop a reservation whose request failed without a usable response (no usage, no cost recorded)."""
        with self._lock:
            self._reserved = max(self._reserved - reservation.worst_usd, 0.0)
            self._append({"event": "failed", "purpose": purpose, "error": error[:300], "cost_usd": 0.0,
                          "total_usd": self._spent, "cap_usd": self.cap_usd})

    def _append(self, entry: dict) -> None:
        entry = {"ts": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), **entry}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, sort_keys=True) + "\n")
            f.flush()
            os.fsync(f.fileno())
