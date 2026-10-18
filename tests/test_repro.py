"""Unit tests for aib.repro: commit-hash shape and Stamp round-trip (Req 0.1)."""

from __future__ import annotations

import re

from aib.repro import NO_COMMIT_SENTINEL, Stamp, commit_hash, deserialize, serialize, stamp

_HEX40 = re.compile(r"\A[0-9a-f]{40}\Z")


def test_commit_hash_is_40_hex_or_sentinel():
    value = commit_hash()
    assert len(value) == 40
    assert _HEX40.match(value) is not None or value == NO_COMMIT_SENTINEL


def test_sentinel_is_40_zeros():
    assert NO_COMMIT_SENTINEL == "0" * 40
    assert len(NO_COMMIT_SENTINEL) == 40


def test_stamp_captures_command_and_seed():
    s = stamp("python -m aib.metrics", 1234)
    assert s.command == "python -m aib.metrics"
    assert s.seed == 1234
    assert len(s.commit) == 40


def test_stamp_round_trips_through_serialize_deserialize():
    original = Stamp(command="cmd --flag", seed=7, commit="a" * 40)
    restored = deserialize(serialize(original))
    assert restored == original


def test_deserialize_coerces_seed_to_int():
    restored = deserialize(serialize(Stamp(command="c", seed=3, commit="b" * 40)))
    assert isinstance(restored.seed, int)
    assert restored.seed == 3
