"""Unit tests for aib.metrics (Wilson interval; task 5.2 adds the rest)."""

from __future__ import annotations

import pytest

from aib import metrics as M


def test_wilson_known_values():
    a = M.wilson_interval(27, 30)
    assert a.point == pytest.approx(0.9) and a.lower == pytest.approx(0.744, abs=1e-3) and a.upper == pytest.approx(0.965, abs=1e-3)
    b = M.wilson_interval(15, 30)
    assert b.lower == pytest.approx(0.332, abs=1e-3) and b.upper == pytest.approx(0.668, abs=1e-3)
    assert b.method == "wilson" and b.n == 30


def test_wilson_edges_and_errors():
    assert M.wilson_interval(0, 5).lower == 0.0 and M.wilson_interval(5, 5).upper == 1.0
    with pytest.raises(ValueError):
        M.wilson_interval(1, 0)
    with pytest.raises(ValueError):
        M.wilson_interval(6, 5)
    nc = M.rate_or_not_computed(0, 0, "no flagged outputs")
    assert isinstance(nc, M.NotComputed) and nc.to_dict() == {"not_computed": True, "cause": "no flagged outputs"}
    assert M.rate_or_not_computed(1, 2, "x").to_dict()["point"] == 0.5
