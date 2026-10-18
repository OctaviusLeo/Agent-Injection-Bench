"""Unit and property tests for aib.metrics (design.md Properties 5-9)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from aib import metrics as M

TASKS = [f"user_task_{i}" for i in range(6)]
SETTINGS = dict(max_examples=100, derandomize=True, deadline=None, suppress_health_check=[HealthCheck.too_slow])


def rec(task, status, inj=None, ts=None, defense="none", condition="direct", rep=0, model="m", exposed=None, decisions=()):
    if exposed is None:
        exposed = status == "classified"
    return {"case_id": f"{model}|{defense}|{condition}|{task}|r{rep}", "model": model, "defense": defense,
            "condition": condition, "user_task_id": task, "rep": rep, "status": status,
            "exposed": None if status in ("clean", "errored") else exposed,
            "injection_success": inj if status == "classified" else None,
            "task_success": None if status == "errored" else ts, "filter_decisions": list(decisions),
            "sampling": {"temperature": 0.7}, "model_version": "v"}


KINDS = ["carried", "not_carried", "unclassified", "errored"]


def from_kind(task, kind, ts, rep):
    if kind == "carried":
        return rec(task, "classified", True, ts, rep=rep)
    if kind == "not_carried":
        return rec(task, "classified", False, ts, rep=rep)
    if kind == "unclassified":
        return rec(task, "unclassified_not_exposed", None, ts, rep=rep)
    return rec(task, "errored", rep=rep)


records_strategy = st.lists(
    st.tuples(st.sampled_from(TASKS), st.sampled_from(KINDS), st.booleans()), min_size=0, max_size=40,
).map(lambda rows: [from_kind(t, k, ts, i) for i, (t, k, ts) in enumerate(rows)])


# ---------------------------------------------------------------- unit tests
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


def test_cluster_interval_is_wider_than_wilson_for_clustered_outcomes():
    by_task = {"a": (15, 15), "b": (0, 15)}
    boot = M.cluster_bootstrap_rate(by_task, seed=1, resamples=2000)
    wil = M.wilson_interval(15, 30)
    assert boot.point == wil.point == 0.5
    assert boot.upper - boot.lower > wil.upper - wil.lower
    assert boot.method == "bootstrap-task-cluster-2000" and boot.n == 30


def test_precision_and_recall_on_labeled_fixture():
    d = lambda inj, flag: {"injected": inj, "flagged": flag, "score": 0.9 if flag else 0.1, "output_sha256": "x"}  # noqa: E731
    rs = [rec("user_task_0", "classified", False, True, defense="filter", decisions=[d(True, True), d(False, True), d(True, False)]),
          rec("user_task_1", "classified", False, True, defense="filter", decisions=[d(True, True), d(False, False)])]
    p, r = M.filter_precision_recall(rs, seed=3)
    assert p.point == pytest.approx(2 / 3) and r.point == pytest.approx(2 / 3)
    fpr = M.false_positive_rate(rs, seed=4)
    assert fpr.point == pytest.approx(1 / 2)
    none_flagged = [rec("user_task_0", "classified", False, True, defense="filter", decisions=[d(False, False)])]
    p2, r2 = M.filter_precision_recall(none_flagged)
    assert isinstance(p2, M.NotComputed) and isinstance(r2, M.NotComputed)


def test_empty_cell_is_not_computed_with_cause():
    rs = [rec("user_task_0", "unclassified_not_exposed", None, True), rec("user_task_1", "errored")]
    asr = M.attack_success_rate(rs)
    assert isinstance(asr, M.NotComputed) and "no classified" in asr.cause
    assert isinstance(M.task_success_rate([rec("user_task_1", "errored")]), M.NotComputed)
    assert isinstance(M.exposure_rate([rec("user_task_0", "clean", ts=True, condition="clean")]), M.NotComputed)
    assert isinstance(M.paired_bootstrap_difference({"a": (1, 2)}, {"a": (0, 0)}, 1), M.NotComputed)


def _write_store(directory: Path, model: str, records: list[dict]):
    directory.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps({"header": True, "stamp": {"commit": "a" * 40}, "config": {}})] + [json.dumps(r) for r in records]
    (directory / f"{model}.jsonl").write_text("\n".join(lines) + "\n")


def _grid_records(model="m"):
    out = []
    for d in ("none", "delimit", "filter"):
        for c in ("clean", "direct", "hidden_field"):
            for i, t in enumerate(TASKS):
                for rep in range(3):
                    if c == "clean":
                        out.append(rec(t, "clean", ts=(i + rep) % 3 != 0, defense=d, condition=c, rep=rep, model=model))
                    else:
                        carried = (i + rep + len(d)) % 4 == 0
                        dec = [{"injected": True, "flagged": d == "filter" and i % 2 == 0, "score": 0.0, "output_sha256": "x"},
                               {"injected": False, "flagged": i == 5, "score": 0.0, "output_sha256": "y"}]
                        out.append(rec(t, "classified", carried, (i + rep) % 2 == 0, defense=d, condition=c, rep=rep,
                                       model=model, decisions=dec if d == "filter" else ()))
    return out


def test_compute_metrics_covers_cells_costs_pooled_and_filter(tmp_path):
    _write_store(tmp_path, "m", _grid_records())
    m = M.compute_metrics(tmp_path, seed=7, resamples=500)
    assert len(m["cells"]) == 9 and m["models"]["m"]["repetitions"] == 3
    clean = next(c for c in m["cells"] if c["condition"] == "clean" and c["defense"] == "none")
    assert clean["asr"] is None and clean["tsr"]["method"] == "bootstrap-task-cluster-500"
    assert {p["defense"] for p in m["pooled"]} == {"none", "delimit", "filter"}
    cost = next(c for c in m["defense_cost"] if c["defense"] == "delimit" and c["condition"] == "clean")
    assert cost["reference"] == "none" and cost["asr_reduction"] is None
    assert cost["defense_cost"]["point"] == pytest.approx(cost["tsr_none"]["point"] - cost["tsr_defense"]["point"])
    assert any(c["condition"] == "pooled" for c in m["defense_cost"])
    filt = {f["condition"]: f for f in m["filter"]}
    assert set(filt) == {"clean", "direct", "hidden_field", "pooled"} and "precision" in filt["direct"]
    assert m["settings"]["resamples"] == 500


def test_main_writes_stamped_metrics(tmp_path, monkeypatch, capsys):
    import aib.repro

    store = tmp_path / "raw"
    _write_store(store, "m", _grid_records())
    monkeypatch.setattr(aib.repro, "require_clean", lambda: "a" * 40)
    out = tmp_path / "metrics.json"
    assert M.main(["--store", str(store), "--seed", "5", "--resamples", "200", "--out", str(out)]) == 0
    saved = json.loads(out.read_text())
    assert saved["stamp"]["seed"] == 5 and len(saved["stamp"]["commit"]) == 40 and "--resamples 200" in saved["stamp"]["command"]
    assert "9 cells" in capsys.readouterr().out


def test_load_store_skips_torn_lines(tmp_path):
    p = tmp_path / "x.jsonl"
    p.write_text(json.dumps({"header": True, "stamp": {}}) + "\n" + json.dumps(rec("user_task_0", "clean", ts=True)) + "\n{\"cut")
    header, records = M.load_store(p)
    assert header["header"] and len(records) == 1


# ---------------------------------------------------------------- properties
# Feature: agent-injection-bench, Property 5: Attack_Success_Rate counts only classified injected cases
@settings(**SETTINGS)
@given(records=records_strategy)
def test_property_5_asr_counts_only_classified(records):
    carried = sum(r["status"] == "classified" and r["injection_success"] is True for r in records)
    classified = sum(r["status"] == "classified" for r in records)
    asr = M.attack_success_rate(records, seed=1, resamples=200)
    if classified == 0:
        assert isinstance(asr, M.NotComputed) and asr.cause
    else:
        assert asr.point == pytest.approx(carried / classified, abs=1e-12) and 0.0 <= asr.point <= 1.0
        assert asr.n == classified
    with tempfile.TemporaryDirectory() as d:
        if records:
            _write_store(Path(d), "m", records)
            cell = M.compute_metrics(d, seed=1, resamples=100)["cells"][0]
            assert cell["unclassified"] == sum(r["status"] == "unclassified_not_exposed" for r in records)
            assert cell["errored"] == sum(r["status"] == "errored" for r in records)


# Feature: agent-injection-bench, Property 6: Rates and filter metrics lie in their ranges
@settings(**SETTINGS)
@given(outcomes=st.lists(st.tuples(st.sampled_from(TASKS), st.booleans(), st.booleans(), st.booleans(), st.booleans()),
                         min_size=1, max_size=40))
def test_property_6_rates_lie_in_their_ranges(outcomes):
    records = [rec(t, "classified", inj, ts, defense="filter", rep=i,
                   decisions=[{"injected": a, "flagged": b, "score": 0.0, "output_sha256": "z"}])
               for i, (t, inj, ts, a, b) in enumerate(outcomes)]
    values = [M.attack_success_rate(records, 1, 200), M.task_success_rate(records, 2, 200), *M.filter_precision_recall(records, 3, 200)]
    for v in values:
        if isinstance(v, M.RateCI):
            assert 0.0 <= v.lower <= v.point <= v.upper <= 1.0
    all_carried = [dict(r, injection_success=True) for r in records]
    none_carried = [dict(r, injection_success=False) for r in records]
    assert M.attack_success_rate(all_carried, 1, 200).point == 1.0
    assert M.attack_success_rate(none_carried, 1, 200).point == 0.0


per_task_counts = st.dictionaries(st.sampled_from(TASKS), st.integers(min_value=0, max_value=6).flatmap(
    lambda n: st.tuples(st.integers(min_value=0, max_value=n), st.just(n))), min_size=1)


# Feature: agent-injection-bench, Property 7: Confidence interval brackets the point estimate and is deterministic
@settings(**SETTINGS)
@given(n=st.integers(min_value=1, max_value=500), frac=st.floats(min_value=0, max_value=1), counts=per_task_counts,
       seed=st.integers(min_value=0, max_value=2**32))
def test_property_7_intervals_bracket_and_are_deterministic(n, frac, counts, seed):
    k = int(round(frac * n))
    w1, w2 = M.wilson_interval(k, n), M.wilson_interval(k, n)
    assert 0.0 <= w1.lower <= w1.point <= w1.upper <= 1.0 and w1 == w2
    b1 = M.cluster_bootstrap_rate(counts, seed, 300)
    b2 = M.cluster_bootstrap_rate(counts, seed, 300)
    if isinstance(b1, M.RateCI):
        assert 0.0 <= b1.lower <= b1.point <= b1.upper <= 1.0
        assert abs(b1.lower - b2.lower) <= 1e-6 and abs(b1.upper - b2.upper) <= 1e-6 and b1.point == b2.point
    else:
        assert all(v[1] == 0 for v in counts.values())


# Feature: agent-injection-bench, Property 8: Defense_Cost equals the task-success-rate difference from the control
@settings(**SETTINGS)
@given(a=per_task_counts, b=per_task_counts, seed=st.integers(min_value=0, max_value=2**32))
def test_property_8_defense_cost_is_the_tsr_difference(a, b, seed):
    diff = M.paired_bootstrap_difference(a, b, seed, 300)
    na, nb = sum(v[1] for v in a.values()), sum(v[1] for v in b.values())
    if na == 0 or nb == 0:
        assert isinstance(diff, M.NotComputed)
        return
    tsr_a = sum(v[0] for v in a.values()) / na
    tsr_b = sum(v[0] for v in b.values()) / nb
    assert abs(diff.point - M.defense_cost(tsr_a, tsr_b)) <= 1e-6
    assert -1.0 <= diff.lower <= diff.point <= diff.upper <= 1.0


# Feature: agent-injection-bench, Property 9: Metrics are deterministic under a fixed seed
@settings(max_examples=100, derandomize=True, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(records=records_strategy.filter(bool), seed=st.integers(min_value=0, max_value=2**32))
def test_property_9_metrics_are_deterministic(records, seed):
    with tempfile.TemporaryDirectory() as d:
        _write_store(Path(d), "m", records)
        first = M.compute_metrics(d, seed, 200)
        second = M.compute_metrics(d, seed, 200)
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
