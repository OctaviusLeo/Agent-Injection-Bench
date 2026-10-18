"""Property test for Stub_Model determinism (design.md Property 10)."""

from __future__ import annotations

import json

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from aib.attacks import STYLES
from aib.classifier import KeywordClassifier
from aib.config import load_defenses
from aib.defenses import DEFENSES
from aib.goals import goal_for
from aib.runner import CaseSpec, RunContext, run_case, trace_file
from aib.stub_model import StubLLM
from tests.sandbox_helpers import frozen_tool_clock

CFG = load_defenses()
CONDITIONS = ["clean", *sorted(STYLES)]


# Feature: agent-injection-bench, Property 10: Stub_Model drives a deterministic case outcome
@settings(max_examples=100, derandomize=True, deadline=None,
          suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture])
@given(k=st.integers(min_value=0, max_value=39), condition=st.sampled_from(CONDITIONS),
       defense=st.sampled_from(DEFENSES), policy=st.sampled_from(["follow_user", "follow_injection"]),
       seed=st.integers(min_value=0, max_value=2**31 - 1))
def test_property_10_stub_cases_are_deterministic(tmp_path, k, condition, defense, policy, seed):
    task = f"user_task_{k}"
    spec = CaseSpec("stub", defense, condition, task, None if condition == "clean" else goal_for(task), 0, seed)
    outcomes = []
    for run in ("a", "b"):
        ctx = RunContext(sampling={}, stamp={}, defense_cfg=CFG, trace_dir=tmp_path / run)
        with frozen_tool_clock():  # tools stamp new emails/files with the wall clock
            r = run_case(spec, StubLLM(policy), KeywordClassifier(), ctx)
        trace = json.loads(trace_file(tmp_path / run, spec.case_id).read_text())
        outcomes.append({
            "messages": trace["messages"], "tool_calls": r.tool_calls, "status": r.status, "exposed": r.exposed,
            "task_success": r.task_success, "injection_success": r.injection_success, "unmet": r.unmet,
            "filter": r.filter_decisions, "final_output_empty": r.final_output_empty,
        })
    assert outcomes[0] == outcomes[1]
