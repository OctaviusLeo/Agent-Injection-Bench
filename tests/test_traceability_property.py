"""Property-based test for the number-traceability check (Correctness Property 11).

Feature: agent-injection-bench, Property 11: README/Report number traceability
"""

from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from aib.traceability import check_text, normalize_number

# Two DISJOINT pools of integer values so a "traceable" number can never
# normalize to the same canonical form as an "untraceable" one: traceable values
# are even, untraceable values are odd. Hypothesis picks subsets of each.
_EVENS = st.integers(min_value=0, max_value=400).map(lambda n: n * 2)
_ODDS = st.integers(min_value=0, max_value=400).map(lambda n: n * 2 + 1)


@given(
    traceable=st.lists(_EVENS, min_size=0, max_size=8, unique=True),
    untraceable=st.lists(_ODDS, min_size=0, max_size=8, unique=True),
)
def test_readme_report_number_traceability(traceable, untraceable):
    """Feature: agent-injection-bench, Property 11: README/Report number traceability."""
    # A generated source that originates exactly the traceable numbers.
    generated_source = "Results: " + " ".join(str(n) for n in traceable) + "."

    # A README/Report fragment that mixes traceable and untraceable numbers.
    all_numbers = [str(n) for n in traceable] + [str(n) for n in untraceable]
    doc_text = "The agent scored " + " and ".join(all_numbers) + " overall."

    result = check_text(doc_text, [generated_source])

    expected_norms = {normalize_number(str(n)) for n in untraceable}
    flagged_norms = {normalize_number(tok) for tok in result.untraceable}

    # The check flags EXACTLY the untraceable numbers (by canonical value).
    assert flagged_norms == expected_norms
    # ... and fails iff any untraceable number exists.
    assert result.ok == (len(untraceable) == 0)
