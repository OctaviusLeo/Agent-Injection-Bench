"""Unit and property tests for aib.traceability and scripts/check_traceability.py (design.md Property 11)."""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from aib import traceability as T
from scripts import check_traceability as CLI


def test_ignored_contexts():
    text = (
        "Models qwen3-4b, llama3.1-8b, and gpt-6-luna; milestone M0; agentdojo 0.1.35 and v7.0.1; read 2026-10-17.\n"
        "See https://arxiv.org/abs/2406.13352 and [the plot](results/plots/asr_by_defense.png), file "
        "docs/evidence/m0_smoke_qwen3-4b.json, and `python -m aib.metrics --seed 20261017`.\n"
        "```\nbash scripts/run_benchmark.sh qwen3-4b --workers 2\n```\n<!-- 99 -->\n"
        "1. first item\n2. second item\n"
    )
    assert T.extract_numbers(text) == []


def test_reported_numbers_are_extracted():
    text = "ASR 0.523 [0.401, 0.640]; cost -0.023 [-0.050, 0.004]; 3 Repetitions and 40 tasks."
    assert T.extract_numbers(text) == ["0.523", "0.401", "0.640", "-0.023", "-0.050", "0.004", "3", "40"]


def test_check_text_requires_verbatim_presence():
    sources = ['| qwen3-4b | none | direct | 0.523 [0.401, 0.640] | -0.023 [-0.050, 0.004] |', '{"repetitions": 3}']
    ok = T.check_text("ASR was 0.523 (0.401 to 0.640), cost -0.023, over 3 Repetitions.", sources)
    assert ok.ok and ok.untraceable == ()
    bad = T.check_text("ASR was 0.52 and 0.523 and 0.52 again; 7 runs.", sources)
    assert not bad.ok and bad.untraceable == ("0.52", "7")


def test_check_files_and_cli(tmp_path, capsys):
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "summary.md").write_text("| 0.523 [0.401, 0.640] |\n")
    (tmp_path / "results" / "x.png").write_bytes(b"\x89PNG 0.777")
    (tmp_path / "README.md").write_text("ASR 0.523 [0.401, 0.640].\n")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "REPORT.md").write_text("ASR 0.777.\n")
    res = T.check_files([tmp_path / "README.md", tmp_path / "docs" / "REPORT.md", tmp_path / "missing.md"],
                        tmp_path / "results")
    assert res[str(tmp_path / "README.md")].ok
    assert res[str(tmp_path / "docs" / "REPORT.md")].untraceable == ("0.777",)  # PNG bytes are not a source
    assert res[str(tmp_path / "missing.md")] is None
    assert CLI.main([], root=tmp_path) == 1
    out = capsys.readouterr().out
    assert "docs/REPORT.md: UNTRACEABLE number 0.777" in out and "README.md: every number traces" in out
    assert CLI.main(["--doc", "README.md", "--doc", "missing.md"], root=tmp_path) == 0
    assert "missing.md: not found (skipped)" in capsys.readouterr().out


present = st.sampled_from(["0.523", "0.401", "0.640", "-0.023", "3", "40", "2880", "0.125"])
absent = st.sampled_from(["0.999", "0.52", "12", "-0.5", "77", "1.5"])
filler = st.sampled_from(["ASR", "was", "and", "the", "rate", "qwen3-4b", "2026-10-17", "v0.1.35", "M2", "`--seed 5`",
                          "results/plots/a1.png", "(", ")", ";"])


# Feature: agent-injection-bench, Property 11: README/Report number traceability
@settings(max_examples=100, derandomize=True)
@given(words=st.lists(st.one_of(st.tuples(st.just("p"), present), st.tuples(st.just("a"), absent),
                                st.tuples(st.just("f"), filler)), max_size=30))
def test_property_11_traceability(words):
    sources = ["| 0.523 [0.401, 0.640] | -0.023 | 3 | 40 |", '{"n": 2880, "price": 0.125}']
    text = " ".join(w for _, w in words)
    expected = []
    for kind, w in words:
        if kind == "a" and w not in expected:
            expected.append(w)
    result = T.check_text(text, sources)
    assert result.untraceable == tuple(expected)
    assert result.ok == (not expected)
