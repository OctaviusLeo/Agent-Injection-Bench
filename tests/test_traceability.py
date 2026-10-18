"""Unit tests for the number-traceability check (aib.traceability; Req 7.4, 9.3, 9.4).

Covers extraction, normalization, the traceable/untraceable verdict, and the
file-based wrapper.
"""

from __future__ import annotations

import os

from aib.traceability import (
    check_files,
    check_text,
    extract_numbers,
    normalize_number,
)


def test_extract_numbers_handles_signs_commas_decimals():
    nums = extract_numbers("ASR 0.500 cost -0.125 n=1,024 and +3")
    assert nums == ["0.500", "-0.125", "1,024", "+3"]


def test_normalize_folds_cosmetic_variants():
    assert normalize_number("0.5") == normalize_number("0.50")
    assert normalize_number("0.5") == normalize_number("+0.500")
    assert normalize_number("1,024") == normalize_number("1024")
    assert normalize_number("-0") == normalize_number("0")


def test_normalize_rejects_non_numbers():
    assert normalize_number("+") is None
    assert normalize_number(".") is None
    assert normalize_number("abc") is None


def test_traceable_numbers_pass():
    source = "Results_Table: 0.500 and 0.125 reported."
    doc = "The ASR was 0.5 with a cost of 0.125."
    result = check_text(doc, [source])
    assert result.ok is True
    assert result.untraceable == ()


def test_untraceable_number_fails_and_is_reported():
    source = "Results_Table: 0.500 reported."
    doc = "The ASR was 0.500 but the headline figure was 0.999."
    result = check_text(doc, [source])
    assert result.ok is False
    assert "0.999" in result.untraceable
    assert "0.500" not in result.untraceable


def test_duplicate_untraceable_reported_once():
    result = check_text("saw 0.42 then 0.42 again", ["nothing here"])
    assert result.untraceable == ("0.42",)


def test_check_files_reads_disk(tmp_path):
    doc = os.path.join(str(tmp_path), "README.md")
    src = os.path.join(str(tmp_path), "summary.md")
    with open(src, "w", encoding="utf-8") as fh:
        fh.write("| ASR | 0.750 |")
    with open(doc, "w", encoding="utf-8") as fh:
        fh.write("The ASR was 0.75.")
    assert check_files(doc, [src]).ok is True


def test_check_files_missing_source_makes_number_untraceable(tmp_path):
    doc = os.path.join(str(tmp_path), "README.md")
    with open(doc, "w", encoding="utf-8") as fh:
        fh.write("The ASR was 0.75.")
    result = check_files(doc, [os.path.join(str(tmp_path), "absent.md")])
    assert result.ok is False
    assert "0.75" in result.untraceable
