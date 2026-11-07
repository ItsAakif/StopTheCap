"""Decontamination matching: exact, near, and the cases that must not match."""

from __future__ import annotations

import pytest

from ml.data.decontaminate import (
    NEAR_DUP_THRESHOLD,
    build_reference_index,
    classify,
)

# Long enough that a single-word edit still lands above the 0.85 Jaccard threshold: with ~120
# words a one-word change alters ~5 of ~116 shingles.
REF = " ".join([
    "In most of the developed world tap water is just as safe as bottled water",
    "because regulators set enforceable limits on contaminants and suppliers test against",
    "them regularly throughout the year using accredited laboratories and published methods",
    "so the results can be compared across regions and over time by independent bodies",
    "and any exceedance must be reported to the authority responsible for enforcement",
    "which may then require remedial action and inform the public about the finding",
])

REF2 = ("The Prime Meridian runs through Greenwich because the Royal Observatory was the "
        "world's leading astronomical institution when the international convention was agreed "
        "and most maritime charts of the period already used it as their reference meridian "
        "which made the choice practical rather than purely political at the time")


@pytest.fixture(scope="module")
def index():
    return build_reference_index([
        ("hc3:1", REF),
        ("ghostbuster:2", REF2),
        ("raid:3", "Some entirely unrelated reference text about maritime navigation and tides."),
    ])


def test_exact_match_is_detected(index):
    reason, matched, j = classify(REF, index)
    assert reason == "exact"
    assert matched == "hc3:1"
    assert j == 1.0


def test_exact_match_survives_whitespace(index):
    variant = "   " + REF.replace(" safe ", "  safe  ") + "  "
    reason, matched, _ = classify(variant, index)
    assert reason == "exact"
    assert matched == "hc3:1"


def test_near_duplicate_is_detected(index):
    edited = REF.replace("just as safe", "just as safe")   # identity guard
    edited = REF.replace("accredited", "approved")
    assert edited != REF
    reason, matched, j = classify(edited, index)
    assert reason == "near"
    assert matched == "hc3:1"
    assert j >= NEAR_DUP_THRESHOLD


def test_clean_text_is_not_flagged(index):
    novel = ("A completely different passage about compiler optimisation, register allocation "
             "and the trade-offs of instruction scheduling in modern superscalar processors, "
             "written from scratch for this test and sharing no phrasing with the references")
    reason, _, _ = classify(novel, index)
    assert reason is None


def test_empty_text_is_flagged_separately(index):
    reason, matched, _ = classify("   \n  ", index)
    assert reason == "empty"
    assert matched is None


def test_labels_identify_the_reference_dataset(index):
    assert classify(REF2, index)[1].startswith("ghostbuster:")
    assert classify(REF, index)[1].startswith("hc3:")


def test_index_records_size(index):
    assert index.size == 3


def test_threshold_is_the_documented_value():
    assert NEAR_DUP_THRESHOLD == 0.85
