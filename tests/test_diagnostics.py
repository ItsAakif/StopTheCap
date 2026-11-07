"""Diagnostics: per-source, per-variant and dataset-fingerprint reporting."""

from __future__ import annotations

from ml.audit.diagnostics import run_diagnostics
from ml.ingest.schema import Split
from tests.conftest import text_record


def _corpus():
    recs = []
    for i in range(320):
        split = Split.TRAIN if i < 200 else Split.TEST_ID
        recs.append(text_record(f"h{i}", f"human passage {i} about rivers " * 5, 0,
                                group_id=f"g{i}", split=split))
        recs.append(text_record(f"m{i}", f"machine passage {i} about mountains " * 5, 1,
                                group_id=f"g{i}", split=split, generator="gpt"))
    return recs


def test_diagnostics_returns_per_source_entry():
    diag = run_diagnostics(_corpus())
    assert any(s.source == "ghostbuster" for s in diag.sources)
    src = next(s for s in diag.sources if s.source == "ghostbuster")
    assert src.split_basis == "project_split"
    assert src.char_auc is not None


def test_diagnostics_reports_variants():
    diag = run_diagnostics(_corpus())
    assert diag.variants, "expected at least one variant diagnostic"
    assert all(v.variant for v in diag.variants)


def test_diagnostics_handles_single_class_source():
    recs = [
        text_record(f"h{i}", "human text " * 10, 0, group_id=f"g{i}",
                    split=Split.QUARANTINE, dataset="raid")
        for i in range(10)
    ]
    diag = run_diagnostics(recs)
    assert diag.sources
    assert diag.sources[0].char_auc is None


def test_quarantine_only_source_uses_internal_diagnostic_basis():
    recs = []
    for i in range(60):
        recs.append(text_record(f"r{i}", f"human wire {i} " * 8, 0, group_id=f"g{i}",
                                split=Split.QUARANTINE, dataset="raid"))
        recs.append(text_record(f"q{i}", f"machine wire {i} " * 8, 1, group_id=f"g{i}",
                                split=Split.QUARANTINE, dataset="raid", generator="gpt"))
    diag = run_diagnostics(recs)
    src = next(s for s in diag.sources if s.source == "raid")
    assert src.split_basis == "internal_diagnostic"


def test_diagnostics_serialises():
    payload = run_diagnostics(_corpus()).to_dict()
    assert {"sources", "variants", "dataset_fingerprints", "notes"} <= set(payload)
