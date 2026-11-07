"""Audit gates: they must PASS on a clean corpus and FAIL on a contaminated one.

The negative tests are the important ones. An audit that cannot fail is decoration.
"""

from __future__ import annotations

from ml.audit.audit import run_audit
from ml.ingest.schema import Split
from tests.conftest import text_record, video_record


def _clean_split_corpus(n=60):
    recs = []
    for i in range(n):
        split = Split.TRAIN if i < n * 0.7 else Split.TEST_ID
        recs.append(text_record(f"h{i}", f"human passage {i} with ordinary wording " * 6,
                                0, group_id=f"g{i}", split=split))
        recs.append(text_record(f"m{i}", f"machine passage {i} with different wording " * 6,
                                1, group_id=f"g{i}", split=split, generator="gpt"))
    return recs


def test_clean_corpus_passes():
    res = run_audit(_clean_split_corpus(), duplicates=[])
    assert res.gates["unique_sample_ids"]
    assert res.gates["required_provenance"]
    assert res.gates["no_group_split_overlap"]
    assert res.data["split_digest_recomputed"]


def test_detects_duplicate_sample_ids():
    recs = _clean_split_corpus(10)
    dup = recs[0].model_copy(update={"split": Split.TRAIN})
    res = run_audit(recs + [dup], duplicates=[])
    assert not res.gates["unique_sample_ids"]


def test_detects_missing_provenance():
    recs = _clean_split_corpus(10)
    recs[0] = recs[0].model_copy(update={"licence_ref": ""})
    res = run_audit(recs, duplicates=[])
    assert not res.gates["required_provenance"]


def test_detects_group_spanning_splits():
    recs = _clean_split_corpus(10)
    recs[0] = recs[0].model_copy(update={"group_id": "shared", "split": Split.TRAIN})
    recs.append(text_record("x", "other text " * 8, 1, group_id="shared",
                            split=Split.TEST_ID, generator="gpt"))
    res = run_audit(recs, duplicates=[])
    assert not res.gates["no_group_split_overlap"]


def test_detects_eval_only_in_training():
    recs = _clean_split_corpus(10)
    recs[0] = recs[0].model_copy(
        update={"source_dataset": "raid", "split": Split.TRAIN})
    res = run_audit(recs, duplicates=[])
    assert not res.gates["eval_only_not_trained"]


def test_single_feature_artifact_gate_catches_one_character_leak():
    """Reproduces the seed project's failure mode: a marker in one class only."""
    recs = []
    n = 100
    for i in range(n):
        split = Split.TRAIN if i < 30 else Split.TEST_ID
        # human texts all contain an asterisk, machine texts never do
        recs.append(text_record(f"h{i}", f"* human passage {i} " * 6, 0,
                                group_id=f"g{i}", split=split))
        recs.append(text_record(f"m{i}", f"machine passage {i} " * 6, 1,
                                group_id=f"g{i}", split=split, generator="gpt"))
    res = run_audit(recs, duplicates=[])
    assert not res.gates["shortcut_single_feature"]
    assert any("single-feature" in n_ for n_ in res.notes)


def test_single_feature_gate_passes_when_no_feature_dominates():
    recs = []
    n = 100
    for i in range(n):
        split = Split.TRAIN if i < 30 else Split.TEST_ID
        recs.append(text_record(f"h{i}", f"human passage {i} about rivers " * 6, 0,
                                group_id=f"g{i}", split=split))
        recs.append(text_record(f"m{i}", f"machine passage {i} about mountains " * 6, 1,
                                group_id=f"g{i}", split=split, generator="gpt"))
    res = run_audit(recs, duplicates=[])
    assert res.gates["shortcut_single_feature"]


def test_detects_cross_split_duplicates():
    from ml.data.dedup import DuplicatePair

    recs = _clean_split_corpus(10)
    a = text_record("d1", "same text " * 10, 0, group_id="g1", split=Split.TRAIN)
    b = text_record("d2", "same text " * 10, 0, group_id="g2", split=Split.TEST_ID)
    recs = recs + [a, b]
    idx_a, idx_b = len(recs) - 2, len(recs) - 1
    res = run_audit(recs, duplicates=[DuplicatePair(idx_a, idx_b, 1.0, "exact")])
    assert not res.gates["no_cross_split_duplicates"]


def test_video_records_checked_separately():
    recs = _clean_split_corpus(10)
    v = video_record("v1", 1, group_id="vg", identity_group="sora", split=Split.TRAIN)
    res = run_audit(recs + [v], duplicates=[])
    assert res.gates["video_media_absent_marked"]
    assert res.data["video_records"] == 1


def test_determinism_gate_rejects_mismatched_digest():
    recs = _clean_split_corpus(10)
    res = run_audit(recs, duplicates=[], expected_split_digest="deadbeef")
    assert not res.gates["split_deterministic"]
