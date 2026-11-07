"""Split assignment: determinism, group integrity, and role enforcement."""

from __future__ import annotations

from ml.data.splits import (
    DEFAULT_SALT,
    apply_plan,
    assert_plan_invariants,
    build_split_plan,
    bucket_for,
    split_counts,
    split_digest,
)
from ml.ingest.schema import Split
from tests.conftest import text_record


def _corpus(n=80, dataset="ghostbuster", eval_only=False):
    recs = []
    for i in range(n):
        recs.append(text_record(f"h{i}", f"human text {i} " * 8, 0, group_id=f"g{i}",
                                dataset=dataset, eval_only=eval_only))
        recs.append(text_record(f"m{i}", f"machine text {i} " * 8, 1, group_id=f"g{i}",
                                dataset=dataset, generator="gpt", eval_only=eval_only))
    return recs


def test_bucket_is_deterministic_and_bounded():
    a = bucket_for("group-1", DEFAULT_SALT)
    assert a == bucket_for("group-1", DEFAULT_SALT)
    assert 0 <= a < 100


def test_salt_changes_assignment():
    # different salts must produce different buckets for at least some groups
    diffs = sum(1 for i in range(50)
                if bucket_for(f"g{i}", "salt-a") != bucket_for(f"g{i}", "salt-b"))
    assert diffs > 0


def test_plan_is_reproducible():
    recs = _corpus()
    p1 = build_split_plan(recs, {"ghostbuster"})
    p2 = build_split_plan(recs, {"ghostbuster"})
    assert p1.digest() == p2.digest()


def test_apply_plan_keeps_groups_together():
    recs = _corpus()
    plan = build_split_plan(recs, {"ghostbuster"})
    out = apply_plan(recs, plan)
    assert assert_plan_invariants(out) == []
    # human and machine of one group must land in the same split
    by_group: dict[str, set[str]] = {}
    for r in out:
        by_group.setdefault(r.group_id, set()).add(r.split.value)
    assert all(len(v) == 1 for v in by_group.values())


def test_eval_only_dataset_is_quarantined():
    recs = _corpus(dataset="raid", eval_only=True)
    plan = build_split_plan(recs, train_allowed_datasets=set())
    out = apply_plan(recs, plan)
    assert all(r.split is Split.QUARANTINE for r in out)


def test_trainable_dataset_produces_all_splits():
    recs = _corpus(n=200)
    plan = build_split_plan(recs, {"ghostbuster"})
    out = apply_plan(recs, plan)
    counts = split_counts(out)
    assert counts.get("train", 0) > 0
    assert counts.get("val", 0) > 0
    assert counts.get("test_id", 0) > 0


def test_short_records_excluded():
    recs = [
        text_record("short", "too few words", 0, word_count=3),
        text_record("ok", "human text " * 20, 0),
    ]
    recs[0].metadata["quality_flags"] = ["too_short"]
    plan = build_split_plan(recs, {"ghostbuster"})
    out = apply_plan(recs, plan)
    short = next(r for r in out if r.sample_id.startswith("short"))
    assert short.split is Split.EXCLUDED


def test_split_digest_detects_change():
    recs = _corpus()
    plan = build_split_plan(recs, {"ghostbuster"})
    out = apply_plan(recs, plan)
    d1 = split_digest(out, DEFAULT_SALT)
    mutated = [r.model_copy(update={"split": Split.TRAIN}) for r in out]
    assert split_digest(mutated, DEFAULT_SALT) != d1


def test_assert_plan_invariants_detects_spanning_group():
    a = text_record("a", "some human text here " * 5, 0, group_id="shared",
                    split=Split.TRAIN)
    b = text_record("b", "some machine text here " * 5, 1, group_id="shared",
                    split=Split.TEST_ID, generator="gpt")
    problems = assert_plan_invariants([a, b])
    assert problems
    assert "shared" in problems[0]


def test_explicit_exclusion_is_not_overwritten_by_split_assignment():
    """Regression: duplicate detection marks a record EXCLUDED, and the split pass
    used to silently reassign it, resurrecting duplicates into training."""
    recs = _corpus(20)
    victim = recs[0].model_copy(
        update={"metadata": {**recs[0].metadata, "exclusion_reason": "duplicate_of:x"}})
    recs[0] = victim
    plan = build_split_plan(recs, {"ghostbuster"})
    out = apply_plan(recs, plan)
    got = next(r for r in out if r.sample_id == victim.sample_id)
    assert got.split is Split.EXCLUDED


def test_eval_only_record_in_plan_is_quarantined_not_split():
    recs = _corpus(20)
    recs[0] = recs[0].model_copy(
        update={"metadata": {**recs[0].metadata, "_eval_only": True}})
    plan = build_split_plan(recs, {"ghostbuster"})
    out = apply_plan(recs, plan)
    got = next(r for r in out if r.sample_id == recs[0].sample_id)
    assert got.split is not Split.TRAIN
