"""Pilot manifest, category matching, grouped splits and robustness transforms."""

from __future__ import annotations

import numpy as np
import pytest

from ml.video.pilot import (
    CATEGORIES,
    ClipRecord,
    assign_splits,
    build_record,
    compression_variant,
    leakage_report,
    match_categories,
    resize_variant,
    validate_all,
)

LIC = "pexels:https://www.pexels.com/license/"
LIC_GEN = "apache-2.0:https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B"


def rec(cid, label, cat, group=None, gen=None):
    return ClipRecord(clip_id=cid, label=label,
                      source="pexels" if label == 0 else "wan2.1-t2v-1.3b",
                      group_id=group or cid, category=cat, path=f"/tmp/{cid}.mp4",
                      licence_ref=LIC if label == 0 else LIC_GEN,
                      generator=gen)


def test_every_category_is_valid():
    for c in CATEGORIES:
        assert rec(f"c-{c}", 0, c).validate() == []


def test_unknown_category_rejected():
    assert rec("x", 0, "nonsense").validate()


def test_generated_requires_generator():
    assert rec("g", 1, "nature", gen=None).validate()
    assert rec("g", 1, "nature", gen="wan2.1").validate() == []


def test_missing_licence_rejected():
    r = rec("m", 0, "nature")
    r.licence_ref = ""
    assert r.validate()


def test_duplicate_ids_detected():
    a, b = rec("same", 0, "nature"), rec("same", 1, "nature", gen="wan")
    rep = validate_all([a, b])
    assert not rep["ok"] and any("duplicate" in p for p in rep["problems"])


def test_category_matching_is_balanced_and_reported():
    real = [rec(f"r{i}", 0, "nature") for i in range(5)] + [rec("r-city", 0, "city")]
    gen = [rec(f"g{i}", 1, "nature", gen="wan") for i in range(3)]
    pairs, report = match_categories(real, gen)
    assert len(pairs) == 3                      # limited by the smaller side
    assert report["nature"]["paired"] == 3
    assert report["nature"]["dropped"] == 2     # two real clips unmatched
    assert report["city"]["paired"] == 0


def test_split_assignment_is_deterministic():
    records = [rec(f"r{i}", 0, "nature", group=f"src{i}") for i in range(30)]
    a = assign_splits(records)
    b = assign_splits(list(reversed(records)))
    by_id_a = {r.clip_id: r.split for r in a}
    by_id_b = {r.clip_id: r.split for r in b}
    assert by_id_a == by_id_b


def test_group_members_share_a_split():
    records = [rec("a", 0, "nature", group="video-1"),
               rec("b", 0, "nature", group="video-1"),
               rec("c", 1, "nature", group="gen-1", gen="wan")]
    out = assign_splits(records)
    splits = {r.clip_id: r.split for r in out}
    assert splits["a"] == splits["b"]


def test_leakage_report_is_clean_for_grouped_splits():
    records = [rec(f"r{i}", 0, "nature", group=f"src{i}") for i in range(20)]
    out = assign_splits(records)
    rep = leakage_report(out)
    assert rep["leaking_groups"] == 0


def test_leakage_report_detects_a_spanning_group():
    a = rec("a", 0, "nature", group="g"); a.split = "train"
    b = rec("b", 0, "nature", group="g"); b.split = "test"
    rep = leakage_report([a, b])
    assert rep["leaking_groups"] == 1


def test_compression_variant_preserves_shape():
    frames = np.random.default_rng(0).integers(0, 255, (3, 32, 32, 3), dtype=np.uint8)
    out = compression_variant(frames, quality=30)
    assert out.shape == frames.shape and out.dtype == np.uint8


def test_resize_variant_preserves_shape():
    frames = np.random.default_rng(0).integers(0, 255, (2, 64, 64, 3), dtype=np.uint8)
    out = resize_variant(frames, size=16)
    assert out.shape == frames.shape


def test_build_record_hashes_existing_file(tmp_path):
    p = tmp_path / "clip.mp4"
    p.write_bytes(b"not really a video")
    r = build_record("c1", 0, "pexels", "src1", "nature", p, LIC)
    assert r.sha256 and len(r.sha256) == 64
