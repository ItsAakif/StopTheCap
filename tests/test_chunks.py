"""Chunk planning, resume semantics and frame sampling (offline-testable)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ml.video.chunks import Chunk, ChunkStore, frame_indices, plan_chunks, stable_hash


def test_plan_is_deterministic_and_order_independent():
    a = plan_chunks(["c", "a", "b", "d"], "frames", size=2)
    b = plan_chunks(["d", "b", "a", "c"], "frames", size=2)
    assert [c.key for c in a] == [c.key for c in b]
    assert [c.item_ids for c in a] == [("a", "b"), ("c", "d")]


def test_plan_hash_changes_when_items_change():
    a = plan_chunks(["a", "b"], "frames", size=2)
    b = plan_chunks(["a", "c"], "frames", size=2)
    assert a[0].plan_hash != b[0].plan_hash


def test_chunk_ids_are_zero_padded_and_stable():
    chunks = plan_chunks([str(i) for i in range(5)], "gen", size=2)
    assert [c.chunk_id for c in chunks] == ["gen-00000", "gen-00001", "gen-00002"]


def test_empty_plan():
    assert plan_chunks([], "frames", size=10) == []


def test_invalid_size():
    with pytest.raises(ValueError):
        plan_chunks(["a"], "frames", size=0)


def test_marker_roundtrip_and_pending(tmp_path: Path):
    chunks = plan_chunks(["a", "b", "c", "d"], "frames", size=2)
    store = ChunkStore(tmp_path)
    art = tmp_path / "out.parquet"
    art.write_bytes(b"payload")

    assert len(store.pending(chunks)) == 2
    store.mark_done(chunks[0], art)
    assert store.is_done(chunks[0], art)
    assert len(store.pending(chunks)) == 1
    assert store.summary(chunks)["done"] == 1


def test_truncated_artefact_is_not_done(tmp_path: Path):
    """An interrupted write must never be mistaken for a finished chunk."""
    chunk = plan_chunks(["a"], "frames", size=1)[0]
    store = ChunkStore(tmp_path)
    art = tmp_path / "out.parquet"
    art.write_bytes(b"complete-payload")
    store.mark_done(chunk, art)

    art.write_bytes(b"trunc")          # simulate a cut-off write
    assert store.is_done(chunk, art) is False


def test_missing_artefact_is_not_done(tmp_path: Path):
    chunk = plan_chunks(["a"], "frames", size=1)[0]
    store = ChunkStore(tmp_path)
    art = tmp_path / "out.parquet"
    art.write_bytes(b"x")
    store.mark_done(chunk, art)
    art.unlink()
    assert store.is_done(chunk, art) is False


def test_corrupt_marker_is_ignored(tmp_path: Path):
    chunk = plan_chunks(["a"], "frames", size=1)[0]
    store = ChunkStore(tmp_path)
    art = tmp_path / "out.parquet"
    art.write_bytes(b"x")
    p = store.mark_done(chunk, art)
    p.write_text("{not json", encoding="utf-8")
    assert store.is_done(chunk, art) is False


def test_plan_change_invalidates_markers(tmp_path: Path):
    store = ChunkStore(tmp_path)
    art = tmp_path / "out.parquet"
    art.write_bytes(b"x")
    old = plan_chunks(["a", "b"], "frames", size=2)[0]
    store.mark_done(old, art)

    new = plan_chunks(["a", "b", "c"], "frames", size=2)[0]
    assert store.is_done(new, art) is False


def test_frame_indices_are_in_range_and_unique():
    idx = frame_indices(100, 8)
    assert len(idx) == 8
    assert idx == sorted(idx)
    assert len(set(idx)) == len(idx)
    assert all(0 <= i < 100 for i in idx)


def test_frame_indices_handle_short_clips():
    assert frame_indices(3, 8) == [0, 1, 2]
    assert frame_indices(1, 8) == [0]
    assert frame_indices(0, 8) == []


def test_frame_indices_single_sample_is_central():
    assert frame_indices(100, 1) == [50]


def test_frame_indices_rejects_unknown_mode():
    with pytest.raises(ValueError):
        frame_indices(100, 4, mode="random")


def test_stable_hash_is_short_and_stable():
    assert stable_hash("a", "b") == stable_hash("a", "b")
    assert stable_hash("a", "b") != stable_hash("a", "c")
    assert len(stable_hash("a", 1, length=8)) == 8
