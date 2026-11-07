"""Video pipeline plumbing. Skipped automatically when ffmpeg is unavailable."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from ml.video.pipeline import (
    ClipItem,
    basic_frame_features,
    extract_frames,
    ffmpeg_available,
    make_synthetic_clip,
    probe_clip,
    run_features_stage,
    run_frames_stage,
)

pytestmark = pytest.mark.skipif(not ffmpeg_available(), reason="ffmpeg/ffprobe not installed")

N_FRAMES = 4
SIZE = 64


@pytest.fixture(scope="module")
def clip(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("clips")
    return make_synthetic_clip(d / "moving.mp4", "moving", seconds=1, fps=8, size=SIZE)


def test_synthetic_clip_is_created(clip: Path):
    assert clip.is_file() and clip.stat().st_size > 0


def test_probe_reads_metadata(clip: Path):
    meta = probe_clip(clip)
    assert meta.width == SIZE and meta.height == SIZE
    assert meta.fps == pytest.approx(8.0, abs=0.5)
    assert meta.duration_s and meta.duration_s > 0
    assert len(meta.sha256) == 64


def test_extract_frames_shape_and_dtype(clip: Path):
    frames = extract_frames(clip, n_frames=N_FRAMES, size=SIZE)
    assert frames.shape == (N_FRAMES, SIZE, SIZE, 3)
    assert frames.dtype == np.uint8


def test_extract_frames_is_deterministic(clip: Path):
    a = extract_frames(clip, n_frames=N_FRAMES, size=SIZE)
    b = extract_frames(clip, n_frames=N_FRAMES, size=SIZE)
    assert np.array_equal(a, b)


def test_basic_features_one_vector_per_clip(clip: Path):
    frames = extract_frames(clip, n_frames=N_FRAMES, size=SIZE)
    v = basic_frame_features(frames)          # 4-D in, single vector out
    assert v.ndim == 1 and v.size > 0


def test_frames_stage_writes_ids_and_resumes(clip: Path, tmp_path: Path):
    items = [ClipItem("c1", clip, "g1", 1), ClipItem("c2", clip, "g2", 0)]
    out = tmp_path / "frames"
    first = run_frames_stage(items, out, chunk_size=2, n_frames=N_FRAMES, size=SIZE)
    assert len(first) == 1
    side = out / "frames-00000.ids.json"
    assert json.loads(side.read_text()) == ["c1", "c2"]

    second = run_frames_stage(items, out, chunk_size=2, n_frames=N_FRAMES, size=SIZE)
    assert second == [], "completed chunks must not be redone"


def test_features_stage_produces_one_row_per_clip(clip: Path, tmp_path: Path):
    items = [ClipItem("c1", clip, "g1", 1), ClipItem("c2", clip, "g2", 0)]
    fdir = tmp_path / "frames"
    run_frames_stage(items, fdir, chunk_size=2, n_frames=N_FRAMES, size=SIZE)
    feats = run_features_stage(fdir, tmp_path / "features")
    assert len(feats) == 1
    payload = np.load(feats[0].artefact, allow_pickle=False)
    assert payload["X"].shape[0] == 2          # two clips, not eight frames
    assert list(payload["ids"]) == ["c1", "c2"]


def test_truncated_frames_are_redone(clip: Path, tmp_path: Path):
    items = [ClipItem("c1", clip, "g1", 1)]
    out = tmp_path / "frames"
    run_frames_stage(items, out, chunk_size=1, n_frames=N_FRAMES, size=SIZE)
    (out / "frames-00000.npy").write_bytes(b"corrupt")
    again = run_frames_stage(items, out, chunk_size=1, n_frames=N_FRAMES, size=SIZE)
    assert len(again) == 1, "a corrupted artefact must be regenerated"
