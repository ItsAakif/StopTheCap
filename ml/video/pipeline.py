"""Video pipeline: decode → frames → features → split, with chunked resume.

Scope: this validates the *plumbing*, not detection. It deliberately has no opinion about
whether a video is real or generated — that is the model's job, and no model has been trained.

Design:

* Decoding uses **ffmpeg via subprocess**. No OpenCV, decord or PyAV dependency is introduced;
  ffmpeg is already present on the development machine and on Colab/Kaggle images.
* Frame extraction is a pure function of (path, n_frames, size), so it is reproducible and
  testable without any real media.
* Features are pluggable. The default is a cheap deterministic descriptor so the pipeline can be
  exercised end-to-end without downloading a backbone. A CLIP-based extractor plugs into the same
  interface later.
* Every stage writes one artefact per chunk plus a marker (see ``ml.video.chunks``), so a
  terminated Colab or Kaggle session resumes without redoing completed work.

**No restricted media is ever downloaded or redistributed by this module.**
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np

from .chunks import Chunk, ChunkStore, frame_indices, plan_chunks

FFMPEG = "ffmpeg"
FFPROBE = "ffprobe"


# --------------------------------------------------------------------------- probe


@dataclass
class ClipMeta:
    path: str
    duration_s: float | None
    fps: float | None
    width: int | None
    height: int | None
    n_frames_estimate: int | None
    sha256: str


def ffmpeg_available() -> bool:
    return shutil.which(FFMPEG) is not None and shutil.which(FFPROBE) is not None


def probe_clip(path: Path, timeout: int = 30) -> ClipMeta:
    """Read duration/fps/dimensions. Missing metadata stays None rather than being guessed."""
    sha = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            sha.update(block)

    duration = fps = width = height = None
    n_est = None
    if ffmpeg_available():
        try:
            out = subprocess.run(
                [FFPROBE, "-v", "error", "-select_streams", "v:0",
                 "-show_entries", "stream=width,height,r_frame_rate,duration",
                 "-show_entries", "format=duration",
                 "-of", "json", str(path)],
                capture_output=True, text=True, timeout=timeout, check=False,
            )
            data = json.loads(out.stdout or "{}")
            streams = data.get("streams") or [{}]
            st = streams[0]
            width = st.get("width")
            height = st.get("height")
            rate = st.get("r_frame_rate") or ""
            if "/" in rate:
                num, den = rate.split("/", 1)
                fps = float(num) / float(den) if float(den) else None
            duration = st.get("duration") or (data.get("format") or {}).get("duration")
            duration = float(duration) if duration else None
            if duration and fps:
                n_est = int(round(duration * fps))
        except (subprocess.SubprocessError, ValueError, json.JSONDecodeError):
            pass

    return ClipMeta(path=str(path), duration_s=duration, fps=fps, width=width,
                    height=height, n_frames_estimate=n_est, sha256=sha.hexdigest())


# --------------------------------------------------------------------------- frames


def extract_frames(path: Path, n_frames: int = 12, size: int = 224,
                   timeout: int = 120) -> np.ndarray:
    """Decode ``n_frames`` uniformly spaced RGB frames as uint8 (n, size, size, 3).

    Uses ffmpeg's ``fps`` filter to resample the whole clip to ``n_frames`` frames, which is
    simpler and more deterministic than seeking, and avoids depending on a decoder library.
    """
    if not ffmpeg_available():
        raise RuntimeError("ffmpeg/ffprobe not found on PATH")

    with tempfile.TemporaryDirectory() as td:
        pattern = str(Path(td) / "f_%04d.png")
        cmd = [
            FFMPEG, "-v", "error", "-y", "-i", str(path),
            "-vf", f"fps={n_frames},scale={size}:{size}:force_original_aspect_ratio=decrease,"
                   f"pad={size}:{size}:(ow-iw)/2:(oh-ih)/2",
            "-frames:v", str(n_frames), pattern,
        ]
        subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)

        files = sorted(Path(td).glob("f_*.png"))
        if not files:
            raise RuntimeError(f"no frames decoded from {path.name}")

        from PIL import Image

        arrs = [np.asarray(Image.open(f).convert("RGB"), dtype=np.uint8) for f in files]
        arr = np.stack(arrs)
        if arr.shape[0] < n_frames:  # pad by repeating the last frame
            pad = np.repeat(arr[-1:], n_frames - arr.shape[0], axis=0)
            arr = np.concatenate([arr, pad], axis=0)
        return arr[:n_frames]


def frame_indices_for_clip(meta: ClipMeta, n_frames: int) -> list[int]:
    """Deterministic sample positions, in source-frame coordinates."""
    total = meta.n_frames_estimate or max(n_frames, 1)
    return frame_indices(total, n_frames)


# --------------------------------------------------------------------------- features


def basic_frame_features(frames: np.ndarray) -> np.ndarray:
    """Cheap deterministic descriptor: per-frame colour stats + temporal deltas.

    This is a *placeholder* feature extractor. It exists so the pipeline can run end-to-end
    without a backbone. It is not a detector and must not be reported as one.
    """
    x = frames.astype(np.float32) / 255.0
    mean = x.mean(axis=(1, 2))                       # (n, 3)
    std = x.std(axis=(1, 2))                         # (n, 3)
    if len(x) > 1:
        delta = np.abs(np.diff(x, axis=0)).mean(axis=(1, 2, 3))
        temporal = np.array([delta.mean(), delta.std(), delta.max()])
    else:
        temporal = np.zeros(3, dtype=np.float32)
    pooled = np.concatenate([mean.mean(axis=0), std.mean(axis=0), temporal])
    return pooled.astype(np.float32)


FeatureFn = Callable[[np.ndarray], np.ndarray]


# --------------------------------------------------------------------------- stages


@dataclass
class StageResult:
    stage: str
    chunk_id: str
    ids: tuple[str, ...]
    artefact: Path
    seconds: float
    bytes: int

    @property
    def n_items(self) -> int:
        return len(self.ids)

    @property
    def items_per_s(self) -> float:
        return self.n_items / self.seconds if self.seconds > 0 else float("inf")


@dataclass
class ClipItem:
    clip_id: str
    path: Path
    group_id: str
    label: int
    meta: dict = field(default_factory=dict)


def run_frames_stage(items: Sequence[ClipItem], outdir: Path, stage: str = "frames",
                     chunk_size: int = 20, n_frames: int = 12, size: int = 224,
                     store: ChunkStore | None = None) -> list[StageResult]:
    """Decode each clip into a stacked uint8 array; one .npy per chunk, resumable.

    Each chunk also writes a small ``.ids.json`` sidecar so the next stage knows which clip
    each row belongs to without re-deriving the plan.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    store = store or ChunkStore(outdir / "_markers")
    chunks = plan_chunks([i.clip_id for i in items], stage, size=chunk_size)
    by_id = {i.clip_id: i for i in items}

    def artefact_for(c: Chunk) -> Path:
        return outdir / f"{c.stage}-{c.index:05d}.npy"

    results: list[StageResult] = []
    for chunk in store.pending(chunks, artefact_for=artefact_for):
        t0 = time.perf_counter()
        arrays = [extract_frames(by_id[cid].path, n_frames=n_frames, size=size)
                  for cid in chunk.item_ids]
        arr = np.stack(arrays)
        out = artefact_for(chunk)
        np.save(out, arr)
        out.with_suffix(".ids.json").write_text(
            json.dumps(list(chunk.item_ids)), encoding="utf-8")
        dt = time.perf_counter() - t0
        store.mark_done(chunk, out, {"n_frames": n_frames, "size": size})
        results.append(StageResult(stage, chunk.chunk_id, tuple(chunk.item_ids),
                                   out, dt, out.stat().st_size))
    return results


def run_features_stage(frames_dir: Path, outdir: Path, stage: str = "features",
                       feature_fn: FeatureFn = basic_frame_features,
                       store: ChunkStore | None = None) -> list[StageResult]:
    """Reduce each chunk's frames to **one feature vector per clip**."""
    outdir.mkdir(parents=True, exist_ok=True)
    store = store or ChunkStore(outdir / "_markers")

    frame_files = sorted(frames_dir.glob("frames-*.npy"))
    ids = [p.stem for p in frame_files]
    chunks = plan_chunks(ids, stage, size=1)  # one chunk per frames file

    def artefact_for(c: Chunk) -> Path:
        return outdir / f"{c.stage}-{c.index:05d}.npz"

    results: list[StageResult] = []
    for chunk in store.pending(chunks, artefact_for=artefact_for):
        t0 = time.perf_counter()
        vecs: list[np.ndarray] = []
        clip_ids: list[str] = []
        for stem in chunk.item_ids:
            frames = np.load(frames_dir / f"{stem}.npy")           # (n_clips, T, H, W, 3)
            side = frames_dir / f"{stem}.ids.json"
            side_ids = json.loads(side.read_text(encoding="utf-8")) if side.is_file() else []
            for i in range(frames.shape[0]):
                vecs.append(feature_fn(frames[i]))                 # (T, H, W, 3) -> (D,)
                clip_ids.append(side_ids[i] if i < len(side_ids) else f"{stem}:{i}")
        out = artefact_for(chunk)
        np.savez_compressed(out, X=np.stack(vecs), ids=np.array(clip_ids))
        dt = time.perf_counter() - t0
        store.mark_done(chunk, out)
        results.append(StageResult(stage, chunk.chunk_id, tuple(clip_ids),
                                   out, dt, out.stat().st_size))
    return results


# --------------------------------------------------------------------------- fixture

def make_synthetic_clip(path: Path, kind: str, seconds: int = 2, fps: int = 12,
                        size: int = 224) -> Path:
    """Create a synthetic clip locally with ffmpeg. No external media, no licence questions.

    ``kind`` selects a deliberately separable pattern so that *plumbing* can be verified:
    ``"static"`` (near-uniform colour) versus ``"moving"`` (travelling gradient).
    This is a fixture generator, not a dataset.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if kind == "static":
        src = f"color=c=0x224466:s={size}x{size}:r={fps}:d={seconds}"
    elif kind == "moving":
        src = (f"testsrc2=s={size}x{size}:r={fps}:d={seconds}")
    else:
        raise ValueError(f"unknown kind {kind!r}")

    cmd = [FFMPEG, "-v", "error", "-y", "-f", "lavfi", "-i", src,
           "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)]
    subprocess.run(cmd, capture_output=True, text=True, timeout=120, check=False)
    return path
