"""Video pipeline proof-of-concept runner.

Purpose: prove the plumbing end-to-end and produce real measurements. It is **not** a detector,
and its output must never be reported as detection performance.

Two modes:

* ``--source synthetic`` — generates locally-created test clips with ffmpeg. No external media,
  no licensing questions, works on a CPU-only machine. This is the default and is what the
  repository cites as "pipeline validation".
* ``--source <dir>`` — points at a directory of clips you have the right to process (for example
  Pexels downloads). The directory must contain sub-folders ``real/`` and ``generated/``.

Splits are assigned by ``group_id`` with the same deterministic bucket rule used for text, so a
source video can never straddle train and test.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from ..data.splits import bucket_for
from .chunks import ChunkStore
from .pipeline import (
    ClipItem,
    basic_frame_features,
    ffmpeg_available,
    make_synthetic_clip,
    probe_clip,
    run_features_stage,
    run_frames_stage,
)

DEFAULT_SALT = "stopthecap-2026-10-29"


@dataclass
class PocReport:
    mode: str
    n_clips: int
    n_real: int
    n_generated: int
    n_groups: int
    split_counts: dict[str, int]
    frames: dict
    features: dict
    storage: dict
    resume: dict
    model: dict | None = None
    notes: list[str] = field(default_factory=list)
    ffmpeg: bool = False


def _assign_split(group_id: str, salt: str) -> str:
    b = bucket_for(group_id, salt)
    if b < 70:
        return "train"
    if b < 85:
        return "val"
    return "test"


def _build_synthetic(root: Path, n_per_class: int) -> list[ClipItem]:
    items: list[ClipItem] = []
    for i in range(n_per_class):
        # each clip is its own source video -> its own group
        p = make_synthetic_clip(root / "generated" / f"gen_{i:03d}.mp4", "moving")
        items.append(ClipItem(clip_id=f"gen_{i:03d}", path=p,
                              group_id=f"synthetic_src_gen_{i:03d}", label=1,
                              meta={"kind": "moving", "source": "synthetic"}))
    for i in range(n_per_class):
        p = make_synthetic_clip(root / "real" / f"real_{i:03d}.mp4", "static")
        items.append(ClipItem(clip_id=f"real_{i:03d}", path=p,
                              group_id=f"synthetic_src_real_{i:03d}", label=0,
                              meta={"kind": "static", "source": "synthetic"}))
    return items


def _build_from_dir(root: Path) -> list[ClipItem]:
    items: list[ClipItem] = []
    for label, sub in ((0, "real"), (1, "generated")):
        d = root / sub
        if not d.is_dir():
            continue
        for p in sorted(d.rglob("*.mp4")):
            cid = p.stem
            items.append(ClipItem(clip_id=cid, path=p,
                                  group_id=f"{sub}:{cid}", label=label,
                                  meta={"source": str(d)}))
    return items


def run_poc(source: str = "synthetic", n_per_class: int = 12, n_frames: int = 8,
            size: int = 224, workdir: Path | None = None, salt: str = DEFAULT_SALT) -> PocReport:
    if not ffmpeg_available():
        raise RuntimeError("ffmpeg/ffprobe not found on PATH")

    tmp = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="stc_video_poc_"))
    clips_dir = tmp / "clips"
    frames_dir = tmp / "frames"
    feats_dir = tmp / "features"
    for d in (clips_dir, frames_dir, feats_dir):
        d.mkdir(parents=True, exist_ok=True)

    t_setup = time.perf_counter()
    if source == "synthetic":
        items = _build_synthetic(clips_dir, n_per_class)
        mode = "synthetic"
    else:
        items = _build_from_dir(Path(source))
        mode = f"dir:{source}"
    if not items:
        raise RuntimeError("no clips found")
    setup_s = time.perf_counter() - t_setup

    # ---- probe + metadata (real measurement of decode-adjacent cost) ----
    t0 = time.perf_counter()
    metas = {i.clip_id: probe_clip(i.path) for i in items}
    probe_s = time.perf_counter() - t0
    for i in items:
        m = metas[i.clip_id]
        i.meta.update({"duration_s": m.duration_s, "fps": m.fps,
                       "width": m.width, "height": m.height, "sha256": m.sha256[:16]})

    # ---- deterministic group splits ----
    splits = {i.clip_id: _assign_split(i.group_id, salt) for i in items}
    split_counts: dict[str, int] = {}
    for s in splits.values():
        split_counts[s] = split_counts.get(s, 0) + 1

    # ---- stage S2: frames ----
    t0 = time.perf_counter()
    frame_results = run_frames_stage(items, frames_dir, chunk_size=8,
                                     n_frames=n_frames, size=size)
    frames_s = time.perf_counter() - t0

    # ---- stage S3: features ----
    t0 = time.perf_counter()
    feat_results = run_features_stage(frames_dir, feats_dir)
    feats_s = time.perf_counter() - t0

    # ---- resume test: everything already done => zero work ----
    t0 = time.perf_counter()
    rerun_frames = run_frames_stage(items, frames_dir, chunk_size=8,
                                    n_frames=n_frames, size=size)
    resume_s = time.perf_counter() - t0

    frame_bytes = sum(f.stat().st_size for f in frames_dir.glob("*.npy"))
    feat_bytes = sum(f.stat().st_size for f in feats_dir.glob("*.npz"))
    clip_bytes = sum(p.stat().st_size for p in clips_dir.rglob("*.mp4"))

    # ---- assemble features and fit a linear head (plumbing check only) ----
    X, y, names = [], [], []
    for npz in sorted(feats_dir.glob("features-*.npz")):
        payload = np.load(npz, allow_pickle=False)
        data = payload["X"]
        ids = [str(x) for x in payload["ids"]]
        for name, vec in zip(ids, data):
            X.append(vec)
            y.append(1 if name.startswith("gen") else 0)
            names.append(name)
    model_info = None
    if len(set(y)) == 2 and len(y) >= 8:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score

        Xa = np.stack(X)
        ya = np.array(y)
        tr = np.array([splits[n] == "train" for n in names])
        te = np.array([splits[n] == "test" for n in names])
        if tr.sum() >= 4 and te.sum() >= 2 and len(set(ya[te])) == 2:
            clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(Xa[tr], ya[tr])
            score = clf.predict_proba(Xa[te])[:, 1]
            model_info = {"head": "logistic_regression",
                          "n_train": int(tr.sum()), "n_test": int(te.sum()),
                          "test_roc_auc": float(roc_auc_score(ya[te], score)),
                          "note": "plumbing check on synthetic fixtures; NOT detection performance"}
        else:
            model_info = {"skipped": "insufficient clips in train/test after group split"}

    report = PocReport(
        mode=mode,
        n_clips=len(items),
        n_real=sum(1 for i in items if i.label == 0),
        n_generated=sum(1 for i in items if i.label == 1),
        n_groups=len({i.group_id for i in items}),
        split_counts=split_counts,
        frames={"chunks": len(frame_results),
                "clips": sum(r.n_items for r in frame_results),
                "seconds": round(frames_s, 2),
                "clips_per_s": round(sum(r.n_items for r in frame_results) / frames_s, 3)
                if frames_s else None,
                "frames_per_s": round(sum(r.n_items for r in frame_results) * n_frames / frames_s, 1)
                if frames_s else None,
                "n_frames": n_frames, "size": size},
        features={"chunks": len(feat_results),
                  "seconds": round(feats_s, 2),
                  "clips_per_s": round(sum(r.n_items for r in feat_results) / feats_s, 2)
                  if feats_s else None,
                  "extractor": "basic_frame_features (placeholder, not a detector)"},
        storage={"clips_mb": round(clip_bytes / 1e6, 2),
                 "frames_mb": round(frame_bytes / 1e6, 2),
                 "features_kb": round(feat_bytes / 1e3, 1),
                 "frames_mb_per_clip": round(frame_bytes / max(1, len(items)) / 1e6, 3)},
        resume={"completed_before": len(frame_results),
                "completed_on_rerun": len(rerun_frames),
                "seconds": round(resume_s, 3),
                "verdict": "PASS" if len(rerun_frames) == 0 else "FAIL"},
        model=model_info,
        notes=[
            "Pipeline validation only. The feature extractor is a placeholder and the fixtures are "
            "synthetic; a ROC-AUC here says nothing about detecting real AI video.",
            f"setup (clip generation) took {setup_s:.1f}s and is excluded from stage timings",
            f"probe/metadata pass took {probe_s:.2f}s for {len(items)} clips",
            "on CPU; GPU timings must be measured separately on Colab/Kaggle",
        ],
        ffmpeg=True,
    )
    return report


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="ml.video.poc")
    p.add_argument("--source", default="synthetic",
                   help="'synthetic' or a directory containing real/ and generated/")
    p.add_argument("--n-per-class", type=int, default=12)
    p.add_argument("--frames", type=int, default=8)
    p.add_argument("--size", type=int, default=224)
    p.add_argument("--workdir", default=None)
    p.add_argument("--out", default="reports/video_poc.json")
    args = p.parse_args(argv)

    report = run_poc(source=args.source, n_per_class=args.n_per_class,
                     n_frames=args.frames, size=args.size,
                     workdir=Path(args.workdir) if args.workdir else None)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")

    print(f"mode: {report.mode}")
    print(f"clips: {report.n_clips} ({report.n_real} real / {report.n_generated} generated), "
          f"groups={report.n_groups}")
    print(f"splits: {report.split_counts}")
    print(f"frames: {report.frames['chunks']} chunks, {report.frames['seconds']}s, "
          f"{report.frames['clips_per_s']} clips/s, {report.frames['frames_per_s']} frames/s")
    print(f"features: {report.features['seconds']}s, {report.features['clips_per_s']} clips/s")
    print(f"storage: {report.storage}")
    print(f"resume: {report.resume['verdict']} ({report.resume['completed_on_rerun']} chunks redone)")
    if report.model:
        print(f"model: {report.model}")
    print(f"report: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
