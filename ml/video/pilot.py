"""Pexels + Wan 2.1 pilot preparation.

This module holds the parts of the video pilot that do **not** need a GPU or an API key:
record schema, category matching, grouped splits, and a compression-robustness transform.

It deliberately contains no generation code and downloads nothing. Generation and download are
run manually (documented in docs/video-pilot-execution.md) so that a failed job cannot corrupt
the repository's manifest.

Licence positions this module assumes (see DATASET_REGISTRY.md):
  * Pexels clips  - permissive; modification allowed; do not republish as stock media.
  * Wan 2.1 output - Apache-2.0; outputs may be published.
Neither requires contacting anyone.
"""

from __future__ import annotations

import hashlib
import io
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

from ..data.splits import bucket_for

DEFAULT_SALT = "stopthecap-2026-10-29"

#: Categories used to match the real and generated sides. Kept coarse on purpose: fine-grained
#: labels make matching brittle and the pilot needs balanced content, not taxonomy.
CATEGORIES = (
    "nature", "city", "people", "animals", "food", "transport", "indoor", "abstract",
)


@dataclass
class ClipRecord:
    clip_id: str
    label: int                     # 0 = real (Pexels), 1 = generated (Wan)
    source: str                    # "pexels" | "wan2.1-t2v-1.3b"
    group_id: str                  # source video or generation batch; never split across sets
    category: str
    path: str
    licence_ref: str
    generator: str | None = None
    prompt: str | None = None
    resolution: tuple[int, int] | None = None
    duration_s: float | None = None
    fps: float | None = None
    sha256: str | None = None
    split: str | None = None
    meta: dict = field(default_factory=dict)

    def validate(self) -> list[str]:
        problems = []
        if self.label not in (0, 1):
            problems.append(f"{self.clip_id}: label must be 0 or 1")
        if self.label == 1 and not self.generator:
            problems.append(f"{self.clip_id}: generated clip needs a generator")
        if self.category not in CATEGORIES:
            problems.append(f"{self.clip_id}: unknown category {self.category!r}")
        if not self.licence_ref:
            problems.append(f"{self.clip_id}: missing licence_ref")
        if not self.group_id:
            problems.append(f"{self.clip_id}: missing group_id")
        return problems


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def build_record(clip_id: str, label: int, source: str, group_id: str, category: str,
                 path: Path, licence_ref: str, **kwargs) -> ClipRecord:
    return ClipRecord(clip_id=clip_id, label=label, source=source, group_id=group_id,
                      category=category, path=str(path), licence_ref=licence_ref,
                      sha256=file_sha256(path) if Path(path).is_file() else None,
                      **kwargs)


def match_categories(real: Sequence[ClipRecord],
                     generated: Sequence[ClipRecord]) -> tuple[list[tuple[str, str]], dict]:
    """Greedy one-to-one category matching, balanced to the smaller side.

    Returns (pairs, report). The report records how many clips were dropped per category so the
    imbalance is visible rather than silently absorbed.
    """
    from collections import defaultdict

    by_real: dict[str, list[ClipRecord]] = defaultdict(list)
    by_gen: dict[str, list[ClipRecord]] = defaultdict(list)
    for r in real:
        by_real[r.category].append(r)
    for g in generated:
        by_gen[g.category].append(g)

    pairs: list[tuple[str, str]] = []
    report: dict[str, dict[str, int]] = {}
    for cat in CATEGORIES:
        n = min(len(by_real[cat]), len(by_gen[cat]))
        for i in range(n):
            pairs.append((by_real[cat][i].clip_id, by_gen[cat][i].clip_id))
        report[cat] = {"real": len(by_real[cat]), "generated": len(by_gen[cat]),
                       "paired": n,
                       "dropped": (len(by_real[cat]) - n) + (len(by_gen[cat]) - n)}
    return pairs, report


def assign_splits(records: Iterable[ClipRecord], salt: str = DEFAULT_SALT,
                  train: int = 70, val: int = 15) -> list[ClipRecord]:
    """Assign splits by group_id so a source video never straddles train/test."""
    out = []
    for r in records:
        b = bucket_for(r.group_id, salt)
        split = "train" if b < train else ("val" if b < train + val else "test")
        out.append(ClipRecord(**{**asdict(r), "split": split}))
    return out


def leakage_report(records: Sequence[ClipRecord]) -> dict:
    """Groups spanning more than one split, per class. Must be zero."""
    spans: dict[str, set[str]] = {}
    class_of: dict[str, int] = {}
    for r in records:
        if r.split is None:
            continue
        spans.setdefault(r.group_id, set()).add(r.split)
        class_of[r.group_id] = r.label
    bad = {g: sorted(s) for g, s in spans.items() if len(s) > 1}
    by_class = {0: 0, 1: 0}
    for g in bad:
        by_class[class_of[g]] += 1
    return {"groups": len(spans), "leaking_groups": len(bad), "by_class": by_class,
            "examples": dict(list(bad.items())[:5])}


def write_manifest(records: Sequence[ClipRecord], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in records:
        d = asdict(r)
        if isinstance(d.get("resolution"), (list, tuple)):
            d["resolution"] = list(d["resolution"])
        rows.append(d)
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return path


def validate_all(records: Sequence[ClipRecord]) -> dict:
    problems: list[str] = []
    seen: set[str] = set()
    for r in records:
        problems.extend(r.validate())
        if r.clip_id in seen:
            problems.append(f"duplicate clip_id {r.clip_id}")
        seen.add(r.clip_id)
    return {"n": len(records), "problems": problems, "ok": not problems}


def compression_variant(frames, quality: int = 40):
    """Re-encode frames as JPEG at a given quality, to test compression robustness.

    Returns a uint8 array of the same shape. Used by the third control: a detector that only
    works on pristine frames is not useful on real uploads.
    """
    import numpy as np
    from PIL import Image

    arr = np.asarray(frames, dtype=np.uint8)
    out = np.empty_like(arr)
    for i, frame in enumerate(arr):
        buf = io.BytesIO()
        Image.fromarray(frame).save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        out[i] = np.asarray(Image.open(buf).convert("RGB"), dtype=np.uint8)
    return out


def resize_variant(frames, size: int = 112):
    """Downscale then upscale, to test resolution robustness."""
    import numpy as np
    from PIL import Image

    arr = np.asarray(frames, dtype=np.uint8)
    out = np.empty_like(arr)
    for i, frame in enumerate(arr):
        img = Image.fromarray(frame)
        small = img.resize((size, size), Image.BILINEAR)
        out[i] = np.asarray(small.resize((arr.shape[2], arr.shape[1]), Image.BILINEAR),
                            dtype=np.uint8)
    return out
