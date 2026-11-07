"""Chunk planning, completion markers and resume — the core of a cloud workflow.

Why this exists: Colab and Kaggle sessions are terminated without warning, and their local
disks are ephemeral. Any long video job must therefore be (a) split into chunks that each
produce a durable artefact, and (b) able to tell, on restart, exactly which chunks are already
done and whether their output is intact.

This module is deliberately dependency-free and media-agnostic: it plans and bookkeeps, it
does not decode or download. That makes it testable offline, which is the only way to trust it
before a multi-hour GPU run.

Design rules
------------
* A chunk's identity is ``stage + chunk_id + plan_hash``. If the plan changes, old markers are
  ignored rather than silently reused.
* A marker records the SHA-256 of the produced artefact. A truncated file is treated as
  *not done*, so an interrupted write can never be mistaken for a completed chunk.
* Ordering is stable, so two runs over the same input produce the same chunk ids.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

MARKER_SUFFIX = ".done.json"


def stable_hash(*parts: object, length: int = 16) -> str:
    h = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8"))
    return h.hexdigest()[:length]


def file_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(chunk_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


@dataclass(frozen=True)
class Chunk:
    """A unit of work that produces exactly one durable artefact."""

    stage: str
    index: int
    item_ids: tuple[str, ...]
    plan_hash: str

    @property
    def chunk_id(self) -> str:
        return f"{self.stage}-{self.index:05d}"

    @property
    def key(self) -> str:
        """Identity including the plan, so a changed plan invalidates old markers."""
        return f"{self.chunk_id}-{self.plan_hash}"


def plan_chunks(item_ids: Sequence[str], stage: str, size: int = 100) -> list[Chunk]:
    """Split item ids into deterministic chunks.

    Items are sorted first: input order must not change the plan, otherwise a resumed run
    would produce different chunk boundaries and re-do everything.
    """
    if size <= 0:
        raise ValueError("size must be positive")
    ordered = sorted(item_ids)
    plan_hash = stable_hash(stage, len(ordered), size, *ordered, length=12)
    chunks: list[Chunk] = []
    for i in range(0, len(ordered), size):
        group = tuple(ordered[i : i + size])
        chunks.append(Chunk(stage=stage, index=len(chunks), item_ids=group,
                            plan_hash=plan_hash))
    return chunks


@dataclass
class ChunkStore:
    """Tracks completion markers under a directory (local, Drive or a mounted dataset)."""

    root: Path
    markers: dict[str, dict] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _marker_path(self, chunk: Chunk) -> Path:
        return self.root / f"{chunk.key}{MARKER_SUFFIX}"

    def is_done(self, chunk: Chunk, artefact: Path | None = None) -> bool:
        """True only if the marker exists, matches, and the artefact is intact."""
        p = self._marker_path(chunk)
        if not p.is_file():
            return False
        try:
            meta = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        if meta.get("chunk_key") != chunk.key:
            return False
        if meta.get("n_items") != len(chunk.item_ids):
            return False
        if artefact is not None:
            if not artefact.is_file():
                return False
            if meta.get("artefact_sha256") != file_sha256(artefact):
                return False
        return True

    def mark_done(self, chunk: Chunk, artefact: Path, extra: dict | None = None) -> Path:
        meta = {
            "chunk_key": chunk.key,
            "stage": chunk.stage,
            "index": chunk.index,
            "n_items": len(chunk.item_ids),
            "plan_hash": chunk.plan_hash,
            "artefact": str(artefact),
            "artefact_bytes": artefact.stat().st_size if artefact.is_file() else None,
            "artefact_sha256": file_sha256(artefact),
        }
        if extra:
            meta.update(extra)
        p = self._marker_path(chunk)
        p.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return p

    def pending(self, chunks: Iterable[Chunk],
                artefact_for: "callable | None" = None) -> list[Chunk]:
        out = []
        for c in chunks:
            art = artefact_for(c) if artefact_for else None
            if not self.is_done(c, art):
                out.append(c)
        return out

    def summary(self, chunks: Sequence[Chunk]) -> dict:
        done = sum(1 for c in chunks if self.is_done(c))
        return {"total": len(chunks), "done": done, "pending": len(chunks) - done}


def frame_indices(total_frames: int, n_sample: int, mode: str = "uniform") -> list[int]:
    """Deterministic frame sampling.

    ``uniform`` spreads samples across the clip with the first and last deciles avoided,
    because generated videos are least informative at the very start and end. Returns
    strictly increasing indices, never duplicated, always within range.
    """
    if total_frames <= 0:
        return []
    n = max(1, min(n_sample, total_frames))
    if n == 1:
        return [total_frames // 2]
    if mode != "uniform":
        raise ValueError(f"unknown mode {mode!r}")
    lo = max(0, int(total_frames * 0.05))
    hi = min(total_frames - 1, int(total_frames * 0.95))
    if hi <= lo:
        lo, hi = 0, total_frames - 1
    step = (hi - lo) / (n - 1)
    idx = sorted({min(total_frames - 1, max(0, int(round(lo + i * step))))
                  for i in range(n)})
    return idx
