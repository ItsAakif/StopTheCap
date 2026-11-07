"""Video inventory reader -- metadata only, no media.

Milestone-1 scope for video is *inventory and feasibility*, not pixels: GenVidBench's
media is ~201 GB across RAR and multi-part 7z archives and cannot be partially
extracted, so downloading it to prove a schema would be wasteful. Instead we read
the benchmark's own label manifests (a few MB) and emit one record per clip with:

  * ``source_uri``   -- the canonical download URL for that clip's archive
  * ``content_hash`` -- hash of the *identity string*, since we do not hold the bytes
  * ``group_id``     -- the source video / prompt group, used for leakage control
  * ``metadata``     -- generator, archive, and an explicit ``media_present=false``

Ground truth for the layout (verified 2026-10-29) -- each label file is
``<relative/path>.mp4 <label>``:

    Pair1: ms|pika|t2vz|vc2 = 1 (generated, 54,004)   vript = 0 (real, 20,131)
    Pair2: cogvideo|mora|musev|svd = 1 (generated, 54,444)   hd_vg_130m = 0 (real, 13,416)

Classification is taken from the ``<generator>`` path component, **not** from the
numeric label alone, so we never silently trust a manifest whose convention changes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

from ..schema import Modality, Task
from .base import RawSample

#: generator directory -> canonical label. 0 = real footage, 1 = fully generated.
GENERATED = {"ms", "pika", "t2vz", "vc2", "cogvideo", "mora", "musev", "svd"}
REAL = {"vript", "hd_vg_130m"}

HF_BASE = "https://huggingface.co/datasets/jian-0/GenVidBench/resolve/main/GenVidBench"


class GenVidBenchInventoryReader:
    """Reads ``Pair1_labels.txt`` / ``Pair2_labels.txt`` into video records."""

    name = "genvidbench"

    def __init__(self, label_files: tuple[str, ...] = ("Pair1_labels.txt", "Pair2_labels.txt")) -> None:
        self.label_files = label_files

    def read(self, root: Path) -> Iterator[RawSample]:
        for fname in self.label_files:
            path = root / fname
            if not path.is_file():
                continue
            pair = "Pair1" if "Pair1" in fname else "Pair2"
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                line = line.strip()
                if not line:
                    continue
                rel, _, lab = line.rpartition(" ")
                if not rel or not lab.strip().lstrip("-").isdigit():
                    continue
                manifest_label = int(lab)
                parts = rel.split("/")
                if len(parts) < 3:
                    continue
                generator_dir = parts[1]
                stem = Path(parts[-1]).stem

                if generator_dir in GENERATED:
                    label, orig = 1, "generated"
                elif generator_dir in REAL:
                    label, orig = 0, "real"
                else:
                    continue  # unknown convention: skip rather than guess

                # Cross-check the manifest's numeric label against the directory
                # convention; a disagreement is recorded, not silently resolved.
                conflict = manifest_label != label

                yield RawSample(
                    source_sample_id=rel,
                    original_label=orig,
                    label=label,
                    task=Task.VIDEO_GENERATED,
                    modality=Modality.VIDEO,
                    text=None,
                    domain=f"genvidbench_{pair.lower()}",
                    generator=None if label == 0 else generator_dir,
                    # Group by the clip's source video / prompt identity so clips
                    # derived from one source never straddle a split.
                    group_id=f"genvidbench:{pair}:{generator_dir}:{stem}",
                    identity_group=generator_dir,
                    source_uri=f"{HF_BASE}/{rel}",
                    media_hash=f"manifest:{rel}",
                    frame_sampling={"n_frames": None, "size": None, "note": "not yet extracted"},
                    metadata={
                        "pair": pair,
                        "generator_dir": generator_dir,
                        "manifest_label": manifest_label,
                        "manifest_label_conflict": conflict,
                        "media_present": False,
                        "archive_hint": f"GenVidBench/{pair}/{generator_dir}.rar",
                    },
                )


class DFDCInventoryReader:
    """Placeholder that refuses to invent records.

    DFDC terms could not be verified without authenticated acceptance of the Kaggle
    competition rules (research/13 s2), so no records are emitted. Emitting
    speculative entries would violate the "do not invent metadata" rule.
    """

    name = "dfdc"

    def read(self, _root: Path) -> Iterator[RawSample]:
        return
        yield  # pragma: no cover - makes this a generator
