"""RAID reader (MIT) -- EVALUATION ONLY, always QUARANTINE.

Design note: RAID's official *test* split has only two columns (id + label) and no
text, so it is a submission-only leaderboard. Local evaluation must use the
``train`` split -- which is legitimate only because RAID is *never trained on* here.
Every RAID record is therefore forced into ``Split.QUARANTINE`` regardless of what
the split builder would otherwise choose; ``tests/test_splits.py`` asserts this.

Sampling
--------
The full ``train`` split is ~800 MB (without adversarial) to 11.8 GB (with). For the
first milestone we take a *systematic random* sample through the Hugging Face
datasets-server rows API (100 rows per request), using a fixed seed so the same
sample is reproducible. The sample is cached as JSONL and never re-fetched once
present, so subsequent runs are offline and deterministic.

RAID is ordered by domain then attack, so a naive head-of-file slice would be
catastrophically biased (all `abstracts`/`none`). Offsets are therefore drawn
uniformly across the full 2,270,000 rows.

The sample is a convenience for pipeline development and early signal, **not** a
benchmark result. Reporting must say so.
"""

from __future__ import annotations

import json
import random
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Iterator

from ..schema import Modality, Task
from .base import RawSample

API = "https://datasets-server.huggingface.co/rows"
DATASET = "liamdugan/raid"
CONFIG = "raid"
SPLIT = "train"
PAGE = 100          # datasets-server maximum
DEFAULT_N = 2000
DEFAULT_SEED = 20261029


def _fetch(offset: int, length: int, timeout: int = 60) -> dict:
    q = urllib.parse.urlencode(
        {"dataset": DATASET, "config": CONFIG, "split": SPLIT,
         "offset": offset, "length": length}
    )
    req = urllib.request.Request(f"{API}?{q}", headers={"User-Agent": "stopthecap-research"})
    with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310 - fixed host
        return json.loads(r.read().decode("utf-8"))


def ensure_sample(cache: Path, n: int = DEFAULT_N, seed: int = DEFAULT_SEED) -> Path:
    """Fetch (once) a cached systematic-random sample of RAID train."""
    if cache.is_file():
        return cache
    cache.parent.mkdir(parents=True, exist_ok=True)

    meta = _fetch(0, 1)
    total = int(meta["num_rows_total"])
    pages = max(1, n // PAGE)
    rng = random.Random(seed)
    max_offset = max(1, total - PAGE)
    offsets = sorted({rng.randrange(0, max_offset) for _ in range(pages)})

    with cache.open("w", encoding="utf-8") as out:
        for off in offsets:
            try:
                payload = _fetch(off, PAGE)
            except Exception as exc:  # network hiccup: record and continue
                out.write(json.dumps({"_fetch_error": str(exc), "_offset": off}) + "\n")
                continue
            for row in payload.get("rows", []):
                rec = dict(row.get("row", {}))
                rec["_offset"] = off
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
    (cache.parent / "sample_meta.json").write_text(
        json.dumps({"dataset": DATASET, "split": SPLIT, "seed": seed,
                    "page": PAGE, "n_target": n, "n_rows_total": total,
                    "offsets": offsets}, indent=2),
        encoding="utf-8",
    )
    return cache


class RaidReader:
    name = "raid"

    #: RAID is evaluation-only in this project.
    forced_split_is_quarantine = True

    def __init__(self, cache: Path, n: int = DEFAULT_N, seed: int = DEFAULT_SEED) -> None:
        self.cache = cache
        self.n = n
        self.seed = seed

    def read(self, _root: Path) -> Iterator[RawSample]:
        path = ensure_sample(self.cache, self.n, self.seed)
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if "_fetch_error" in obj:
                    continue
                text = obj.get("generation") or ""
                if not text.strip():
                    continue
                model = obj.get("model") or "unknown"
                is_human = model == "human"
                yield RawSample(
                    source_sample_id=str(obj.get("id") or f"{obj.get('_offset')}:{model}"),
                    original_label=model,
                    label=0 if is_human else 1,
                    task=Task.TEXT_BINARY,
                    modality=Modality.TEXT,
                    text=text,
                    domain=obj.get("domain"),
                    generator=None if is_human else model,
                    # source_id links a generation to its human source text
                    group_id=f"raid:{obj.get('source_id') or obj.get('id')}",
                    source_uri=f"https://huggingface.co/datasets/{DATASET}",
                    metadata={
                        "attack": obj.get("attack"),
                        "decoding": obj.get("decoding"),
                        "repetition_penalty": obj.get("repetition_penalty"),
                        "title": obj.get("title"),
                        "adv_source_id": obj.get("adv_source_id"),
                        "sample_offset": obj.get("_offset"),
                    },
                )
