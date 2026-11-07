"""Content-based decontamination of a candidate corpus against reference sets.

Why content-based: `artem9k/ai-text-detection-pile` has only `source`, `id`, `text` - no row-level
provenance. Its card says it contains HC3, but nothing marks which rows those are. The only way to
remove them is to match on content.

Two-stage matching, cheapest first:

1. **Exact** - SHA-256 of the normalised text. Catches verbatim copies.
2. **Near** - MinHash (64 permutations) over 5-word shingles, LSH-banded into 16 x 4, then verified
   with exact Jaccard on the real shingle sets. Threshold 0.85, the same value used elsewhere.

Nothing is removed silently: every removal is counted by reason and by reference dataset.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np

from ..ingest.normalise import normalise_text, shingles
from .dedup import BANDS, N_PERM, ROWS_PER_BAND, _shingle_hashes, minhash_signature

NEAR_DUP_THRESHOLD = 0.85
SHINGLE_K = 5


@dataclass
class ReferenceIndex:
    """LSH index over the datasets we must not contaminate."""

    exact: dict[str, str] = field(default_factory=dict)      # sha256 -> "dataset:id"
    bands: dict[bytes, list[int]] = field(default_factory=dict)
    signatures: np.ndarray | None = None
    _sig_list: list[np.ndarray] = field(default_factory=list)
    shingle_sets: list[set[int]] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    size: int = 0

    def add(self, text: str, label: str) -> None:
        norm = normalise_text(text)
        h = hashlib.sha256(norm.encode("utf-8")).hexdigest()
        self.exact.setdefault(h, label)

        hashes = _shingle_hashes(norm, SHINGLE_K)
        sig = minhash_signature(hashes, N_PERM)
        idx = len(self.labels)
        self.labels.append(label)
        self.shingle_sets.append(set(hashes.tolist()))
        self._sig_list.append(sig)

        for b in range(BANDS):
            key = sig[b * ROWS_PER_BAND:(b + 1) * ROWS_PER_BAND].tobytes()
            self.bands.setdefault(key, []).append(idx)
        self.size = idx + 1

    def finalise(self) -> None:
        self.exact = dict(self.exact)
        self.bands = dict(self.bands)
        if self._sig_list:
            self.signatures = np.stack(self._sig_list)
        self._sig_list = []


def build_reference_index(items: Iterable[tuple[str, str]]) -> ReferenceIndex:
    """items: (label, text) pairs."""
    idx = ReferenceIndex()
    for label, text in items:
        if text and text.strip():
            idx.add(text, label)
    idx.finalise()
    return idx


def classify(text: str, index: ReferenceIndex, threshold: float = NEAR_DUP_THRESHOLD):
    """Return (reason, matched_label, jaccard) or (None, None, None)."""
    norm = normalise_text(text)
    if not norm.strip():
        return "empty", None, None

    h = hashlib.sha256(norm.encode("utf-8")).hexdigest()
    if h in index.exact:
        return "exact", index.exact[h], 1.0

    hashes = _shingle_hashes(norm, SHINGLE_K)
    query = set(hashes.tolist())
    if not query:
        return None, None, None

    sig = minhash_signature(hashes, N_PERM)
    candidates: set[int] = set()
    for b in range(BANDS):
        key = sig[b * ROWS_PER_BAND:(b + 1) * ROWS_PER_BAND].tobytes()
        for other in index.bands.get(key, ())[:64]:
            candidates.add(other)

    best_j, best_label = 0.0, None
    for i in candidates:
        ref = index.shingle_sets[i]
        inter = len(query & ref)
        if not inter:
            continue
        j = inter / (len(query) + len(ref) - inter)
        if j > best_j:
            best_j, best_label = j, index.labels[i]
    if best_j >= threshold:
        return "near", best_label, round(best_j, 4)
    return None, None, (round(best_j, 4) if best_j else None)


def decontaminate_parquet(shards: Sequence[Path], index: ReferenceIndex, out_dir: Path,
                          text_col: str = "text", source_col: str = "source",
                          batch: int = 50_000, limit: int | None = None) -> dict:
    """Stream parquet shards, keep clean rows, and account for every removal.

    Rows are written incrementally: 1.4M texts will not fit in RAM as Python objects, and the
    digest is computed as we go so it covers actual content rather than a row count.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "kept.parquet"

    schema = pa.schema([(source_col, pa.string()), (text_col, pa.string())])
    writer = pq.ParquetWriter(out_path, schema, compression="zstd")

    removed_reasons: Counter = Counter()
    removed_by_ref: Counter = Counter()
    kept_by_class: Counter = Counter()
    total = 0
    kept_total = 0
    digest = hashlib.sha256()
    buffer: list[dict] = []
    samples: list[dict] = []
    t0 = time.perf_counter()
    stop = False

    try:
        for shard in shards:
            if stop:
                break
            try:
                pf = pq.ParquetFile(shard)
            except Exception as exc:  # partially downloaded file
                print(f"  skipping unreadable shard {shard.name}: {str(exc)[:80]}", flush=True)
                continue
            for rg in range(pf.metadata.num_row_groups):
                if stop:
                    break
                cols = pf.read_row_group(rg, columns=[text_col, source_col]).to_pydict()
                for text, src in zip(cols[text_col], cols[source_col]):
                    total += 1
                    if limit and total > limit:
                        stop = True
                        break
                    reason, matched, j = classify(text, index)
                    if reason:
                        removed_reasons[reason] += 1
                        if matched:
                            removed_by_ref[matched.split(":")[0]] += 1
                        if len(samples) < 25:
                            samples.append({"reason": reason, "matched": matched, "jaccard": j,
                                            "source": src, "preview": (text or "")[:120]})
                    else:
                        kept_by_class[src] += 1
                        kept_total += 1
                        digest.update(hashlib.sha256(
                            normalise_text(text).encode("utf-8")).digest())
                        buffer.append({source_col: src, text_col: text})
                        if len(buffer) >= batch:
                            writer.write_table(pa.Table.from_pylist(buffer, schema=schema))
                            buffer = []
                if total % 200_000 < 10_000:
                    print(f"  {total:,} examined | {removed_reasons.get('exact',0)+removed_reasons.get('near',0):,} removed",
                          flush=True)
    finally:
        if buffer:
            writer.write_table(pa.Table.from_pylist(buffer, schema=schema))
        writer.close()

    elapsed = time.perf_counter() - t0
    stats = {
        "shards": [s.name for s in shards],
        "records_examined": total,
        "removed_exact": removed_reasons.get("exact", 0),
        "removed_near": removed_reasons.get("near", 0),
        "removed_empty": removed_reasons.get("empty", 0),
        "removed_total": sum(removed_reasons.values()),
        "removed_by_reference": dict(removed_by_ref),
        "kept_human": kept_by_class.get("human", 0),
        "kept_ai": kept_by_class.get("ai", 0),
        "kept_total": kept_total,
        "removal_rate": round(sum(removed_reasons.values()) / total, 6) if total else None,
        "thresholds": {"near_dup_jaccard": NEAR_DUP_THRESHOLD, "shingle_k": SHINGLE_K,
                       "n_perm": N_PERM, "bands": BANDS, "rows_per_band": ROWS_PER_BAND},
        "reference_size": index.size,
        "seconds": round(elapsed, 1),
        "records_per_s": round(total / elapsed, 1) if elapsed else None,
        "corpus_digest": digest.hexdigest() if kept_total else None,
        "examples": samples,
        "kept_path": str(out_path) if kept_total else None,
    }
    (out_dir / "decontamination.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


# --------------------------------------------------------------------------- CLI

def _reference_items(datasets: Sequence[str] = ("ghostbuster", "hc3", "raid")):
    """(label, text) pairs from the committed canonical records."""
    from ..baselines.tfidf import load_text_records

    out = []
    for r in load_text_records():
        if r.source_dataset in datasets and r.text:
            out.append((f"{r.source_dataset}:{r.sample_id}", r.text))
    return out


def _hc3_full_items(root: Path | None = None) -> list[tuple[str, str]]:
    """Every HC3 answer from the raw corpus, not the 8k canonical sample.

    Round 1 of the pile audit used only the canonical HC3 records (8,000 of 85,449 - 9.4% coverage)
    and therefore under-reported HC3 contamination by roughly an order of magnitude.
    """
    root = root or Path("datasets/raw/hc3")
    out: list[tuple[str, str]] = []
    for path in sorted(root.glob("*.jsonl")):
        subset = path.stem
        with path.open(encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                for j, ans in enumerate(obj.get("human_answers") or []):
                    if isinstance(ans, str) and ans.strip():
                        out.append((f"hc3:{subset}:{i}:human:{j}", ans))
                for j, ans in enumerate(obj.get("chatgpt_answers") or []):
                    if isinstance(ans, str) and ans.strip():
                        out.append((f"hc3:{subset}:{i}:chatgpt:{j}", ans))
    return out


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="ml.data.decontaminate")
    ap.add_argument("--shards", default="datasets/raw/aitextpile")
    ap.add_argument("--out", default="datasets/raw/aitextpile/decon")
    ap.add_argument("--limit", type=int, default=None, help="stop after N rows (timing runs)")
    ap.add_argument("--references", default="ghostbuster,hc3,raid")
    ap.add_argument("--full-hc3", action="store_true",
                    help="use all 85k HC3 answers as reference, not the 8k canonical sample")
    args = ap.parse_args(argv)

    refs = _reference_items(tuple(args.references.split(",")))
    if args.full_hc3:
        refs = [r for r in refs if not r[0].startswith("hc3:")] + _hc3_full_items()
    print(f"building reference index from {len(refs)} records ...", flush=True)
    t0 = time.perf_counter()
    index = build_reference_index(refs)
    print(f"  index built in {time.perf_counter()-t0:.1f}s, {index.size} entries", flush=True)

    shards = sorted(Path(args.shards).glob("*.parquet"))
    print(f"decontaminating {len(shards)} shard(s) ...", flush=True)
    stats = decontaminate_parquet(shards, index, Path(args.out), limit=args.limit)

    print(json.dumps({k: stats[k] for k in (
        "records_examined", "removed_exact", "removed_near", "removed_total",
        "kept_human", "kept_ai", "kept_total", "removal_rate",
        "removed_by_reference", "seconds", "records_per_s", "corpus_digest")}, indent=2))
    if stats["examples"]:
        print("\nfirst removals:")
        for e in stats["examples"][:5]:
            print(f"  {e['reason']:6} {str(e['matched']):28} j={e['jaccard']} "
                  f"[{e['source']}] {e['preview'][:70]!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
