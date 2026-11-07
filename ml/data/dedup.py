"""Exact and near-duplicate detection.

Two passes:

1. **Exact** -- identical ``content_hash``. Catches the trivial case and is exact.
2. **Near** -- MinHash + LSH banding. MinHash keeps memory bounded (a 64-integer
   signature per document rather than a full shingle set), and every candidate pair
   produced by banding is then **verified with exact Jaccard** on the real shingle
   sets. Nothing is reported as a duplicate on the strength of a hash collision
   alone.

Duplicate policy (deliberate, and reported rather than silent)
-------------------------------------------------------------
A duplicate pair is *never* resolved by deleting rows independently -- that would
split a pair across splits. Instead a duplicate pair emits a **union** of their
group keys, so they are forced into the same split, and the later index is dropped
from the canonical set with ``exclusion_reason='near_duplicate'``.

The threshold is Jaccard >= 0.85 on 5-word shingles, matching research/14 s14.3.
"""

from __future__ import annotations

import hashlib
import zlib
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from ..ingest.normalise import jaccard, shingles

SHINGLE_K = 5
NEAR_DUP_THRESHOLD = 0.85
N_PERM = 64
BANDS = 16
ROWS_PER_BAND = N_PERM // BANDS  # 4
#: Cap shingles per document. Duplicate detection is unaffected in practice (a
#: duplicate shares its opening shingles too) and it keeps hashing linear and fast.
MAX_SHINGLES_PER_DOC = 400


def _shingle_hashes(text: str, k: int = SHINGLE_K) -> np.ndarray:
    """Stable 64-bit shingle hashes.

    Uses ``zlib.crc32`` rather than a cryptographic hash purely for speed: it is
    deterministic across runs and platforms, which is the property that matters for
    reproducible duplicate detection. Collisions are harmless because every candidate
    pair is re-verified with exact Jaccard.
    """
    sh = shingles(text, k)
    if not sh:
        return np.zeros(1, dtype=np.uint64)
    if len(sh) > MAX_SHINGLES_PER_DOC:
        sh = set(sorted(sh)[:MAX_SHINGLES_PER_DOC])
    return np.fromiter((zlib.crc32(s.encode()) for s in sh), dtype=np.uint64, count=len(sh))


def minhash_signature(hashes: np.ndarray, n_perm: int = N_PERM, seed: int = 42) -> np.ndarray:
    """MinHash signature. Deterministic in (hashes, n_perm, seed)."""
    rng = np.random.default_rng(seed)
    a = rng.integers(1, (1 << 61) - 1, size=n_perm, dtype=np.int64)
    b = rng.integers(0, (1 << 61) - 1, size=n_perm, dtype=np.int64)
    mod = np.int64((1 << 61) - 1)
    h = hashes.astype(np.int64) % mod
    # (a*h + b) mod p, evaluated per permutation
    sig = (np.outer(a, h) + b[:, None]) % mod
    return sig.min(axis=1).astype(np.uint64)


@dataclass
class DuplicatePair:
    i: int
    j: int
    jaccard: float
    kind: str  # "exact" | "near"


def find_duplicates(
    texts: Sequence[str],
    content_hashes: Sequence[str],
    group_ids: Sequence[str],
    threshold: float = NEAR_DUP_THRESHOLD,
    n_perm: int = N_PERM,
) -> list[DuplicatePair]:
    """Return duplicate pairs. O(n) hashing, candidate generation via LSH bands."""
    n = len(texts)
    if n == 0:
        return []

    pairs: list[DuplicatePair] = []
    seen: set[tuple[int, int]] = set()

    # ---- pass 1: exact ----
    by_hash: dict[str, int] = {}
    for idx, h in enumerate(content_hashes):
        if h in by_hash:
            j = by_hash[h]
            key = (min(idx, j), max(idx, j))
            if key not in seen:
                seen.add(key)
                pairs.append(DuplicatePair(key[0], key[1], 1.0, "exact"))
        else:
            by_hash[h] = idx

    # ---- pass 2: near ----
    sigs = np.empty((n, n_perm), dtype=np.uint64)
    shingle_sets: list[set[int]] = []
    for idx, t in enumerate(texts):
        h = _shingle_hashes(t)
        sigs[idx] = minhash_signature(h, n_perm)
        shingle_sets.append(set(h.tolist()))

    candidates: set[tuple[int, int]] = set()
    for band in range(BANDS):
        lo = band * ROWS_PER_BAND
        hi = lo + ROWS_PER_BAND
        buckets: dict[bytes, list[int]] = {}
        for idx in range(n):
            key = sigs[idx, lo:hi].tobytes()
            lst = buckets.get(key)
            if lst is None:
                buckets[key] = [idx]
            else:
                # cap pathological buckets (e.g. many empty docs) to bound work
                if len(lst) < 64:
                    for other in lst:
                        a, b = (other, idx) if other < idx else (idx, other)
                        candidates.add((a, b))
                lst.append(idx)

    for a, b in candidates:
        if (a, b) in seen:
            continue
        j = jaccard(
            {x for x in shingle_sets[a]},
            {x for x in shingle_sets[b]},
        )
        if j >= threshold:
            seen.add((a, b))
            pairs.append(DuplicatePair(a, b, float(j), "near"))

    return pairs


def summarise(pairs: Iterable[DuplicatePair]) -> dict[str, int]:
    exact = near = 0
    for p in pairs:
        if p.kind == "exact":
            exact += 1
        else:
            near += 1
    return {"exact_pairs": exact, "near_pairs": near, "total_pairs": exact + near}
