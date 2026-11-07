"""Duplicate detection: exact, near, and the non-detection cases."""

from __future__ import annotations

from ml.data.dedup import find_duplicates, summarise, NEAR_DUP_THRESHOLD
from ml.ingest.schema import make_content_hash

BASE = " ".join([
    "the quick brown fox jumps over the lazy dog and then runs away into the forest",
    "where nobody can find it at all today because the trees are thick and the path",
    "is narrow and the evening light is fading fast behind the distant hills while a",
    "cold wind carries the scent of pine and damp earth across the quiet valley below",
    "and the river keeps moving past the old stone bridge that the villagers built",
    "long before any of them were born so that travellers could cross in safety and",
    "reach the market on the far side before the sun went down behind the mountain",
    "and the lamps were lit along the road that led away toward the northern coast",
])


def _hashes(texts):
    return [make_content_hash(t) for t in texts]


def test_exact_duplicate_detected():
    texts = [BASE, BASE]
    pairs = find_duplicates(texts, _hashes(texts), ["a", "b"])
    assert len(pairs) == 1
    assert pairs[0].kind == "exact"
    assert pairs[0].jaccard == 1.0


def test_near_duplicate_detected():
    near = BASE.replace("lazy", "sleepy")
    texts = [BASE, near]
    pairs = find_duplicates(texts, _hashes(texts), ["a", "b"])
    assert len(pairs) == 1
    assert pairs[0].kind == "near"
    assert pairs[0].jaccard >= NEAR_DUP_THRESHOLD


def test_distinct_texts_not_flagged():
    other = ("completely unrelated material about maritime navigation, tides and "
             "the history of cartography in the early modern period")
    texts = [BASE, other]
    assert find_duplicates(texts, _hashes(texts), ["a", "b"]) == []


def test_empty_corpus():
    assert find_duplicates([], [], []) == []


def test_summarise_counts_kinds():
    texts = [BASE, BASE, BASE.replace("lazy", "sleepy")]
    pairs = find_duplicates(texts, _hashes(texts), ["a", "b", "c"])
    s = summarise(pairs)
    assert s["exact_pairs"] >= 1
    assert s["total_pairs"] == s["exact_pairs"] + s["near_pairs"]


def test_pair_indices_are_ordered_and_unique():
    texts = [BASE, BASE, BASE.replace("lazy", "sleepy")]
    pairs = find_duplicates(texts, _hashes(texts), ["a", "b", "c"])
    seen = set()
    for p in pairs:
        assert p.i < p.j
        assert (p.i, p.j) not in seen
        seen.add((p.i, p.j))
