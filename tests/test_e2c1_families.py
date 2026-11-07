"""Near-duplicate prompt families: clustering, golden ids, validation."""

from __future__ import annotations

import random

from ml.e2c1.constants import NEAR_DUP_THRESHOLD
from ml.e2c1.families import (
    build_families,
    build_families_with_stats,
    near_dup_pairs,
    validate_families,
)
from ml.e2c1.prompts import make_prompt_hash


def words(seed: str, n: int = 80) -> list[str]:
    return [f"{seed}{index}" for index in range(n)]


def near_texts() -> tuple[str, str, str]:
    base = words("alpha")
    a = " ".join(base)
    b = " ".join(base[:-3] + ["changed0", "changed1", "changed2"])
    far = " ".join(words("other"))
    return a, b, far


def make_prompt(index: int, text: str, page_id: int, topic: str = "other") -> dict:
    return {
        "prompt_id": f"pmt:{index:024x}",
        "prompt_text": text,
        "prompt_hash": make_prompt_hash(text),
        "page_id": page_id,
        "topic_category": topic,
    }


def test_near_dup_pairs_uses_the_085_relation_and_retains_members():
    a, b, far = near_texts()
    pairs = near_dup_pairs([a, b, far])

    assert [(i, j) for i, j, _ in pairs] == [(0, 1)]
    assert pairs[0][2] >= NEAR_DUP_THRESHOLD
    assert all(2 not in (i, j) for i, j, _ in pairs)


def test_build_families_clusters_near_duplicates():
    a, b, far = near_texts()
    prompts = [
        make_prompt(1, a, page_id=1),
        make_prompt(2, b, page_id=2),
        make_prompt(3, far, page_id=3),
    ]

    families = build_families(prompts)

    assert len(families) == 3  # near-duplicates are retained, not deleted
    assert families["pmt:" + f"{1:024x}"] == families["pmt:" + f"{2:024x}"]
    assert families["pmt:" + f"{1:024x}"] != families["pmt:" + f"{3:024x}"]


def test_same_page_edges_are_transitive_with_near_dup_edges():
    a, b, far = near_texts()
    prompts = [
        make_prompt(1, a, page_id=10),   # near-duplicate of prompt 2
        make_prompt(2, b, page_id=20),
        make_prompt(3, far, page_id=20),  # same page as prompt 2
    ]

    families = build_families(prompts)

    assert len(set(families.values())) == 1


def test_families_are_order_independent():
    a, b, far = near_texts()
    prompts = [
        make_prompt(1, a, page_id=10),
        make_prompt(2, b, page_id=20),
        make_prompt(3, far, page_id=20),
        make_prompt(4, " ".join(words("delta")), page_id=30),
    ]

    first = build_families(prompts)
    shuffled = list(prompts)
    random.Random(11).shuffle(shuffled)

    # ``build_families`` takes rows, so shuffle rows; also prove the result does
    # not depend on input order by rebuilding from reversed input.
    second = build_families(list(reversed(shuffled)))
    assert first == second
    assert first == build_families(prompts)


def test_family_id_is_hasha_of_minimum_member():
    p1 = make_prompt(0, " ".join(words("left")), page_id=5)
    # Force the golden minimum prompt id.
    p1["prompt_id"] = "pmt:0123456789abcdef01234567"
    p2 = make_prompt(1, " ".join(words("right")), page_id=5)
    p2["prompt_id"] = "pmt:ffffffffffffffffffffffff"

    families = build_families([p1, p2])

    assert families["pmt:0123456789abcdef01234567"] == "fam:4a5cf8a1cca77c0f"


def test_family_stats_are_recorded():
    a, b, far = near_texts()
    prompts = [
        make_prompt(1, a, page_id=1),
        make_prompt(2, b, page_id=2),
        make_prompt(3, far, page_id=2),
        make_prompt(4, " ".join(words("delta")), page_id=4),
    ]

    families, stats = build_families_with_stats(prompts)

    assert len(families) == 4
    assert stats["n"] == len(set(families.values()))
    assert stats["size_histogram"] == {"1": 1, "3": 1}
    assert stats["max_size"] == 3
    assert stats["near_dup_edges"] == 1
    assert stats["near_dup_members_retained"] == 2
    assert stats["page_edges"] == 1


def test_validate_families_accepts_a_clean_mapping():
    a, b, far = near_texts()
    prompts = [
        make_prompt(1, a, page_id=1),
        make_prompt(2, b, page_id=2),
        make_prompt(3, far, page_id=3),
    ]
    assert validate_families(prompts, build_families(prompts)) == []


def test_validate_families_reports_missing_and_extra_prompts():
    prompts = [
        make_prompt(1, " ".join(words("a")), page_id=1),
        make_prompt(2, " ".join(words("b")), page_id=2),
    ]
    families = build_families(prompts)

    missing = dict(families)
    del missing["pmt:" + f"{1:024x}"]
    assert any("has no family" in problem for problem in validate_families(prompts, missing))

    extra = dict(families)
    extra["pmt:" + f"{99:024x}"] = "fam:" + "0" * 16
    assert any("unknown prompt" in problem for problem in validate_families(prompts, extra))


def test_validate_families_reports_non_min_hashing():
    prompts = [
        make_prompt(1, " ".join(words("a")), page_id=1),
        make_prompt(2, " ".join(words("b")), page_id=1),
    ]
    families = build_families(prompts)
    correct = families["pmt:" + f"{1:024x}"]
    wrong = "fam:" + ("1" * 16)
    assert wrong != correct

    broken = {prompt_id: wrong for prompt_id in families}
    problems = validate_families(prompts, broken)
    assert any("min(member prompt_ids)" in problem for problem in problems)


def test_validate_families_reports_a_family_id_reused_across_components():
    prompts = [
        make_prompt(1, " ".join(words("a")), page_id=1),
        make_prompt(2, " ".join(words("b")), page_id=2),
    ]
    families = build_families(prompts)
    first_id = families["pmt:" + f"{1:024x}"]
    second_id = families["pmt:" + f"{2:024x}"]
    assert first_id != second_id

    reused = {
        "pmt:" + f"{1:024x}": first_id,
        "pmt:" + f"{2:024x}": first_id,
    }
    problems = validate_families(prompts, reused)
    assert any("reused" in problem or "spans" in problem for problem in problems)
