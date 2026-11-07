"""E2-C1 prompt split: family atomicity, digest goldens, counts, refusal."""

from __future__ import annotations

import copy
import hashlib
import json
import random
from pathlib import Path

import pytest

from ml.data.splits import DEFAULT_SALT, bucket_for
from ml.e2c1 import build_splits
from ml.e2c1.constants import SPLIT_FRACTIONS, TOPIC_TAXONOMY
from ml.e2c1.families import build_families
from ml.e2c1.ids import make_sample_id
from ml.e2c1.prompts import (
    make_attribution,
    make_prompt_hash,
    make_prompt_id,
    make_source_url,
    write_manifest,
)
from ml.e2c1.split import (
    PARTITIONS,
    SplitToleranceError,
    assign_partitions,
    build_split_manifest,
    e2c1_bucket,
    e2c1_split_digest,
    expected_sample_ids,
    validate_split_manifest,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "ml" / "e2c1" / "generation_config.json"
TOPICS = list(TOPIC_TAXONOMY)


def load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def text_for(index: int, words: int = 60) -> str:
    return " ".join(f"token{index}x{position}" for position in range(words))


def family_for(index: int) -> str:
    return "fam:" + hashlib.sha256(f"family-{index}".encode()).hexdigest()[:16]


def make_prompt(
    index: int,
    *,
    topic: str | None = None,
    family_id: str | None = None,
    page_id: int | None = None,
    text: str | None = None,
    prompt_hash: str | None = None,
) -> dict:
    value = text if text is not None else text_for(index)
    return {
        "prompt_id": f"pmt:{index:024x}",
        "prompt_hash": prompt_hash or make_prompt_hash(value),
        "prompt_text": value,
        "topic_category": topic or TOPICS[index % len(TOPICS)],
        "family_id": family_id or family_for(index),
        "page_id": page_id if page_id is not None else index,
        "words": len(value.split()),
    }


def make_corpus(
    n: int, family_sizes: tuple[int, ...] = (1,)
) -> list[dict]:
    rows: list[dict] = []
    index = 0
    family_index = 0
    while index < n:
        size = family_sizes[family_index % len(family_sizes)]
        family_id = family_for(100000 + family_index)
        for _ in range(size):
            if index >= n:
                break
            rows.append(make_prompt(index, family_id=family_id))
            index += 1
        family_index += 1
    return rows


def families_of(rows: list[dict]) -> dict[str, str]:
    return {row["prompt_id"]: row["family_id"] for row in rows}


@pytest.fixture(scope="module")
def config() -> dict:
    return load_config()


@pytest.fixture(scope="module")
def big_corpus() -> list[dict]:
    return make_corpus(5000)


@pytest.fixture(scope="module")
def big_assignments(big_corpus) -> dict[str, str]:
    return assign_partitions(big_corpus, families_of(big_corpus))


@pytest.fixture(scope="module")
def big_split(big_corpus, big_assignments, config) -> dict:
    return build_split_manifest(
        big_corpus, big_assignments, config, "0" * 64
    )


def small_corpus() -> list[dict]:
    rows: list[dict] = []
    shared_family = family_for(999999)
    for index in range(100):
        family_id = shared_family if index in (0, 1) else family_for(index)
        rows.append(make_prompt(index, family_id=family_id))
    return rows


def small_split(config: dict) -> tuple[list[dict], dict]:
    rows = small_corpus()
    assignments = assign_partitions(rows, families_of(rows))
    split = build_split_manifest(rows, assignments, config, "0" * 64)
    assert validate_split_manifest(split, rows) == []
    return rows, split


# ---------------------------------------------------------------------------
# §C.6 required tests 1-6
# ---------------------------------------------------------------------------
def test_exact_hash_cannot_cross_partitions(config):
    rows, split = small_split(config)
    assignments = split["assignments"]

    by_partition: dict[str, str] = {}
    for prompt_id, partition in assignments.items():
        by_partition.setdefault(partition, prompt_id)
    first_id = by_partition["train"]
    second_id = by_partition["val"]

    shared_hash = make_prompt_hash("a synthetic exact duplicate")
    for row in rows:
        if row["prompt_id"] in (first_id, second_id):
            row["prompt_hash"] = shared_hash

    problems = validate_split_manifest(split, rows)
    assert any("duplicate prompt_hash" in problem for problem in problems)
    assert any("across partitions" in problem for problem in problems)


def test_family_cannot_cross_partitions(config):
    rows, split = small_split(config)
    shared_family = family_for(999999)
    members = [
        row["prompt_id"] for row in rows if row["family_id"] == shared_family
    ]
    assert len(members) == 2

    current = split["assignments"][members[0]]
    other = "train" if current != "train" else "val"
    split["assignments"][members[1]] = other

    problems = validate_split_manifest(split, rows)
    assert any(f"family {shared_family}" in problem and "crosses" in problem for problem in problems)


def test_near_duplicate_family_cannot_cross_partitions(config):
    base = [f"alpha{index}" for index in range(80)]
    near_a = " ".join(base)
    near_b = " ".join(base[:-3] + ["changed0", "changed1", "changed2"])

    rows = [
        make_prompt(0, text=near_a, page_id=1, family_id=""),
        make_prompt(1, text=near_b, page_id=2, family_id=""),
    ]
    for index in range(2, 60):
        rows.append(make_prompt(index, page_id=index))
    families = build_families(rows)
    for row in rows:
        row["family_id"] = families[row["prompt_id"]]

    assert families[rows[0]["prompt_id"]] == families[rows[1]["prompt_id"]]

    assignments = assign_partitions(rows, families)
    assert assignments[rows[0]["prompt_id"]] == assignments[rows[1]["prompt_id"]]
    split = build_split_manifest(rows, assignments, config, "0" * 64)

    target = rows[1]["prompt_id"]
    current = assignments[target]
    split["assignments"][target] = "train" if current != "train" else "val"

    problems = validate_split_manifest(split, rows)
    assert any("crosses" in problem for problem in problems)


def test_split_seed_determinism_and_order_independence():
    rows = make_corpus(300, family_sizes=(1, 2, 3))
    families = families_of(rows)
    first = assign_partitions(rows, families)
    first_digest = e2c1_split_digest(first, "salt")

    shuffled_rows = list(rows)
    random.Random(7).shuffle(shuffled_rows)
    shuffled_families = list(families.items())
    random.Random(8).shuffle(shuffled_families)

    second = assign_partitions(shuffled_rows, dict(shuffled_families))
    assert second == first
    assert e2c1_split_digest(second, "salt") == first_digest


def test_both_generators_in_every_partition(big_split, config):
    matrix = big_split["counts"]["by_partition_generator_decoding"]
    for partition in PARTITIONS:
        for generator in config["generators"]:
            for decoding in config["decodings"]:
                cell = matrix[partition][generator["generator_id"]][
                    decoding["decoding_config_id"]
                ]
                assert cell > 0, (partition, generator["generator_id"], decoding)


def test_no_orphaned_generation(big_corpus, big_assignments, big_split, config):
    sample_ids = expected_sample_ids(big_corpus, big_assignments, config)

    assert len(sample_ids) == 20_000
    assert len(set(sample_ids)) == 20_000
    assert all(sample_id.startswith("gen:") and len(sample_id) == 36 for sample_id in sample_ids)

    assert set(big_assignments) == {row["prompt_id"] for row in big_corpus}
    assert set(big_split["assignments"]) == set(big_assignments)
    assert validate_split_manifest(big_split, big_corpus) == []


# ---------------------------------------------------------------------------
# §C.6 additional tests 7-13
# ---------------------------------------------------------------------------
def test_split_has_no_test_id(big_assignments, big_split):
    assert set(big_assignments.values()) == set(PARTITIONS)
    assert "test_id" not in set(big_assignments.values())
    assert "test_id" not in big_split["partitions"]


def test_split_ratios_within_tolerance(big_split):
    counts = big_split["counts"]["by_partition"]
    n = big_split["n_prompts"]
    assert n == 5000
    for partition in PARTITIONS:
        target = int(n * SPLIT_FRACTIONS[partition] + 0.5)
        assert abs(counts[partition] - target) <= 10

    assert 0.79 <= counts["train"] / n <= 0.81
    assert 0.09 <= counts["val"] / n <= 0.11
    assert 0.09 <= counts["test_holdout"] / n <= 0.11


def test_split_digest_salt_and_partition_sensitive():
    assignments = {"pmt:aa": "train", "pmt:bb": "val"}
    base = e2c1_split_digest(assignments, "salt")

    assert e2c1_split_digest(assignments, "other-salt") != base
    assert e2c1_split_digest({"pmt:aa": "train", "pmt:bb": "train"}, "salt") != base


def test_split_digest_order_independent_golden():
    golden = "cbddd5234e5c18672de10d6f6e56bbcbf43636fdc430657656f2a44e6ab1ad8a"

    assert e2c1_split_digest({"pmt:aa": "train", "pmt:bb": "val"}, "s") == golden
    assert e2c1_split_digest({"pmt:bb": "val", "pmt:aa": "train"}, "s") == golden


def test_topic_strata_are_represented(big_corpus, big_assignments, big_split):
    corpus_topics = {row["topic_category"] for row in big_corpus}
    assert corpus_topics == set(TOPICS)

    for partition in ("val", "test_holdout"):
        partition_topics = {
            row["topic_category"]
            for row in big_corpus
            if big_assignments[row["prompt_id"]] == partition
        }
        assert len(partition_topics) >= 8


def test_e2c1_bucket_separator_safety():
    key, salt = "pmt:0123456789abcdef01234567", "salt"
    expected = int(
        hashlib.sha256(f"{key}\x1f{salt}".encode("utf-8")).hexdigest()[:8], 16
    ) % 100
    assert e2c1_bucket(key, salt) == expected
    assert 0 <= e2c1_bucket(key, salt) < 100

    assert e2c1_bucket("a", "bc") != e2c1_bucket("ab", "c")

    # E1's frozen bucket function is unchanged, including its known collision.
    assert bucket_for("group-1", DEFAULT_SALT) == 83
    assert bucket_for("a", "bc") == bucket_for("ab", "c") == 19


def test_counts_matrix_is_complete(big_split, config):
    counts = big_split["counts"]

    assert set(counts["by_partition"]) == set(PARTITIONS)
    for partition in PARTITIONS:
        assert set(counts["by_partition_topic"][partition]) == set(TOPICS)
        for generator in config["generators"]:
            generator_block = counts["by_partition_generator_decoding"][partition][
                generator["generator_id"]
            ]
            assert set(generator_block) == {
                decoding["decoding_config_id"] for decoding in config["decodings"]
            }
            for decoding in config["decodings"]:
                topic_block = counts["by_partition_generator_decoding_topic"][
                    partition
                ][generator["generator_id"]][decoding["decoding_config_id"]]
                assert set(topic_block) == set(TOPICS)


def test_tolerance_error_is_raised_rather_than_splitting_a_family():
    rows = [make_prompt(index, family_id=family_for(0)) for index in range(100)]
    with pytest.raises(SplitToleranceError) as info:
        assign_partitions(rows, families_of(rows))
    assert info.value.counts


def test_sample_id_golden():
    assert (
        make_sample_id(
            "pmt:0123456789abcdef01234567",
            "mistral_7b_instruct_v02",
            "63a8b081895390a26e140280378bc85ec8bce07a",
            "greedy",
        )
        == "gen:02c4c41a3165d8d3cab8522a4493eada"
    )


# ---------------------------------------------------------------------------
# build_splits refusal paths
# ---------------------------------------------------------------------------
def _assert_refused(tmp_path, rows: list[dict], split: dict, needle: str) -> None:
    out_path = tmp_path / "split.json"
    with pytest.raises(build_splits.SplitValidationError) as info:
        build_splits.write_split(out_path, split, rows)
    assert any(needle in problem for problem in info.value.problems)
    assert not out_path.exists()


def test_build_splits_refuses_family_crossing(tmp_path, config):
    rows, split = small_split(config)
    shared_family = family_for(999999)
    members = [row["prompt_id"] for row in rows if row["family_id"] == shared_family]
    current = split["assignments"][members[0]]
    split["assignments"][members[1]] = "train" if current != "train" else "val"
    _assert_refused(tmp_path, rows, split, "crosses partitions")


def test_build_splits_refuses_digest_mismatch(tmp_path, config):
    rows, split = small_split(config)
    split["split_digest"] = "0" * 64
    _assert_refused(tmp_path, rows, split, "split_digest mismatch")


def test_build_splits_refuses_missing_prompt(tmp_path, config):
    rows, split = small_split(config)
    victim = sorted(split["assignments"])[0]
    del split["assignments"][victim]
    _assert_refused(tmp_path, rows, split, "has no partition assignment")


def test_build_splits_refuses_test_id_partition(tmp_path, config):
    rows, split = small_split(config)
    victim = sorted(split["assignments"])[0]
    split["assignments"][victim] = "test_id"
    _assert_refused(tmp_path, rows, split, "test_id")


def test_build_splits_cli_writes_valid_manifest(tmp_path):
    rows = []
    for index in range(200):
        revision_id = str(100000 + index)
        prompt_text = text_for(index)
        prompt_hash = make_prompt_hash(prompt_text)
        source_url = make_source_url(revision_id)
        rows.append(
            {
                "prompt_id": make_prompt_id(revision_id, prompt_hash),
                "source_title": f"Title {index}",
                "source_url": source_url,
                "revision_id": revision_id,
                "retrieved_at": "2026-11-06",
                "source_licence": "CC-BY-SA-4.0",
                "attribution": make_attribution(
                    f"Title {index}", revision_id, "2026-11-06", source_url
                ),
                "prompt_text": prompt_text,
                "prompt_hash": prompt_hash,
                "topic_category": TOPICS[index % len(TOPICS)],
                "content_safety_status": "ok",
                "page_id": index,
                "words": len(prompt_text.split()),
                "script_latin": True,
                "topic_rule": "other",
                "family_id": family_for(index),
            }
        )

    prompts_path = tmp_path / "e2_c1_prompts.jsonl"
    write_manifest(prompts_path, rows)

    config = load_config()
    config["expected_prompts"] = len(rows)
    config["prompt_manifest"] = str(prompts_path)
    config_path = tmp_path / "generation_config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    out_path = tmp_path / "e2_c1_splits.json"
    assert build_splits.main(
        [
            "--prompts",
            str(prompts_path),
            "--out",
            str(out_path),
            "--config",
            str(config_path),
        ]
    ) == 0

    split = json.loads(out_path.read_text(encoding="utf-8"))
    assert split["n_prompts"] == len(rows)
    assert split["prompt_manifest_sha256"] == hashlib.sha256(
        prompts_path.read_bytes()
    ).hexdigest()
    assert split["split_digest"] == e2c1_split_digest(
        split["assignments"], split["salt"]
    )
    assert validate_split_manifest(split, rows) == []
