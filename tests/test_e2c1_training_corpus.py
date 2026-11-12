"""Offline tests for the E2-C1 option-C training corpus builder.

All fixtures are synthetic and written under ``tmp_path``; no real dataset or
network access is used.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from pathlib import Path

import pytest

from ml.e2c1.build_training_corpus import (
    HUMAN_SAMPLE_SALT,
    InsufficientHumanChunksError,
    ai_corpus_row,
    chunk_human_records,
    chunk_text,
    human_chunk_id,
    partition_for_group,
    plan_corpus,
    sample_human_chunks,
    serialize_rows,
    split_sentences,
    word_count,
)
from ml.ingest.normalise import normalise_text

EXPECTED_KEYS = {
    "sample_id",
    "text",
    "label",
    "partition",
    "source_dataset",
    "group_id",
    "generator",
    "topic_category",
    "content_hash",
}


# ---------------------------------------------------------------------------
# Fixtures / helpers.
# ---------------------------------------------------------------------------


def _sentence(index: int, words: int = 20) -> str:
    tokens = [f"s{index}w{i}" for i in range(words - 1)] + [f"s{index}w{words - 1}."]
    return " ".join(tokens)


def _text(sentences: int, words: int = 20) -> str:
    return " ".join(_sentence(index, words) for index in range(sentences))


def _one_chunk_text() -> str:
    """100 words -> exactly one chunk under the 80--150 rule."""
    return _text(5, 20)


def _group_for(partition: str, prefix: str = "grp") -> str:
    for index in range(10_000):
        candidate = f"{prefix}-{partition}-{index}"
        if partition_for_group(candidate) == partition:
            return candidate
    raise AssertionError(f"could not find a group id for {partition!r}")


def _human_record(sample_id: str, group_id: str, text: str) -> dict:
    return {
        "sample_id": sample_id,
        "label": 0,
        "text": text,
        "group_id": group_id,
        "content_hash": hashlib.sha256(normalise_text(text).encode()).hexdigest(),
    }


def _ai_record(index: int, partition: str, generator: str = "gen_model") -> dict:
    text = f"Generated E2-C1 sample {index}."
    return {
        "sample_id": f"gen:{index:032x}",
        "generated_text": text,
        "partition": partition,
        "generator_id": generator,
        "topic_category": "stem",
        "generated_text_hash": hashlib.sha256(text.encode()).hexdigest(),
        "prompt_id": f"pmt:{index:024x}",
    }


def _ai_records(counts: dict[str, int], generators: tuple[str, ...] = ("gen_model",)) -> list[dict]:
    records: list[dict] = []
    index = 0
    for partition, count in counts.items():
        for _ in range(count):
            records.append(_ai_record(index, partition, generators[index % len(generators)]))
            index += 1
    return records


def _human_records(counts: dict[str, int]) -> list[dict]:
    records: list[dict] = []
    for partition, count in counts.items():
        for index in range(count):
            records.append(
                _human_record(
                    f"human-{partition}-{index}",
                    _group_for(partition, prefix=f"grp-{partition}-{index}"),
                    _one_chunk_text(),
                )
            )
    return records


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def _chunk(sample_id: str, partition: str, group_id: str) -> dict:
    return {
        "sample_id": sample_id,
        "partition": partition,
        "group_id": group_id,
    }


# ---------------------------------------------------------------------------
# 1. Chunker.
# ---------------------------------------------------------------------------


def test_chunker_respects_word_bounds() -> None:
    chunks = chunk_text(_text(12, 20))  # 240 words
    assert chunks
    assert all(80 <= word_count(chunk) <= 150 for chunk in chunks)


def test_chunker_never_splits_mid_sentence() -> None:
    text = _text(12, 20)
    sentences = split_sentences(text)
    chunks = chunk_text(text)
    for sentence in sentences:
        assert sum(1 for chunk in chunks if sentence in chunk) == 1
    # Chunk boundaries only occur between sentences.
    assert " ".join(chunks) == " ".join(sentences[: len(sentences)])


def test_chunker_drops_short_trailing_remainder() -> None:
    chunks = chunk_text(_text(8, 20))  # 160 words -> 140 + 20 remainder
    assert len(chunks) == 1
    assert word_count(chunks[0]) == 140


def test_chunker_keeps_a_single_long_sentence_whole() -> None:
    long_sentence = " ".join(f"token{i}" for i in range(200)) + "."
    chunks = chunk_text(long_sentence)
    assert chunks == [long_sentence]
    assert word_count(chunks[0]) == 200


def test_chunker_handles_question_and_exclamation_boundaries() -> None:
    text = (
        " ".join(f"q{i}" for i in range(39)) + "? "
        + " ".join(f"e{i}" for i in range(39)) + "! "
        + " ".join(f"d{i}" for i in range(39)) + "."
    )
    sentences = split_sentences(text)
    assert len(sentences) == 3
    chunks = chunk_text(text)
    assert all(80 <= word_count(chunk) <= 150 for chunk in chunks)


# ---------------------------------------------------------------------------
# 2. Group/partition inheritance and ids.
# ---------------------------------------------------------------------------


def test_all_chunks_share_group_and_partition() -> None:
    group_id = _group_for("train")
    records = [_human_record("human-source-1", group_id, _text(20, 20))]
    chunks = chunk_human_records(records)
    assert len(chunks) > 1
    expected_partition = partition_for_group(group_id)
    assert {chunk["group_id"] for chunk in chunks} == {group_id}
    assert {chunk["partition"] for chunk in chunks} == {expected_partition}
    for index, chunk in enumerate(chunks):
        assert chunk["sample_id"] == human_chunk_id("human-source-1", index)
        assert chunk["content_hash"] == hashlib.sha256(
            normalise_text(chunk["text"]).encode("utf-8")
        ).hexdigest()


def test_human_chunk_ids_are_deterministic_and_unique() -> None:
    group_id = _group_for("val")
    records = [_human_record("source-a", group_id, _text(12, 20))]
    first = chunk_human_records(records)
    second = chunk_human_records(records)
    assert [chunk["sample_id"] for chunk in first] == [
        chunk["sample_id"] for chunk in second
    ]
    assert len({chunk["sample_id"] for chunk in first}) == len(first)


# ---------------------------------------------------------------------------
# 3. Deterministic sampling.
# ---------------------------------------------------------------------------


def test_sampling_is_deterministic_under_shuffled_input() -> None:
    targets = {"train": 4, "val": 2, "test_holdout": 1}
    chunks = [
        _chunk(f"{partition}-{index}", partition, f"{partition}-group-{index}")
        for partition, count in targets.items()
        for index in range(count + 3)
    ]
    shuffled = list(chunks)
    random.Random(20261111).shuffle(shuffled)

    first = sample_human_chunks(chunks, targets, sample_salt=HUMAN_SAMPLE_SALT)
    second = sample_human_chunks(shuffled, targets, sample_salt=HUMAN_SAMPLE_SALT)
    assert [chunk["sample_id"] for chunk in first] == [
        chunk["sample_id"] for chunk in second
    ]


def test_sampling_keeps_whole_groups() -> None:
    targets = {"train": 2, "val": 0, "test_holdout": 0}
    chunks = [
        _chunk("big-1", "train", "big"),
        _chunk("big-2", "train", "big"),
        _chunk("small-1", "train", "small"),
        _chunk("small-2", "train", "small2"),
    ]
    selected = sample_human_chunks(chunks, targets)
    selected_ids = {chunk["sample_id"] for chunk in selected}
    assert len(selected) == 2
    for group_id in {chunk["group_id"] for chunk in selected}:
        group_ids = {
            chunk["sample_id"] for chunk in chunks if chunk["group_id"] == group_id
        }
        assert group_ids <= selected_ids


# ---------------------------------------------------------------------------
# 4. Exact balance.
# ---------------------------------------------------------------------------


def test_sampling_hits_exact_partition_balance() -> None:
    targets = {"train": 3, "val": 2, "test_holdout": 1}
    chunks = [
        _chunk(f"{partition}-{index}", partition, f"{partition}-group-{index}")
        for partition, count in targets.items()
        for index in range(count + 2)
    ]
    selected = sample_human_chunks(chunks, targets)
    counts = Counter(chunk["partition"] for chunk in selected)
    assert dict(counts) == targets
    assert len(selected) == sum(targets.values())


# ---------------------------------------------------------------------------
# 5. AI rows preserved.
# ---------------------------------------------------------------------------


def test_ai_rows_preserve_label_generator_partition() -> None:
    record = _ai_record(7, "test_holdout")
    row = ai_corpus_row(record)
    assert row["label"] == 1
    assert row["generator"] == record["generator_id"]
    assert row["partition"] == record["partition"]
    assert row["text"] == record["generated_text"]
    assert row["content_hash"] == record["generated_text_hash"]
    assert row["source_dataset"] == "e2c1_ai"


# ---------------------------------------------------------------------------
# 6. End-to-end plan: schema, uniqueness, ordering, manifest.
# ---------------------------------------------------------------------------


def test_plan_schema_unique_ids_and_manifest(tmp_path: Path) -> None:
    ai_records = [_ai_record(index, partition) for index, partition in enumerate(
        ("train", "val", "test_holdout")
    )]
    human_records = [
        _human_record(
            f"human-{partition}", _group_for(partition), _one_chunk_text()
        )
        for partition in ("train", "val", "test_holdout")
    ]
    ai_path = _write_jsonl(tmp_path / "ai.jsonl", ai_records)
    human_path = _write_jsonl(tmp_path / "human.jsonl", human_records)

    plan = plan_corpus(
        ai_records,
        human_records,
        ai_path=ai_path,
        human_path=human_path,
        min_human_chunks=0,
    )

    assert len(plan.rows) == 6
    ids = [row["sample_id"] for row in plan.rows]
    assert len(ids) == len(set(ids))
    assert ids == sorted(ids)
    for row in plan.rows:
        assert set(row) == EXPECTED_KEYS

    assert plan.corpus_sha256 == hashlib.sha256(serialize_rows(plan.rows)).hexdigest()
    counts = plan.manifest["counts"]
    assert counts["total"] == 6
    assert counts["by_label"] == {"0": 3, "1": 3}
    assert counts["by_partition"] == {"train": 2, "val": 2, "test_holdout": 2}
    assert counts["by_partition_label"]["train"] == {"0": 1, "1": 1}
    assert plan.manifest["corpus_sha256"] == plan.corpus_sha256
    assert plan.manifest["human_chunking"]["min_words"] == 80
    assert plan.manifest["human_chunking"]["max_words"] == 150


def test_plan_ai_and_human_row_contract(tmp_path: Path) -> None:
    ai_records = [_ai_record(0, "train")]
    human_records = [
        _human_record("human-0", _group_for("train"), _one_chunk_text())
    ]
    ai_path = _write_jsonl(tmp_path / "ai.jsonl", ai_records)
    human_path = _write_jsonl(tmp_path / "human.jsonl", human_records)
    plan = plan_corpus(
        ai_records,
        human_records,
        ai_path=ai_path,
        human_path=human_path,
        min_human_chunks=0,
    )

    ai_rows = [row for row in plan.rows if row["label"] == 1]
    human_rows = [row for row in plan.rows if row["label"] == 0]
    assert len(ai_rows) == 1 and len(human_rows) == 1

    ai_row = ai_rows[0]
    assert ai_row["source_dataset"] == "e2c1_ai"
    assert ai_row["generator"] == "gen_model"
    assert ai_row["partition"] == "train"
    assert ai_row["group_id"] == ai_records[0]["prompt_id"]

    human_row = human_rows[0]
    assert human_row["source_dataset"] == "ghostbuster_human_chunks"
    assert human_row["generator"] is None
    assert human_row["topic_category"] == "other"
    assert human_row["group_id"] == human_records[0]["group_id"]
    assert human_row["partition"] == "train"


# ---------------------------------------------------------------------------
# 7. Failure on short partitions.
# ---------------------------------------------------------------------------


def test_sampling_fails_when_partition_is_short() -> None:
    targets = {"train": 3, "val": 0, "test_holdout": 0}
    chunks = [
        _chunk("train-1", "train", "g1"),
        _chunk("train-2", "train", "g2"),
    ]
    with pytest.raises(InsufficientHumanChunksError) as excinfo:
        sample_human_chunks(chunks, targets)
    assert excinfo.value.details["train"]["needed"] == 3
    assert excinfo.value.details["train"]["available"] == 2
    assert excinfo.value.details["train"]["short"] == 1


def test_plan_fails_and_writes_nothing_below_human_floor(tmp_path: Path) -> None:
    ai_records = [_ai_record(index, "train") for index in range(2)]
    human_records = [
        _human_record("human-0", _group_for("train"), _one_chunk_text())
    ]
    ai_path = _write_jsonl(tmp_path / "ai.jsonl", ai_records)
    human_path = _write_jsonl(tmp_path / "human.jsonl", human_records)
    with pytest.raises(InsufficientHumanChunksError) as excinfo:
        plan_corpus(
            ai_records, human_records, ai_path=ai_path, human_path=human_path
        )
    details = excinfo.value.details["train"]
    assert details["needed"] == 1000
    assert details["available"] == 1
    assert details["short"] == 999
    assert not (tmp_path / "e2c1.jsonl").exists()


# ---------------------------------------------------------------------------
# 8. AI capped to the human chunk capacity.
# ---------------------------------------------------------------------------


def test_capping_produces_exact_one_to_one_per_partition(tmp_path: Path) -> None:
    counts = {"train": 6, "val": 5, "test_holdout": 4}
    human_counts = {"train": 2, "val": 1, "test_holdout": 3}
    ai_records = _ai_records(counts, ("gen_a", "gen_b"))
    human_records = _human_records(human_counts)
    ai_path = _write_jsonl(tmp_path / "ai.jsonl", ai_records)
    human_path = _write_jsonl(tmp_path / "human.jsonl", human_records)

    plan = plan_corpus(
        ai_records,
        human_records,
        ai_path=ai_path,
        human_path=human_path,
        min_human_chunks=0,
    )

    per_partition_label = plan.manifest["counts"]["by_partition_label"]
    assert per_partition_label == {
        "train": {"0": 2, "1": 2},
        "val": {"0": 1, "1": 1},
        "test_holdout": {"0": 3, "1": 3},
    }
    assert plan.manifest["counts"]["total"] == 2 * sum(human_counts.values())
    assert plan.manifest["counts"]["by_label"] == {"0": 6, "1": 6}
    assert plan.manifest["ai_used"] == human_counts
    assert plan.manifest["human_used"] == human_counts


def test_capping_is_deterministic_under_shuffled_input(tmp_path: Path) -> None:
    ai_records = _ai_records(
        {"train": 8, "val": 7, "test_holdout": 6}, ("gen_a", "gen_b", "gen_c")
    )
    human_records = _human_records({"train": 3, "val": 2, "test_holdout": 4})
    ai_path = _write_jsonl(tmp_path / "ai.jsonl", ai_records)
    human_path = _write_jsonl(tmp_path / "human.jsonl", human_records)

    def build(ai_rows: list[dict], human_rows: list[dict]):
        return plan_corpus(
            ai_rows,
            human_rows,
            ai_path=ai_path,
            human_path=human_path,
            min_human_chunks=0,
        )

    first = build(ai_records, human_records)

    shuffled_ai = list(ai_records)
    shuffled_human = list(human_records)
    rng = random.Random(20261111)
    rng.shuffle(shuffled_ai)
    rng.shuffle(shuffled_human)
    second = build(shuffled_ai, shuffled_human)

    assert [row["sample_id"] for row in first.rows] == [
        row["sample_id"] for row in second.rows
    ]
    assert first.corpus_sha256 == second.corpus_sha256


def test_capping_keeps_every_generator_present(tmp_path: Path) -> None:
    generators = ("gen_a", "gen_b", "gen_c")
    ai_records = _ai_records({"train": 30}, generators)
    human_records = _human_records({"train": 3})
    ai_path = _write_jsonl(tmp_path / "ai.jsonl", ai_records)
    human_path = _write_jsonl(tmp_path / "human.jsonl", human_records)

    plan = plan_corpus(
        ai_records,
        human_records,
        ai_path=ai_path,
        human_path=human_path,
        min_human_chunks=0,
    )

    ai_rows = [row for row in plan.rows if row["label"] == 1]
    assert len(ai_rows) == 3
    assert Counter(row["generator"] for row in ai_rows) == Counter(
        {generator: 1 for generator in generators}
    )


def test_manifest_records_requested_and_used_counts(tmp_path: Path) -> None:
    ai_counts = {"train": 5, "val": 4, "test_holdout": 3}
    human_counts = {"train": 2, "val": 1, "test_holdout": 2}
    ai_records = _ai_records(ai_counts, ("gen_a", "gen_b"))
    human_records = _human_records(human_counts)
    ai_path = _write_jsonl(tmp_path / "ai.jsonl", ai_records)
    human_path = _write_jsonl(tmp_path / "human.jsonl", human_records)

    plan = plan_corpus(
        ai_records,
        human_records,
        ai_path=ai_path,
        human_path=human_path,
        min_human_chunks=0,
    )

    manifest = plan.manifest
    assert manifest["ai_capped"] is True
    assert manifest["ai_requested"] == ai_counts
    assert manifest["ai_used"] == human_counts
    assert manifest["human_used"] == human_counts
    assert manifest["cap_note"] == "AI capped to Ghostbuster chunk capacity"

