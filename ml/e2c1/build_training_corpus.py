"""Build the E2-C1 training corpus (option C).

Option C pairs the generated E2-C1 texts with *chunked* Ghostbuster human
essays.  Human essays are sentence-aligned into 80--150 word chunks so that
their length distribution resembles the generated samples, and the chunks are
sampled group-atomically so that every partition has exactly as many human rows
as AI rows.

Because the AI corpus is larger than the human chunk supply, the default build
caps the AI side per partition to the available human chunks (see
``--cap-to-human``).  AI rows are sampled deterministically and stratified by
generator so that every generator present in a partition keeps at least one
sample whenever the quota allows it.

Everything here is deterministic and offline.  The build refuses to write a
partial corpus: a partition with fewer than :data:`MIN_HUMAN_CHUNKS` human
chunks is treated as a real data problem and raises
:class:`InsufficientHumanChunksError`, and nothing is written.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import re
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ml.ingest.normalise import normalise_text

# ---------------------------------------------------------------------------
# Frozen build parameters (mirrored in the manifest).
# ---------------------------------------------------------------------------

AI_PATH = "datasets/raw/e2c1_generated/merged.jsonl"
HUMAN_PATH = "datasets/canonical/ghostbuster.jsonl"
OUT_CORPUS = "datasets/canonical/e2c1.jsonl"
OUT_MANIFEST = "datasets/manifests/e2_c1_training.json"

SALT = "e2c1-train-v1"
HUMAN_SAMPLE_SALT = "e2c1-human-sample-v1"
AI_CAP_SALT = "e2c1-ai-cap-v1"

MIN_WORDS = 80
MAX_WORDS = 150

#: A partition with fewer human chunks than this cannot produce a usable
#: balanced corpus; the build refuses instead of silently shrinking to nothing.
MIN_HUMAN_CHUNKS = 1000

CAP_NOTE = "AI capped to Ghostbuster chunk capacity"

PARTITIONS: tuple[str, ...] = ("train", "val", "test_holdout")

AI_SOURCE_DATASET = "e2c1_ai"
HUMAN_SOURCE_DATASET = "ghostbuster_human_chunks"
HUMAN_TOPIC = "other"

#: Split by punctuation followed by whitespace, keeping the punctuation attached
#: to the sentence it closes (``. ``, ``? `` and ``! `` plus newlines).
_SENTENCE_SPLIT = re.compile(r"(?<=[.?!])\s+")


class InsufficientHumanChunksError(RuntimeError):
    """Raised when a partition cannot be filled with whole human groups.

    ``details`` maps each short partition to ``needed``, ``available`` and
    ``short`` counts so the CLI can print an actionable message.
    """

    def __init__(self, details: Mapping[str, Mapping[str, int]]) -> None:
        self.details = {partition: dict(values) for partition, values in details.items()}
        parts = []
        for partition in PARTITIONS:
            if partition not in self.details:
                continue
            values = self.details[partition]
            parts.append(
                f"{partition}: need {values['needed']}, available "
                f"{values['available']}, short {values['short']}"
            )
        super().__init__("insufficient human chunks (" + "; ".join(parts) + ")")


@dataclass
class BuildPlan:
    """Everything computed before a byte is written."""

    rows: list[dict[str, Any]]
    manifest: dict[str, Any]
    corpus_sha256: str
    targets: dict[str, int]
    available: dict[str, int]
    human_chunks: list[dict[str, Any]] = field(repr=False, default_factory=list)


# ---------------------------------------------------------------------------
# Hashing / IO helpers.
# ---------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: record is not an object")
            rows.append(row)
    return rows


def serialize_rows(rows: Sequence[Mapping[str, Any]]) -> bytes:
    """Canonical JSONL encoding; the corpus SHA-256 is taken over these bytes."""
    return "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
        for row in rows
    ).encode("utf-8")


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise


def _write_json_atomic(path: Path, obj: Mapping[str, Any]) -> None:
    payload = (
        json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    _write_bytes_atomic(path, payload)


# ---------------------------------------------------------------------------
# Sentence splitting and chunking.
# ---------------------------------------------------------------------------


def word_count(text: str) -> int:
    return len([word for word in text.split() if word])


def split_sentences(text: str) -> list[str]:
    """Split on ``. ``, ``? `` and ``! `` boundaries without losing text."""
    if not text:
        return []
    return [sentence for sentence in _SENTENCE_SPLIT.split(text.strip()) if sentence]


def chunk_text(
    text: str, *, min_words: int = MIN_WORDS, max_words: int = MAX_WORDS
) -> list[str]:
    """Pack sentences greedily into chunks of ``min_words``--``max_words``.

    A sentence is never split.  A chunk is closed only once it has reached
    ``min_words`` and the next sentence would push it past ``max_words``; this
    keeps every non-final chunk at or above the minimum.  A trailing remainder
    below ``min_words`` is dropped.  A single sentence longer than ``max_words``
    is emitted whole because splitting it would violate the sentence rule.
    """
    if min_words <= 0 or max_words < min_words:
        raise ValueError("require 0 < min_words <= max_words")

    chunks: list[str] = []
    current: list[str] = []
    current_words = 0

    for sentence in split_sentences(text):
        sentence_words = word_count(sentence)
        if (
            current
            and current_words >= min_words
            and current_words + sentence_words > max_words
        ):
            chunks.append(" ".join(current))
            current = [sentence]
            current_words = sentence_words
        else:
            current.append(sentence)
            current_words += sentence_words

    if current:
        chunks.append(" ".join(current))

    if chunks and word_count(chunks[-1]) < min_words:
        chunks.pop()

    return chunks


# ---------------------------------------------------------------------------
# Human chunk records.
# ---------------------------------------------------------------------------


def partition_for_group(group_id: str, salt: str = SALT) -> str:
    """Deterministic group-level partition; every chunk of a group agrees."""
    bucket = int(
        hashlib.sha256(f"{group_id}\x1f{salt}".encode("utf-8")).hexdigest()[:8], 16
    ) % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "val"
    return "test_holdout"


def human_chunk_id(source_sample_id: str, chunk_index: int) -> str:
    digest = hashlib.sha256(
        f"{source_sample_id}|{chunk_index}".encode("utf-8")
    ).hexdigest()
    return "e2c1h:" + digest[:24]


def chunk_human_records(
    records: Iterable[Mapping[str, Any]],
    *,
    salt: str = SALT,
    min_words: int = MIN_WORDS,
    max_words: int = MAX_WORDS,
) -> list[dict[str, Any]]:
    """Chunk every ``label == 0`` record and attach ids, hashes and partitions."""
    chunks: list[dict[str, Any]] = []
    for record in records:
        if int(record["label"]) != 0:
            continue
        source_sample_id = str(record["sample_id"])
        group_id = str(record["group_id"])
        partition = partition_for_group(group_id, salt=salt)
        for index, chunk in enumerate(
            chunk_text(str(record["text"]), min_words=min_words, max_words=max_words)
        ):
            chunks.append(
                {
                    "sample_id": human_chunk_id(source_sample_id, index),
                    "text": chunk,
                    "label": 0,
                    "partition": partition,
                    "source_dataset": HUMAN_SOURCE_DATASET,
                    "group_id": group_id,
                    "generator": None,
                    "topic_category": HUMAN_TOPIC,
                    "content_hash": _sha256_text(normalise_text(chunk)),
                }
            )
    return chunks


# ---------------------------------------------------------------------------
# AI rows and partition targets.
# ---------------------------------------------------------------------------


def ai_corpus_row(record: Mapping[str, Any]) -> dict[str, Any]:
    """Project an E2-C1 generation record onto the training-corpus schema."""
    partition = str(record["partition"])
    if partition not in PARTITIONS:
        raise ValueError(f"AI record {record['sample_id']!r} has partition {partition!r}")
    return {
        "sample_id": str(record["sample_id"]),
        "text": str(record["generated_text"]),
        "label": 1,
        "partition": partition,
        "source_dataset": AI_SOURCE_DATASET,
        "group_id": str(record["prompt_id"]),
        "generator": str(record["generator_id"]),
        "topic_category": str(record["topic_category"]),
        "content_hash": str(record["generated_text_hash"]),
    }


def ai_partition_targets(ai_rows: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    targets = {partition: 0 for partition in PARTITIONS}
    for row in ai_rows:
        partition = str(row["partition"])
        if partition not in targets:
            raise ValueError(f"AI record has unknown partition {partition!r}")
        targets[partition] += 1
    return targets


def _human_partition_counts(
    chunks: Iterable[Mapping[str, Any]],
) -> dict[str, int]:
    counts = {partition: 0 for partition in PARTITIONS}
    for chunk in chunks:
        counts[str(chunk["partition"])] += 1
    return counts


# ---------------------------------------------------------------------------
# Group-atomic deterministic sampling.
# ---------------------------------------------------------------------------


def _sample_rank(sample_id: str, salt: str) -> str:
    return hashlib.sha256(f"{salt}\x1f{sample_id}".encode("utf-8")).hexdigest()


def sample_human_chunks(
    chunks: Sequence[Mapping[str, Any]],
    targets: Mapping[str, int],
    *,
    sample_salt: str = HUMAN_SAMPLE_SALT,
) -> list[dict[str, Any]]:
    """Select whole human groups per partition until the quota is exactly met.

    Groups are ranked by their best-ranked member chunk (``sha256(salt + "\\x1f" +
    sample_id)``), so shuffling the input cannot change the result.  A group is
    taken only if it fits in the remaining quota; otherwise it is skipped.  If a
    partition cannot be filled exactly, :class:`InsufficientHumanChunksError`
    reports every short partition and no selection is returned.
    """
    by_partition: dict[str, list[Mapping[str, Any]]] = {
        partition: [] for partition in PARTITIONS
    }
    for chunk in chunks:
        by_partition[str(chunk["partition"])].append(chunk)

    details: dict[str, dict[str, int]] = {}
    selected: list[dict[str, Any]] = []

    for partition in PARTITIONS:
        target = int(targets.get(partition, 0))
        available = len(by_partition[partition])
        if available < target:
            details[partition] = {
                "needed": target,
                "available": available,
                "short": target - available,
            }
            continue

        grouped: dict[str, list[Mapping[str, Any]]] = collections.defaultdict(list)
        for chunk in by_partition[partition]:
            grouped[str(chunk["group_id"])].append(chunk)

        ranked_groups: list[tuple[str, str, list[Mapping[str, Any]]]] = []
        for group_id, members in grouped.items():
            rank = min(
                _sample_rank(str(member["sample_id"]), sample_salt)
                for member in members
            )
            ranked_groups.append(
                (rank, group_id, sorted(members, key=lambda item: str(item["sample_id"])))
            )
        ranked_groups.sort(key=lambda item: (item[0], item[1]))

        remaining = target
        for _rank, _group_id, members in ranked_groups:
            if len(members) <= remaining:
                selected.extend(dict(member) for member in members)
                remaining -= len(members)

        if remaining:
            details[partition] = {
                "needed": target,
                "available": available,
                "short": remaining,
            }

    if details:
        raise InsufficientHumanChunksError(details)

    return selected


# ---------------------------------------------------------------------------
# Deterministic, generator-stratified AI capping.
# ---------------------------------------------------------------------------


def _rank_key(row: Mapping[str, Any], salt: str) -> tuple[str, str]:
    sample_id = str(row["sample_id"])
    return (_sample_rank(sample_id, salt), sample_id)


def sample_ai_rows(
    rows: Sequence[Mapping[str, Any]],
    targets: Mapping[str, int],
    *,
    cap_salt: str = AI_CAP_SALT,
) -> list[dict[str, Any]]:
    """Select at most ``targets[partition]`` AI rows, stratified by generator.

    Rows are ranked by ``sha256(cap_salt + "\\x1f" + sample_id)``, so shuffling
    the input cannot change the result.  One top-ranked row per generator is
    reserved first (generators ordered by their best rank), then the remaining
    quota is filled from the global ranking.  When the quota is smaller than the
    number of generators the best-ranked generators keep their seat and the rest
    cannot; every generator present keeps a sample whenever the quota allows it.
    """
    by_partition: dict[str, list[Mapping[str, Any]]] = {
        partition: [] for partition in PARTITIONS
    }
    for row in rows:
        partition = str(row["partition"])
        if partition not in by_partition:
            raise ValueError(f"AI record has unknown partition {partition!r}")
        by_partition[partition].append(row)

    selected: list[dict[str, Any]] = []
    for partition in PARTITIONS:
        target = int(targets.get(partition, 0))
        candidates = by_partition[partition]
        if target < 0:
            raise ValueError(f"negative AI target for {partition!r}")
        if target == 0:
            continue
        if target >= len(candidates):
            selected.extend(dict(row) for row in candidates)
            continue

        grouped: dict[str, list[Mapping[str, Any]]] = collections.defaultdict(list)
        for row in candidates:
            grouped[str(row["generator"])].append(row)

        ranked_groups: list[
            tuple[tuple[str, str], str, list[Mapping[str, Any]]]
        ] = []
        for generator, members in grouped.items():
            members_sorted = sorted(
                members, key=lambda item: _rank_key(item, cap_salt)
            )
            ranked_groups.append(
                (_rank_key(members_sorted[0], cap_salt), generator, members_sorted)
            )
        ranked_groups.sort(key=lambda item: (item[0], item[1]))

        chosen: list[Mapping[str, Any]] = []
        chosen_ids: set[str] = set()
        for _key, _generator, members in ranked_groups:
            if len(chosen) >= target:
                break
            chosen.append(members[0])
            chosen_ids.add(str(members[0]["sample_id"]))

        if len(chosen) < target:
            pool = [
                row for row in candidates if str(row["sample_id"]) not in chosen_ids
            ]
            pool.sort(key=lambda item: _rank_key(item, cap_salt))
            chosen.extend(pool[: target - len(chosen)])

        selected.extend(dict(row) for row in chosen)

    return selected


# ---------------------------------------------------------------------------
# Manifest and full build.
# ---------------------------------------------------------------------------


def _chunks_per_essay(
    records: Iterable[Mapping[str, Any]],
    *,
    min_words: int = MIN_WORDS,
    max_words: int = MAX_WORDS,
) -> dict[str, int]:
    histogram: collections.Counter[int] = collections.Counter()
    for record in records:
        if int(record["label"]) != 0:
            continue
        histogram[
            len(
                chunk_text(
                    str(record["text"]), min_words=min_words, max_words=max_words
                )
            )
        ] += 1
    return {str(size): histogram[size] for size in sorted(histogram)}


def _counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_label = {"0": 0, "1": 0}
    by_partition = {partition: 0 for partition in PARTITIONS}
    by_partition_label = {
        partition: {"0": 0, "1": 0} for partition in PARTITIONS
    }
    by_generator: collections.Counter[str] = collections.Counter()
    by_topic: collections.Counter[str] = collections.Counter()

    for row in rows:
        label = str(int(row["label"]))
        partition = str(row["partition"])
        by_label[label] += 1
        by_partition[partition] += 1
        by_partition_label[partition][label] += 1
        by_topic[str(row["topic_category"])] += 1
        if row.get("generator") is not None:
            by_generator[str(row["generator"])] += 1

    return {
        "total": len(rows),
        "by_label": by_label,
        "by_partition": by_partition,
        "by_partition_label": by_partition_label,
        "by_generator": dict(sorted(by_generator.items())),
        "by_topic": dict(sorted(by_topic.items())),
    }


def build_manifest(
    rows: Sequence[Mapping[str, Any]],
    *,
    ai_path: Path,
    human_path: Path,
    corpus_sha256: str,
    human_records: Sequence[Mapping[str, Any]],
    ai_requested: Mapping[str, int],
    ai_used: Mapping[str, int],
    human_used: Mapping[str, int],
    ai_capped: bool = True,
    salt: str = SALT,
    min_words: int = MIN_WORDS,
    max_words: int = MAX_WORDS,
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "corpus": "e2c1",
        "created_at": "2026-11-11",
        "salt": salt,
        "ai_source": {"path": str(ai_path), "sha256": sha256_file(ai_path)},
        "human_source": {"path": str(human_path), "sha256": sha256_file(human_path)},
        "counts": _counts(rows),
        "ai_capped": ai_capped,
        "ai_requested": {
            partition: int(ai_requested.get(partition, 0)) for partition in PARTITIONS
        },
        "ai_used": {
            partition: int(ai_used.get(partition, 0)) for partition in PARTITIONS
        },
        "human_used": {
            partition: int(human_used.get(partition, 0)) for partition in PARTITIONS
        },
        "cap_note": CAP_NOTE,
        "human_chunking": {
            "min_words": min_words,
            "max_words": max_words,
            "chunks_per_essay": _chunks_per_essay(
                human_records, min_words=min_words, max_words=max_words
            ),
        },
        "corpus_sha256": corpus_sha256,
        "notes": (
            "Option C: Ghostbuster label-0 essays are sentence-aligned into "
            f"{MIN_WORDS}-{MAX_WORDS} word chunks (trailing <{MIN_WORDS} dropped) and "
            "sampled group-atomically per partition. By default the AI side is "
            "capped to the available human chunk count (deterministic, stratified "
            "by generator) so both labels match 1:1 per partition. Human rows have "
            "generator=null and topic_category='other'. AI group_id is the source "
            "prompt_id."
        ),
    }


def plan_corpus(
    ai_records: Sequence[Mapping[str, Any]],
    human_records: Sequence[Mapping[str, Any]],
    *,
    ai_path: Path,
    human_path: Path,
    salt: str = SALT,
    sample_salt: str = HUMAN_SAMPLE_SALT,
    cap_salt: str = AI_CAP_SALT,
    cap_to_human: bool = True,
    min_human_chunks: int = MIN_HUMAN_CHUNKS,
) -> BuildPlan:
    """Compute the corpus and manifest without writing anything.

    With ``cap_to_human`` (the default) each partition keeps
    ``min(available_ai, available_human_chunks)`` rows per label.  Raises
    :class:`InsufficientHumanChunksError` when a partition has fewer than
    ``min_human_chunks`` human chunks or when whole human groups cannot fill the
    quota exactly; callers must not write in that case.
    """
    ai_rows = [ai_corpus_row(record) for record in ai_records]
    ai_requested = ai_partition_targets(ai_rows)

    human_chunks = chunk_human_records(human_records, salt=salt)
    available = _human_partition_counts(human_chunks)

    floor = int(min_human_chunks)
    details: dict[str, dict[str, int]] = {}
    for partition in PARTITIONS:
        if available[partition] < floor:
            details[partition] = {
                "needed": floor,
                "available": available[partition],
                "short": floor - available[partition],
            }
    if details:
        raise InsufficientHumanChunksError(details)

    if cap_to_human:
        targets = {
            partition: min(ai_requested[partition], available[partition])
            for partition in PARTITIONS
        }
    else:
        targets = dict(ai_requested)

    selected_human = sample_human_chunks(
        human_chunks, targets, sample_salt=sample_salt
    )
    if cap_to_human:
        selected_ai = sample_ai_rows(ai_rows, targets, cap_salt=cap_salt)
    else:
        selected_ai = [dict(row) for row in ai_rows]

    rows = sorted(selected_ai + selected_human, key=lambda row: str(row["sample_id"]))
    sample_ids = [str(row["sample_id"]) for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        duplicates = [
            sample_id
            for sample_id, count in collections.Counter(sample_ids).items()
            if count > 1
        ]
        raise ValueError(f"duplicate sample_id(s): {sorted(duplicates)[:5]}")

    ai_used = _human_partition_counts(selected_ai)
    human_used = _human_partition_counts(selected_human)

    payload = serialize_rows(rows)
    corpus_sha256 = hashlib.sha256(payload).hexdigest()
    manifest = build_manifest(
        rows,
        ai_path=ai_path,
        human_path=human_path,
        corpus_sha256=corpus_sha256,
        human_records=human_records,
        ai_requested=ai_requested,
        ai_used=ai_used,
        human_used=human_used,
        ai_capped=cap_to_human,
        salt=salt,
    )
    return BuildPlan(
        rows=rows,
        manifest=manifest,
        corpus_sha256=corpus_sha256,
        targets=targets,
        available=available,
        human_chunks=human_chunks,
    )


def write_corpus(plan: BuildPlan, *, out_path: Path, manifest_path: Path) -> str:
    """Atomically write the corpus and manifest; returns the corpus SHA-256."""
    payload = serialize_rows(plan.rows)
    digest = hashlib.sha256(payload).hexdigest()
    if digest != plan.corpus_sha256:
        raise ValueError("corpus changed between planning and writing")
    _write_bytes_atomic(Path(out_path), payload)
    _write_json_atomic(Path(manifest_path), plan.manifest)
    return digest


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the E2-C1 training corpus (offline, CPU only)."
    )
    parser.add_argument("--ai", default=AI_PATH, help="E2-C1 merged generation JSONL")
    parser.add_argument("--human", default=HUMAN_PATH, help="canonical Ghostbuster JSONL")
    parser.add_argument("--out", default=OUT_CORPUS, help="output corpus JSONL")
    parser.add_argument(
        "--manifest", default=OUT_MANIFEST, help="output manifest JSON"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate and print the plan without writing any file",
    )
    parser.add_argument(
        "--cap-to-human",
        dest="cap_to_human",
        action="store_true",
        default=True,
        help="cap the AI side per partition to the available human chunks (default)",
    )
    parser.add_argument(
        "--no-cap",
        dest="cap_to_human",
        action="store_false",
        help="keep every AI row; the build fails if human chunks fall short",
    )
    return parser


def _print_plan(plan: BuildPlan) -> None:
    counts = plan.manifest["counts"]
    print(f"rows={counts['total']}")
    print(f"by_label={counts['by_label']}")
    print(f"by_partition={counts['by_partition']}")
    print(f"by_partition_label={counts['by_partition_label']}")
    print(f"by_generator={counts['by_generator']}")
    print(f"ai_requested={plan.manifest['ai_requested']}")
    print(f"ai_used={plan.manifest['ai_used']}")
    print(f"human_used={plan.manifest['human_used']}")
    print(f"human_chunks={len(plan.human_chunks)}")
    print(f"corpus_sha256={plan.corpus_sha256}")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    ai_path = Path(args.ai)
    human_path = Path(args.human)

    try:
        ai_records = read_jsonl(ai_path)
        human_records = read_jsonl(human_path)
    except FileNotFoundError as exc:
        print(f"input not found: {exc}", file=sys.stderr)
        return 2
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"invalid input: {exc}", file=sys.stderr)
        return 2

    try:
        plan = plan_corpus(
            ai_records,
            human_records,
            ai_path=ai_path,
            human_path=human_path,
            cap_to_human=args.cap_to_human,
        )
    except InsufficientHumanChunksError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        for partition in PARTITIONS:
            values = exc.details.get(partition)
            if values is None:
                continue
            print(
                f"  {partition}: need {values['needed']}, available "
                f"{values['available']}, short {values['short']}",
                file=sys.stderr,
            )
        return 2
    except (ValueError, KeyError) as exc:
        print(f"invalid input: {exc}", file=sys.stderr)
        return 2

    if args.dry_run:
        _print_plan(plan)
        print("dry-run: nothing written")
        return 0

    try:
        digest = write_corpus(
            plan, out_path=Path(args.out), manifest_path=Path(args.manifest)
        )
    except OSError as exc:
        print(f"write failed: {exc}", file=sys.stderr)
        return 2

    _print_plan(plan)
    print(f"wrote {args.out}")
    print(f"wrote {args.manifest}")
    print(f"corpus_sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
