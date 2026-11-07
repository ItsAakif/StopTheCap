"""Prompt-level, family-atomic split (§C).

This module deliberately does **not** call ``ml.data.splits.build_split_plan``: that
function maps the top bucket to ``Split.TEST_ID``, E1's frozen in-domain benchmark.
Every E2-C1 partition here is one of ``train``/``val``/``test_holdout``.
"""

from __future__ import annotations

import collections
import hashlib
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from .constants import (
    SPLIT_FRACTIONS,
    SPLIT_SALT,
    SPLIT_TOLERANCE,
    SPLIT_VERSION,
    TOPIC_TAXONOMY,
)
from .ids import make_sample_id

PARTITIONS: tuple[str, ...] = ("train", "val", "test_holdout")


class SplitToleranceError(RuntimeError):
    """Raised when a family-atomic split cannot meet the ±tolerance contract."""

    def __init__(self, counts: Mapping[str, int], targets: Mapping[str, int]) -> None:
        self.counts = dict(counts)
        self.targets = dict(targets)
        super().__init__(
            f"split counts {self.counts} exceed tolerance vs targets {self.targets}"
        )


def _field(row: Any, name: str) -> Any:
    if isinstance(row, Mapping):
        return row[name]
    return getattr(row, name)


def _maybe_field(row: Any, name: str) -> Any:
    if isinstance(row, Mapping):
        return row.get(name)
    return getattr(row, name, None)


def _topic(row: Any) -> str:
    return str(_maybe_field(row, "topic_category") or "other")


def _round_half_up(value: float) -> int:
    return int(math.floor(value + 0.5))


def e2c1_bucket(key: str, salt: str) -> int:
    """Deterministic bucket in ``[0, 99]`` with an explicit field separator.

    The existing E1 ``bucket_for`` concatenates ``key + salt`` without a separator,
    so ``("a", "bc")`` and ``("ab", "c")`` collide. That function is frozen because
    changing it would change E1's split digest; this is the E2-C1 replacement.
    """
    return int(hashlib.sha256(f"{key}\x1f{salt}".encode("utf-8")).hexdigest()[:8], 16) % 100


@dataclass(frozen=True)
class _Unit:
    family_id: str
    members: tuple[str, ...]
    topic: str
    rank: str

    @property
    def size(self) -> int:
        return len(self.members)


def _build_units(
    prompts: Sequence[Any],
    families: Mapping[str, str],
    salt: str,
) -> list[_Unit]:
    grouped: dict[str, list[str]] = collections.defaultdict(list)
    topic_of: dict[str, str] = {}
    for prompt in prompts:
        prompt_id = str(_field(prompt, "prompt_id"))
        if prompt_id not in families:
            raise ValueError(f"prompt {prompt_id} has no family assignment")
        grouped[str(families[prompt_id])].append(prompt_id)
        topic_of[prompt_id] = _topic(prompt)

    units: list[_Unit] = []
    for family_id, members in grouped.items():
        members = sorted(members)
        topic_counts = collections.Counter(topic_of[member] for member in members)
        # mode, tie-break lexicographic
        topic = sorted(topic_counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
        rank = hashlib.sha256(f"{salt}\x1f{family_id}".encode("utf-8")).hexdigest()
        units.append(
            _Unit(
                family_id=family_id,
                members=tuple(members),
                topic=topic,
                rank=rank,
            )
        )
    units.sort(key=lambda unit: (unit.rank, unit.family_id))
    return units


def _per_topic_quotas(
    units_by_topic: Mapping[str, Sequence[_Unit]],
    fractions: Mapping[str, float],
) -> dict[str, dict[str, int]]:
    quotas: dict[str, dict[str, int]] = {}
    for topic, units in units_by_topic.items():
        total = sum(unit.size for unit in units)
        raw = {partition: total * fractions[partition] for partition in PARTITIONS}
        quota = {partition: math.floor(raw[partition]) for partition in PARTITIONS}
        order = sorted(
            PARTITIONS,
            key=lambda partition: (
                -(raw[partition] - math.floor(raw[partition])),
                PARTITIONS.index(partition),
            ),
        )
        index = 0
        while sum(quota.values()) < total:
            quota[order[index % len(order)]] += 1
            index += 1
        quotas[topic] = quota
    return quotas


def _deviation(current: Mapping[str, int], targets: Mapping[str, int]) -> int:
    return sum(abs(current[partition] - targets[partition]) for partition in PARTITIONS)


def _units_by_partition(
    units: Sequence[_Unit], assignment: Mapping[str, str]
) -> dict[str, list[_Unit]]:
    out: dict[str, list[_Unit]] = {partition: [] for partition in PARTITIONS}
    for unit in units:
        out[assignment[unit.members[0]]].append(unit)
    for partition in out:
        out[partition].sort(key=lambda unit: (unit.rank, unit.family_id))
    return out


def assign_partitions(
    prompts: Sequence[Any],
    families: Mapping[str, str],
    salt: str = SPLIT_SALT,
) -> dict[str, str]:
    """Assign each prompt to ``train``/``val``/``test_holdout``.

    Families are never split. Units are filled topic-by-topic under per-topic
    largest-remainder quotas, then repaired globally with whole-unit moves and
    bounded swaps until every partition is within ``SPLIT_TOLERANCE`` prompts of the
    target. Failure raises :class:`SplitToleranceError`; an off-ratio split is never
    returned.
    """
    prompts = list(prompts)
    if not prompts:
        return {}

    units = _build_units(prompts, families, salt)
    target = {
        partition: _round_half_up(len(prompts) * SPLIT_FRACTIONS[partition])
        for partition in PARTITIONS
    }

    units_by_topic: dict[str, list[_Unit]] = collections.defaultdict(list)
    for unit in units:
        units_by_topic[unit.topic].append(unit)
    quotas = _per_topic_quotas(units_by_topic, SPLIT_FRACTIONS)

    assignment: dict[str, str] = {}
    current: dict[str, int] = {partition: 0 for partition in PARTITIONS}

    for topic in sorted(quotas):
        remaining = dict(quotas[topic])
        for unit in sorted(units_by_topic[topic], key=lambda item: (item.rank, item.family_id)):
            fits = [
                partition
                for partition in PARTITIONS
                if remaining[partition] >= unit.size
            ]
            if fits:
                partition = max(
                    fits,
                    key=lambda candidate: (
                        remaining[candidate],
                        -PARTITIONS.index(candidate),
                    ),
                )
            else:
                partition = max(
                    PARTITIONS,
                    key=lambda candidate: (
                        remaining[candidate],
                        -PARTITIONS.index(candidate),
                    ),
                )
            for prompt_id in unit.members:
                assignment[prompt_id] = partition
            remaining[partition] -= unit.size
            current[partition] += unit.size

    # ---- deterministic global repair -------------------------------------
    iterations = 0
    while max(abs(current[p] - target[p]) for p in PARTITIONS) > SPLIT_TOLERANCE:
        iterations += 1
        if iterations > 100_000:  # defensive; strict improvement terminates earlier
            break
        current_deviation = _deviation(current, target)
        donor = max(
            PARTITIONS,
            key=lambda p: (current[p] - target[p], -PARTITIONS.index(p)),
        )
        receiver = min(
            PARTITIONS,
            key=lambda p: (current[p] - target[p], PARTITIONS.index(p)),
        )
        by_partition = _units_by_partition(units, assignment)

        best_move: tuple[tuple[Any, ...], _Unit] | None = None
        for unit in by_partition[donor]:
            candidate = dict(current)
            candidate[donor] -= unit.size
            candidate[receiver] += unit.size
            score = _deviation(candidate, target)
            if score < current_deviation:
                key = (score, unit.rank, unit.family_id)
                if best_move is None or key < best_move[0]:
                    best_move = (key, unit)

        best_swap: tuple[tuple[Any, ...], _Unit, _Unit] | None = None
        if best_move is None:
            surplus = max(current[donor] - target[donor], 0)
            for donor_unit in by_partition[donor][:100]:
                for receiver_unit in by_partition[receiver][:100]:
                    if abs(donor_unit.size - receiver_unit.size) > surplus:
                        continue
                    candidate = dict(current)
                    candidate[donor] += receiver_unit.size - donor_unit.size
                    candidate[receiver] += donor_unit.size - receiver_unit.size
                    score = _deviation(candidate, target)
                    if score < current_deviation:
                        key = (
                            score,
                            donor_unit.rank,
                            receiver_unit.rank,
                            donor_unit.family_id,
                            receiver_unit.family_id,
                        )
                        if best_swap is None or key < best_swap[0]:
                            best_swap = (key, donor_unit, receiver_unit)

        if best_move is not None:
            _key, unit = best_move
            for prompt_id in unit.members:
                assignment[prompt_id] = receiver
            current[donor] -= unit.size
            current[receiver] += unit.size
            continue

        if best_swap is not None:
            _key, donor_unit, receiver_unit = best_swap
            for prompt_id in donor_unit.members:
                assignment[prompt_id] = receiver
            for prompt_id in receiver_unit.members:
                assignment[prompt_id] = donor
            current[donor] += receiver_unit.size - donor_unit.size
            current[receiver] += donor_unit.size - receiver_unit.size
            continue

        break

    if max(abs(current[p] - target[p]) for p in PARTITIONS) > SPLIT_TOLERANCE:
        raise SplitToleranceError(current, target)

    return {prompt_id: assignment[prompt_id] for prompt_id in sorted(assignment)}


def e2c1_split_digest(assignments: Mapping[str, str], salt: str) -> str:
    """Order-independent digest of ``(prompt_id, partition)`` plus the salt."""
    digest = hashlib.sha256()
    digest.update(salt.encode("utf-8"))
    for prompt_id in sorted(assignments):
        digest.update(b"\n")
        digest.update(prompt_id.encode("utf-8"))
        digest.update(b"\t")
        digest.update(assignments[prompt_id].encode("utf-8"))
    return digest.hexdigest()


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _config_generators(config: Mapping[str, Any]) -> list[tuple[str, str]]:
    return [
        (str(generator["generator_id"]), str(generator.get("model_revision", "")))
        for generator in config.get("generators", [])
    ]


def _config_decodings(config: Mapping[str, Any]) -> list[str]:
    return [str(decoding["decoding_config_id"]) for decoding in config.get("decodings", [])]


def build_split_manifest(
    prompts: Sequence[Any],
    assignments: Mapping[str, str],
    config: Mapping[str, Any],
    prompt_manifest_sha256: str,
    *,
    salt: str = SPLIT_SALT,
) -> dict:
    """Build the §C.1 split-manifest object; ``split_digest`` is computed last."""
    rows = list(prompts)
    assignment = {str(key): str(value) for key, value in assignments.items()}
    topics = sorted(
        set(TOPIC_TAXONOMY)
        | {_topic(row) for row in rows}
    )
    generator_ids = [generator_id for generator_id, _revision in _config_generators(config)]
    decoding_ids = _config_decodings(config)

    by_partition = {partition: 0 for partition in PARTITIONS}
    by_partition_topic = {
        partition: {topic: 0 for topic in topics} for partition in PARTITIONS
    }
    by_partition_generator_decoding = {
        partition: {
            generator_id: {decoding_id: 0 for decoding_id in decoding_ids}
            for generator_id in generator_ids
        }
        for partition in PARTITIONS
    }
    by_partition_generator_decoding_topic = {
        partition: {
            generator_id: {
                decoding_id: {topic: 0 for topic in topics}
                for decoding_id in decoding_ids
            }
            for generator_id in generator_ids
        }
        for partition in PARTITIONS
    }

    for row in rows:
        prompt_id = str(_field(row, "prompt_id"))
        partition = assignment[prompt_id]
        topic = _topic(row)
        by_partition[partition] += 1
        by_partition_topic[partition][topic] += 1
        for generator_id in generator_ids:
            for decoding_id in decoding_ids:
                by_partition_generator_decoding[partition][generator_id][decoding_id] += 1
                by_partition_generator_decoding_topic[partition][generator_id][decoding_id][
                    topic
                ] += 1

    family_sizes: collections.Counter[str] = collections.Counter()
    for row in rows:
        prompt_id = str(_field(row, "prompt_id"))
        family_id = _maybe_field(row, "family_id")
        family_sizes[str(family_id or prompt_id)] += 1
    size_histogram = collections.Counter(family_sizes.values())

    manifest: dict[str, Any] = {
        "schema_version": "1.0",
        "split_version": SPLIT_VERSION,
        "generated_at": _now_iso(),
        "salt": salt,
        "fractions": dict(SPLIT_FRACTIONS),
        "tolerance": SPLIT_TOLERANCE,
        "prompt_manifest": str(config.get("prompt_manifest", "")),
        "prompt_manifest_sha256": prompt_manifest_sha256,
        "n_prompts": len(rows),
        "partitions": list(PARTITIONS),
        "counts": {
            "by_partition": by_partition,
            "by_partition_topic": by_partition_topic,
            "by_partition_generator_decoding": by_partition_generator_decoding,
            "by_partition_generator_decoding_topic": by_partition_generator_decoding_topic,
        },
        "families": {
            "n": len(family_sizes),
            "size_histogram": {
                str(size): size_histogram[size] for size in sorted(size_histogram)
            },
            "max_size": max(size_histogram) if size_histogram else 0,
        },
        "assignments": {
            prompt_id: assignment[prompt_id] for prompt_id in sorted(assignment)
        },
    }
    manifest["split_digest"] = e2c1_split_digest(manifest["assignments"], salt)
    return manifest


def validate_split_manifest(
    split: Mapping[str, Any], prompts: Sequence[Any]
) -> list[str]:
    """Return human-readable split violations (empty means clean)."""
    problems: list[str] = []
    if not isinstance(split, Mapping):
        return ["split manifest is not an object"]

    assignments = split.get("assignments")
    if not isinstance(assignments, Mapping):
        return ["split manifest has no assignments object"]
    assignment = {str(key): str(value) for key, value in assignments.items()}
    rows = list(prompts)

    ids = [str(_field(row, "prompt_id")) for row in rows]
    idset = set(ids)
    if len(ids) != len(idset):
        problems.append("duplicate prompt_id in prompt manifest")

    missing = sorted(idset - set(assignment))
    extra = sorted(set(assignment) - idset)
    for prompt_id in missing:
        problems.append(f"prompt {prompt_id} has no partition assignment")
    for prompt_id in extra:
        problems.append(f"assignment references unknown prompt {prompt_id}")

    allowed = set(PARTITIONS)
    unknown = sorted(set(assignment.values()) - allowed)
    for partition in unknown:
        problems.append(f"unknown partition {partition!r}")
    if "test_id" in set(assignment.values()):
        problems.append(
            "partition 'test_id' is forbidden: no E2-C1 prompt may map to TEST_ID"
        )

    header = split.get("partitions")
    if header is not None and list(header) != list(PARTITIONS):
        problems.append(f"partitions header {list(header)!r} != {list(PARTITIONS)!r}")

    if split.get("n_prompts") != len(ids):
        problems.append(f"n_prompts {split.get('n_prompts')!r} != {len(ids)}")

    # ---- family atomicity ------------------------------------------------
    family_partitions: dict[str, set[str]] = collections.defaultdict(set)
    for row in rows:
        prompt_id = str(_field(row, "prompt_id"))
        family_id = _maybe_field(row, "family_id")
        if not family_id or prompt_id not in assignment:
            continue
        family_partitions[str(family_id)].add(assignment[prompt_id])
    for family_id, partitions in sorted(family_partitions.items()):
        if len(partitions) > 1:
            problems.append(
                f"family {family_id} crosses partitions {sorted(partitions)}"
            )

    # ---- exact duplicates cannot cross partitions ------------------------
    hash_rows: dict[str, list[str]] = collections.defaultdict(list)
    for row in rows:
        prompt_hash = _maybe_field(row, "prompt_hash")
        if prompt_hash:
            hash_rows[str(prompt_hash)].append(str(_field(row, "prompt_id")))
    for prompt_hash, prompt_ids in sorted(hash_rows.items()):
        if len(prompt_ids) > 1:
            partitions = sorted(
                {assignment.get(prompt_id, "<unassigned>") for prompt_id in prompt_ids}
            )
            problems.append(
                f"duplicate prompt_hash {prompt_hash} appears {len(prompt_ids)} times "
                f"across partitions {partitions}"
            )

    # ---- digest ----------------------------------------------------------
    expected_digest = e2c1_split_digest(
        assignment, str(split.get("salt", SPLIT_SALT))
    )
    if split.get("split_digest") != expected_digest:
        problems.append(
            f"split_digest mismatch: {split.get('split_digest')!r} != {expected_digest}"
        )

    # ---- counts ----------------------------------------------------------
    counts = split.get("counts")
    if not isinstance(counts, Mapping):
        problems.append("split manifest has no counts object")
        return problems

    actual_by_partition = {partition: 0 for partition in PARTITIONS}
    for partition in assignment.values():
        if partition in actual_by_partition:
            actual_by_partition[partition] += 1

    declared_by_partition = counts.get("by_partition") or {}
    for partition in PARTITIONS:
        if declared_by_partition.get(partition) != actual_by_partition[partition]:
            problems.append(
                f"by_partition[{partition}] {declared_by_partition.get(partition)!r} "
                f"!= {actual_by_partition[partition]}"
            )

    topics = sorted(
        set(TOPIC_TAXONOMY)
        | {_topic(row) for row in rows}
    )
    expected_topic = {
        partition: {topic: 0 for topic in topics} for partition in PARTITIONS
    }
    for row in rows:
        prompt_id = str(_field(row, "prompt_id"))
        partition = assignment.get(prompt_id)
        topic = _topic(row)
        if partition in expected_topic and topic in expected_topic[partition]:
            expected_topic[partition][topic] += 1

    declared_topic = counts.get("by_partition_topic") or {}
    for partition in PARTITIONS:
        block = declared_topic.get(partition) or {}
        for topic in topics:
            if block.get(topic) != expected_topic[partition][topic]:
                problems.append(
                    f"by_partition_topic[{partition}][{topic}] "
                    f"{block.get(topic)!r} != {expected_topic[partition][topic]}"
                )

    declared_gd = counts.get("by_partition_generator_decoding") or {}
    generator_ids = sorted(
        {
            str(generator_id)
            for block in declared_gd.values()
            if isinstance(block, Mapping)
            for generator_id in block
        }
    )
    decoding_ids = sorted(
        {
            str(decoding_id)
            for block in declared_gd.values()
            if isinstance(block, Mapping)
            for generator_block in block.values()
            if isinstance(generator_block, Mapping)
            for decoding_id in generator_block
        }
    )
    if not generator_ids:
        problems.append("by_partition_generator_decoding has no generator cells")
    if not decoding_ids:
        problems.append("by_partition_generator_decoding has no decoding cells")
    for partition in PARTITIONS:
        block = declared_gd.get(partition) or {}
        for generator_id in generator_ids:
            generator_block = block.get(generator_id) or {}
            for decoding_id in decoding_ids:
                actual = generator_block.get(decoding_id)
                expected = actual_by_partition[partition]
                if actual != expected:
                    problems.append(
                        f"by_partition_generator_decoding[{partition}][{generator_id}]"
                        f"[{decoding_id}] {actual!r} != {expected}"
                    )
                if not isinstance(actual, int) or actual <= 0:
                    problems.append(
                        f"planned generation cell "
                        f"[{partition}][{generator_id}][{decoding_id}] is not > 0"
                    )

    declared_gdt = counts.get("by_partition_generator_decoding_topic") or {}
    for partition in PARTITIONS:
        block = declared_gdt.get(partition) or {}
        for generator_id in generator_ids:
            generator_block = block.get(generator_id) or {}
            for decoding_id in decoding_ids:
                topic_block = generator_block.get(decoding_id) or {}
                for topic in topics:
                    expected = expected_topic[partition][topic]
                    if topic_block.get(topic) != expected:
                        problems.append(
                            f"by_partition_generator_decoding_topic[{partition}]"
                            f"[{generator_id}][{decoding_id}][{topic}] "
                            f"{topic_block.get(topic)!r} != {expected}"
                        )

    # ---- ratio tolerance -------------------------------------------------
    fractions = split.get("fractions") or SPLIT_FRACTIONS
    tolerance = int(split.get("tolerance", SPLIT_TOLERANCE))
    targets = {
        partition: _round_half_up(len(ids) * float(fractions[partition]))
        for partition in PARTITIONS
    }
    for partition in PARTITIONS:
        if abs(actual_by_partition[partition] - targets[partition]) > tolerance:
            problems.append(
                f"partition {partition} count {actual_by_partition[partition]} "
                f"is outside ±{tolerance} of target {targets[partition]}"
            )

    return problems


def expected_sample_ids(
    prompts: Sequence[Any],
    assignments: Mapping[str, str],
    config: Mapping[str, Any],
) -> list[str]:
    """Every planned sample id: prompts × generators × decodings (§D.5)."""
    del assignments  # the id depends only on the four §D.5 components
    generators = _config_generators(config)
    decoding_ids = _config_decodings(config)
    out: list[str] = []
    for row in sorted(prompts, key=lambda item: str(_field(item, "prompt_id"))):
        prompt_id = str(_field(row, "prompt_id"))
        for generator_id, model_revision in generators:
            for decoding_id in decoding_ids:
                out.append(
                    make_sample_id(
                        prompt_id, generator_id, model_revision, decoding_id
                    )
                )
    return out
