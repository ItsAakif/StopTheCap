"""Group construction and deterministic split assignment.

Two guarantees the project depends on:

1. **Related samples share a split.** Grouping is a union-find over every linkage key
   a record carries, so a record linked by *either* a prompt id *or* an author id ends
   up in one component.
2. **Splits are reproducible.** Assignment is a pure function of
   ``sha256(group_id + salt)`` -- independent of row order, file order, or when the
   data was fetched. A second person rebuilding from the same manifest and salt gets
   byte-identical split hashes.

Role enforcement (never silent):

* datasets whose manifest says ``permitted_uses.train: false`` are pinned to
  ``QUARANTINE`` and can never be assigned to train/val;
* records flagged ``too_short``/``empty_text`` are ``EXCLUDED`` before assignment;
* a generator holdout is chosen deterministically from *trainable* records only.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from ..ingest.schema import CanonicalRecord, Split

DEFAULT_SALT = "stopthecap-2026-10-29"


class UnionFind:
    def __init__(self) -> None:
        self._p: dict[str, str] = {}

    def find(self, x: str) -> str:
        p = self._p
        if x not in p:
            p[x] = x
            return x
        root = x
        while p[root] != root:
            root = p[root]
        while p[x] != root:  # path compression
            p[x], x = root, p[x]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            # deterministic: lexicographically smaller root wins
            if ra < rb:
                self._p[rb] = ra
            else:
                self._p[ra] = rb

    def components(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = defaultdict(list)
        for x in list(self._p):
            out[self.find(x)].append(x)
        return dict(out)


def _linkage_keys(rec: CanonicalRecord) -> list[str]:
    keys = [f"g:{rec.group_id}"]
    if rec.identity_group:
        keys.append(f"id:{rec.identity_group}")
    if rec.modality.value == "text" and rec.source_dataset == "raid":
        # RAID source_id already forms part of group_id; nothing extra.
        pass
    return keys


@dataclass
class SplitPlan:
    salt: str
    assignments: dict[str, Split] = field(default_factory=dict)   # group_id -> split
    n_groups: int = 0
    held_out_generator: str | None = None
    held_out_domain: str | None = None

    def digest(self) -> str:
        """Stable hash of the whole plan: the 'split hash' quoted in reports."""
        h = hashlib.sha256()
        h.update(self.salt.encode())
        h.update(str(self.held_out_generator).encode())
        h.update(str(self.held_out_domain).encode())
        for gid in sorted(self.assignments):
            h.update(gid.encode())
            h.update(self.assignments[gid].value.encode())
        return h.hexdigest()


def bucket_for(group_id: str, salt: str) -> int:
    return int(hashlib.sha256(f"{group_id}{salt}".encode()).hexdigest()[:8], 16) % 100


def build_split_plan(
    records: Sequence[CanonicalRecord],
    train_allowed_datasets: Iterable[str],
    salt: str = DEFAULT_SALT,
    train_frac: int = 80,
    val_frac: int = 10,
    generator_holdout: str | None = None,
    domain_holdout: str | None = None,
) -> SplitPlan:
    trainable_datasets = set(train_allowed_datasets)

    # 1. union-find over linkage keys
    uf = UnionFind()
    for rec in records:
        keys = _linkage_keys(rec)
        first = keys[0]
        uf.find(first)
        for k in keys[1:]:
            uf.union(first, k)

    # 2. per-component facts
    comp_trainable: dict[str, bool] = defaultdict(bool)
    comp_generators: dict[str, set[str]] = defaultdict(set)
    comp_domains: dict[str, set[str]] = defaultdict(set)
    for rec in records:
        root = uf.find(f"g:{rec.group_id}")
        allowed = rec.source_dataset in trainable_datasets and not rec.metadata.get("_eval_only")
        # A component is trainable only if EVERY record in it is trainable. A mixed
        # component must be quarantined, not promoted.
        if root in comp_trainable:
            comp_trainable[root] = comp_trainable[root] and allowed
        else:
            comp_trainable[root] = allowed
        if rec.generator:
            comp_generators[root].add(rec.generator)
        if rec.domain:
            comp_domains[root].add(rec.domain)

    plan = SplitPlan(salt=salt, held_out_generator=generator_holdout,
                     held_out_domain=domain_holdout)

    # 3. assign by deterministic bucket
    for root in sorted(uf.components()):
        if not comp_trainable[root]:
            split = Split.QUARANTINE
        elif generator_holdout and generator_holdout in comp_generators[root]:
            split = Split.TEST_OOD_GENERATOR
        elif domain_holdout and comp_domains[root] and comp_domains[root] <= {domain_holdout}:
            split = Split.TEST_OOD_DOMAIN
        else:
            b = bucket_for(root, salt)
            if b < train_frac:
                split = Split.TRAIN
            elif b < train_frac + val_frac:
                split = Split.VAL
            else:
                split = Split.TEST_ID
        plan.assignments[root] = split

    plan.n_groups = len(plan.assignments)
    return plan


def apply_plan(
    records: Sequence[CanonicalRecord],
    plan: SplitPlan,
) -> list[CanonicalRecord]:
    """Return copies of records with ``split`` set from the plan."""
    from ..ingest.schema import make_sample_id  # local import avoids cycle

    uf = UnionFind()
    for rec in records:
        keys = _linkage_keys(rec)
        uf.find(keys[0])
        for k in keys[1:]:
            uf.union(keys[0], k)

    out: list[CanonicalRecord] = []
    for rec in records:
        flags = rec.metadata.get("quality_flags") or []
        reason = rec.metadata.get("exclusion_reason")
        if reason or any(f in flags for f in ("too_short", "empty_text", "invalid_label")):
            # An explicit exclusion (quality flag OR duplicate detection) must never
            # be overwritten by split assignment.
            split = Split.EXCLUDED
        else:
            root = uf.find(f"g:{rec.group_id}")
            split = plan.assignments.get(root, Split.EXCLUDED)

        if split.value == rec.split.value:
            out.append(rec)
        else:
            out.append(rec.model_copy(update={"split": split}))
    return out


def assert_plan_invariants(records: Sequence[CanonicalRecord]) -> list[str]:
    """Return a list of violations (empty means clean)."""
    problems: list[str] = []

    # no group may span two splits
    seen: dict[str, str] = {}
    for rec in records:
        if rec.split is Split.EXCLUDED:
            continue
        prev = seen.get(rec.group_id)
        if prev is None:
            seen[rec.group_id] = rec.split.value
        elif prev != rec.split.value:
            problems.append(f"group {rec.group_id} spans {prev} and {rec.split.value}")

    # no evaluation-only dataset may be trainable
    for rec in records:
        if rec.split.is_trainable and rec.metadata.get("_eval_only"):
            problems.append(f"eval-only record {rec.sample_id} in {rec.split.value}")
    return problems


def split_digest(records: Sequence[CanonicalRecord], salt: str) -> str:
    """Stable hash of the realised split assignment.

    This is the single source of truth used both when writing the manifest and when
    verifying reproducibility later -- having two implementations guarantees they
    eventually disagree.
    """
    h = hashlib.sha256()
    h.update(salt.encode())
    seen: dict[str, str] = {}
    for r in records:
        # group ids may be rewritten during dedup merging; take the final value
        if r.group_id not in seen:
            seen[r.group_id] = r.split.value
    for gid in sorted(seen):
        h.update(gid.encode())
        h.update(seen[gid].encode())
    return h.hexdigest()


def split_counts(records: Sequence[CanonicalRecord]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for rec in records:
        counts[rec.split.value] += 1
    return dict(sorted(counts.items()))


def group_counts(records: Sequence[CanonicalRecord]) -> dict[str, int]:
    seen: dict[str, set[str]] = defaultdict(set)
    for rec in records:
        if rec.split is not Split.EXCLUDED:
            seen[rec.split.value].add(rec.group_id)
    return {k: len(v) for k, v in sorted(seen.items())}
