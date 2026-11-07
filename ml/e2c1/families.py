"""Near-duplicate prompt families (§B).

The 0.85 Jaccard relation *groups* prompts; it never deletes them. A family is the
connected component of the graph with two edge kinds:

* a near-duplicate pair (:func:`ml.data.dedup.find_duplicates`, exact + near), and
* any two prompts sharing a ``page_id``.

``ml.data.dedup.find_duplicates`` ignores its ``group_ids`` argument, so page-id
unions are added here explicitly (SPEC §H6). Nothing in this module relies on input
order or Python's salted ``hash()``.
"""

from __future__ import annotations

import collections
import hashlib
import re
from typing import Any, Mapping, Sequence

from ml.data.dedup import find_duplicates
from ml.data.splits import UnionFind

from .constants import FAMILY_ID_RE, NEAR_DUP_THRESHOLD

_FAMILY_ID_RE = re.compile(FAMILY_ID_RE)


def _field(row: Any, name: str) -> Any:
    if isinstance(row, Mapping):
        return row[name]
    return getattr(row, name)


def _maybe_field(row: Any, name: str) -> Any:
    if isinstance(row, Mapping):
        return row.get(name)
    return getattr(row, name, None)


def _text(row: Any) -> str:
    if isinstance(row, Mapping):
        if "prompt_text" in row:
            return str(row["prompt_text"] or "")
        return str(row.get("text") or "")
    return str(getattr(row, "prompt_text", "") or "")


def near_dup_pairs(
    texts: Sequence[str], threshold: float = NEAR_DUP_THRESHOLD
) -> list[tuple[int, int, float]]:
    """Return ordered ``(i, j, jaccard)`` index pairs at ``jaccard >= threshold``.

    Exact duplicates are included: their Jaccard is 1.0, so they are members of the
    same >=0.85 relation. ``find_duplicates`` reports them on its exact pass and
    then skips the (already-seen) pair on the near pass, so filtering to
    ``kind == "near"`` would silently drop them.
    """
    texts = list(texts)
    content_hashes = [hashlib.sha256(text.encode("utf-8")).hexdigest() for text in texts]
    pairs = find_duplicates(
        texts,
        content_hashes,
        [""] * len(texts),
        threshold=threshold,
    )
    out = [(pair.i, pair.j, float(pair.jaccard)) for pair in pairs]
    out.sort(key=lambda item: (item[0], item[1]))
    return out


def _components(
    prompts: Sequence[Any], threshold: float = NEAR_DUP_THRESHOLD
) -> tuple[list[list[str]], dict[str, int]]:
    """Return ``(components, stats)`` for the near-dup + same-page graph.

    Components and their members are sorted so the result does not depend on input
    order.
    """
    rows: list[tuple[str, str, Any]] = []
    for prompt in prompts:
        rows.append(
            (
                str(_field(prompt, "prompt_id")),
                _text(prompt),
                _maybe_field(prompt, "page_id"),
            )
        )

    uf = UnionFind()
    for prompt_id, _text_value, _page_id in rows:
        uf.find(prompt_id)

    pairs = near_dup_pairs([row[1] for row in rows], threshold)
    near_members: set[str] = set()
    for i, j, _jaccard in pairs:
        uf.union(rows[i][0], rows[j][0])
        near_members.add(rows[i][0])
        near_members.add(rows[j][0])

    page_groups: dict[Any, list[str]] = collections.defaultdict(list)
    for prompt_id, _text_value, page_id in rows:
        if page_id is None:
            continue
        page_groups[page_id].append(prompt_id)

    page_edges = 0
    for members in page_groups.values():
        members = sorted(members)
        page_edges += len(members) * (len(members) - 1) // 2
        for member in members[1:]:
            uf.union(members[0], member)

    components = [sorted(members) for members in uf.components().values()]
    components.sort(key=lambda members: members[0] if members else "")

    stats = {
        "near_dup_edges": len(pairs),
        "near_dup_members_retained": len(near_members),
        "page_edges": page_edges,
    }
    return components, stats


def _family_id(members: Sequence[str]) -> str:
    return "fam:" + hashlib.sha256(min(members).encode("utf-8")).hexdigest()[:16]


def build_families_with_stats(
    prompts: Sequence[Any], threshold: float = NEAR_DUP_THRESHOLD
) -> tuple[dict[str, str], dict[str, Any]]:
    """Build the family map and the audit statistics in one pass.

    Returns ``(families, stats)`` where ``families`` is ``prompt_id -> family_id``
    and ``stats`` contains ``n``, ``size_histogram``, ``max_size``,
    ``near_dup_edges``, ``near_dup_members_retained`` and ``page_edges``.
    """
    components, graph_stats = _components(prompts, threshold)

    families: dict[str, str] = {}
    histogram: collections.Counter[int] = collections.Counter()
    for members in components:
        if not members:
            continue
        family_id = _family_id(members)
        for prompt_id in members:
            families[prompt_id] = family_id
        histogram[len(members)] += 1

    stats: dict[str, Any] = {
        "n": len(components),
        "size_histogram": {str(size): histogram[size] for size in sorted(histogram)},
        "max_size": max(histogram) if histogram else 0,
        **graph_stats,
    }
    return families, stats


def build_families(
    prompts: Sequence[Any], threshold: float = NEAR_DUP_THRESHOLD
) -> dict[str, str]:
    """Return ``prompt_id -> family_id``; every prompt is its own node."""
    families, _stats = build_families_with_stats(prompts, threshold)
    return families


def validate_families(
    prompts: Sequence[Any], families: Mapping[str, str]
) -> list[str]:
    """Return human-readable family violations (empty means clean)."""
    problems: list[str] = []

    prompt_ids: list[str] = []
    for prompt in prompts:
        prompt_ids.append(str(_field(prompt, "prompt_id")))

    if len(prompt_ids) != len(set(prompt_ids)):
        problems.append("duplicate prompt_id in prompt set")

    idset = set(prompt_ids)
    family_map = dict(families)

    missing = sorted(idset - set(family_map))
    extra = sorted(set(family_map) - idset)
    for prompt_id in missing:
        problems.append(f"prompt {prompt_id} has no family")
    for prompt_id in extra:
        problems.append(f"family assignment references unknown prompt {prompt_id}")

    components, _stats = _components(prompts)
    component_of: dict[str, int] = {}
    for index, members in enumerate(components):
        for prompt_id in members:
            component_of[prompt_id] = index

    # Every family id must cover exactly one component and be hashed from its
    # minimum member.
    members_by_family: dict[str, set[str]] = collections.defaultdict(set)
    for prompt_id, family_id in family_map.items():
        if prompt_id in idset:
            members_by_family[str(family_id)].add(prompt_id)

    for family_id, members in sorted(members_by_family.items()):
        if not _FAMILY_ID_RE.match(family_id):
            problems.append(f"invalid family id {family_id!r}")
            continue
        if not members:
            problems.append(f"family {family_id} has no members")
            continue
        expected = _family_id(sorted(members))
        if family_id != expected:
            problems.append(
                f"family {family_id} is not hashed from min(member prompt_ids) "
                f"(expected {expected})"
            )
        component_indexes = {component_of[m] for m in members if m in component_of}
        if len(component_indexes) > 1:
            problems.append(
                f"family id {family_id} is reused by {len(component_indexes)} "
                "disconnected components"
            )

    # Every component must carry exactly one family id.
    for members in components:
        assigned = {family_map[prompt_id] for prompt_id in members if prompt_id in family_map}
        if len(assigned) > 1:
            problems.append(
                f"component starting {members[0]} spans {len(assigned)} family ids"
            )

    return problems
