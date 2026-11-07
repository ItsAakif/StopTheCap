"""Build the E2-C1 prompt manifest and audit.

Stages 1-5 are offline. Stages 6-17 need a :class:`~ml.e2c1.wikipedia.WikipediaClient`
(real, cached, or injected in tests). ``--dry-run`` executes only stages 1-5 and
prints the stage table; it never touches the network or writes the manifest.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from ml.ingest.normalise import normalise_text, word_count

from . import prompts
from .constants import (
    LATIN_RATIO_MIN,
    MAX_PROMPTS_PER_PAGE,
    MAX_WORDS,
    MIN_WORDS,
    PIPELINE_VERSION,
    SAFETY_RULESET_VERSION,
    SELECTION_SEED,
    SOURCE_LICENCE,
    TITLE_BLOCKLIST_RE,
    TOPIC_TAXONOMY,
)
from .prompts import (
    InsufficientPoolError,
    make_attribution,
    make_prompt_hash,
    make_prompt_id,
    make_source_url,
    safety_status,
    topic_category,
    write_manifest,
)
from .families import build_families_with_stats
from .wikipedia import WikipediaClient, WikipediaError, WikipediaOfflineError

STAGE_NAMES: tuple[str, ...] = (
    "exact_duplicate",
    "length_short",
    "length_long",
    "non_latin",
    "title_blocklist",
    "api_missing",
    "api_error",
    "api_no_extract",
    "revision_mismatch",
    "disambiguation",
    "list_page",
    "biography",
    "content_blocklist",
    "quote_heavy",
    "boilerplate",
    "page_cap",
    "rank_cutoff",
)

# Safety reason key -> audit stage name. ``blocklist`` is the §A.5 terminal reason;
# the pipeline stage is called ``content_blocklist`` to distinguish it from the
# title-only stage 5.
_REASON_TO_STAGE = {
    "non_latin": "non_latin",
    "length_short": "length_short",
    "length_long": "length_long",
    "disambiguation": "disambiguation",
    "list_page": "list_page",
    "biography": "biography",
    "blocklist": "content_blocklist",
    "quote_heavy": "quote_heavy",
    "boilerplate": "boilerplate",
}

_EXCLUSION_KEYS: tuple[str, ...] = (
    "exact_duplicate",
    "length_short",
    "length_long",
    "non_latin",
    "title_blocklist",
    "api_missing",
    "api_error",
    "api_no_extract",
    "revision_mismatch",
    "disambiguation",
    "list_page",
    "biography",
    "blocklist",
    "quote_heavy",
    "boilerplate",
    "page_cap",
    "rank_cutoff",
)

_TITLE_BLOCKLIST = re.compile(TITLE_BLOCKLIST_RE, re.IGNORECASE)
_API_ERROR = object()

DEFAULT_INPUT = "research/_evidence/wikipedia.jsonl"
DEFAULT_OUT = "datasets/manifests/e2_c1_prompts.jsonl"
DEFAULT_AUDIT = "datasets/manifests/e2_c1_prompt_audit.json"


# --------------------------------------------------------------------------
# Input / stage helpers
# --------------------------------------------------------------------------
def read_input(path: Path, limit: int | None = None) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if limit is not None and len(rows) >= limit:
                break
            if not line.strip():
                continue
            obj = json.loads(line)
            rows.append(
                {
                    "page_title": str(obj.get("page_title", "")),
                    "human_text": str(obj.get("human_text") or ""),
                }
            )
    return rows


def _record_stage(
    stages: list[dict[str, int | str]], name: str, in_count: int, out_count: int
) -> None:
    stages.append(
        {
            "stage": name,
            "in": in_count,
            "removed": in_count - out_count,
            "out": out_count,
        }
    )


def run_offline_stages(
    rows: Sequence[Mapping[str, str]],
    exclusions: collections.Counter[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, int | str]]]:
    """Stages 1-5. Returns ``(survivors, stages)`` and fills ``exclusions``."""
    if exclusions is None:
        exclusions = collections.Counter()
    stages: list[dict[str, int | str]] = []

    prepared: list[dict[str, Any]] = []
    for row in rows:
        text = normalise_text(str(row.get("human_text") or ""))
        prepared.append(
            {
                "page_title": str(row.get("page_title") or ""),
                "human_text": str(row.get("human_text") or ""),
                "prompt_text": text,
                "prompt_hash": make_prompt_hash(text),
                "words": word_count(text),
            }
        )

    # Stage 1: exact duplicate on the normalised text; keep the lexicographically
    # smallest (title, original text).
    groups: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in prepared:
        groups[row["prompt_hash"]].append(row)
    kept: list[dict[str, Any]] = []
    for digest in sorted(groups):
        group = sorted(
            groups[digest],
            key=lambda r: (r["page_title"], r["human_text"]),
        )
        kept.append(group[0])
        for _ in group[1:]:
            exclusions["exact_duplicate"] += 1
    _record_stage(stages, "exact_duplicate", len(prepared), len(kept))

    # Stages 2-3: length.
    current = kept
    short = [row for row in current if row["words"] < MIN_WORDS]
    kept = [row for row in current if row["words"] >= MIN_WORDS]
    for _ in short:
        exclusions["length_short"] += 1
    _record_stage(stages, "length_short", len(current), len(kept))

    current = kept
    long_rows = [row for row in current if row["words"] > MAX_WORDS]
    kept = [row for row in current if row["words"] <= MAX_WORDS]
    for _ in long_rows:
        exclusions["length_long"] += 1
    _record_stage(stages, "length_long", len(current), len(kept))

    # Stage 4: script.
    current = kept
    non_latin = [
        row for row in current if prompts.latin_ratio(row["prompt_text"]) < LATIN_RATIO_MIN
    ]
    kept = [
        row for row in current if prompts.latin_ratio(row["prompt_text"]) >= LATIN_RATIO_MIN
    ]
    for _ in non_latin:
        exclusions["non_latin"] += 1
    _record_stage(stages, "non_latin", len(current), len(kept))

    # Stage 5: title-only blocklist.
    current = kept
    kept = [row for row in current if not _TITLE_BLOCKLIST.search(row["page_title"])]
    for _ in range(len(current) - len(kept)):
        exclusions["title_blocklist"] += 1
    _record_stage(stages, "title_blocklist", len(current), len(kept))

    return kept, stages


def format_stage_table(stages: Sequence[Mapping[str, Any]]) -> str:
    header = f"{'stage':<20} {'in':>6} {'removed':>8} {'out':>6}"
    lines = [header, "-" * len(header)]
    for stage in stages:
        lines.append(
            f"{str(stage['stage']):<20} {int(stage['in']):>6} "
            f"{int(stage['removed']):>8} {int(stage['out']):>6}"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------
# API stages
# --------------------------------------------------------------------------
def _fetch_snapshots(
    titles: Sequence[str], client: Any
) -> dict[str, Any]:
    """Fetch in <=20-title chunks, isolating failures to single titles."""
    snapshots: dict[str, Any] = {}
    titles = sorted(dict.fromkeys(titles))
    total = len(titles)
    errors = 0
    batches = 0
    started = time.monotonic()
    last_report = started
    for start in range(0, len(titles), 20):
        chunk = titles[start : start + 20]
        try:
            snapshots.update(client.fetch_pages(chunk))
        except WikipediaOfflineError:
            raise
        except WikipediaError:
            # A batch failure loses the other results; isolate each title so one
            # bad page cannot hide the rest of the batch.
            for title in chunk:
                try:
                    snapshots.update(client.fetch_pages([title]))
                except WikipediaOfflineError:
                    raise
                except WikipediaError:
                    snapshots[title] = _API_ERROR
                    errors += 1
        batches += 1
        now = time.monotonic()
        if batches % 25 == 0 or now - last_report >= 5.0:
            stats_fn = getattr(client, "stats", None)
            cached = int(stats_fn().get("cache_hits", 0)) if callable(stats_fn) else 0
            print(
                f"resolved={len(snapshots)}/{total} cached={cached} "
                f"errors={errors} elapsed={int(now - started)}s",
                file=sys.stderr,
            )
            last_report = now
    return snapshots


def _run_api_stages(
    survivors: Sequence[Mapping[str, Any]],
    client: Any,
    stages: list[dict[str, int | str]],
    exclusions: collections.Counter[str],
    multi_reason: collections.Counter[str],
) -> list[dict[str, Any]]:
    snapshots = _fetch_snapshots(
        sorted({str(row["page_title"]) for row in survivors}), client
    )

    current = list(survivors)

    # Stages 6-9: resolution / revision check, applied as a sequential funnel.
    for name in (
        "api_missing",
        "api_error",
        "api_no_extract",
        "revision_mismatch",
    ):
        kept_rows: list[dict[str, Any]] = []
        for row in current:
            value = snapshots.get(str(row["page_title"]))
            if name == "api_missing" and value is None:
                exclusions["api_missing"] += 1
            elif name == "api_error" and value is _API_ERROR:
                exclusions["api_error"] += 1
            elif name == "api_no_extract" and not value.extract.strip():
                exclusions["api_no_extract"] += 1
            elif name == "revision_mismatch":
                haystack = normalise_text(value.extract)
                needle = str(row["prompt_text"])
                if not needle or needle not in haystack:
                    exclusions["revision_mismatch"] += 1
                else:
                    kept_rows.append(row)
            else:
                kept_rows.append(row)
        _record_stage(stages, name, len(current), len(kept_rows))
        current = kept_rows

    # Stages 10-15: safety rules. A row is removed at its first matched rule;
    # every additional matched rule is counted in ``multi_reason``.
    safety_stages = (
        "disambiguation",
        "list_page",
        "biography",
        "content_blocklist",
        "quote_heavy",
        "boilerplate",
    )
    for stage_name in safety_stages:
        kept_rows = []
        removed = 0
        for row in current:
            snapshot = snapshots[str(row["page_title"])]
            status, reasons = safety_status(
                title=snapshot.title,
                text=str(row["prompt_text"]),
                categories=snapshot.categories,
                pageprops=snapshot.pageprops,
            )
            if status == "ok":
                kept_rows.append(row)
                continue
            terminal = reasons[0]
            terminal_stage = _REASON_TO_STAGE.get(terminal)
            if terminal_stage == stage_name or (
                stage_name == "disambiguation" and terminal_stage not in safety_stages
            ):
                exclusions[terminal] += 1
                removed += 1
                if len(reasons) > 1:
                    multi_reason["+".join(sorted(reasons))] += 1
            else:
                kept_rows.append(row)
        _record_stage(stages, stage_name, len(current), len(current) - removed)
        current = kept_rows

    verified: list[dict[str, Any]] = []
    for row in current:
        snapshot = snapshots[str(row["page_title"])]
        topic, rule = topic_category(snapshot.categories, snapshot.title)
        candidate = dict(row)
        candidate.update(
            {
                "snapshot": snapshot,
                "topic_category": topic,
                "topic_rule": rule,
                "page_id": snapshot.page_id,
                "prompt_id": make_prompt_id(
                    snapshot.revision_id, str(row["prompt_hash"])
                ),
            }
        )
        verified.append(candidate)
    return verified


# --------------------------------------------------------------------------
# Audit
# --------------------------------------------------------------------------
def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, obj: Mapping[str, Any]) -> None:
    payload = (json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode(
        "utf-8"
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _new_audit(
    *,
    input_path: Path,
    input_rows: int,
    input_pages: int,
    input_sha: str,
    retrieved_at: str,
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "pipeline_version": PIPELINE_VERSION,
        "safety_ruleset_version": SAFETY_RULESET_VERSION,
        "status": "ok",
        "generated_at": _now_iso(),
        "input": {
            "path": str(input_path),
            "sha256": input_sha,
            "rows": input_rows,
            "pages": input_pages,
        },
        "retrieved_at": retrieved_at,
        "selection": {
            "rank_seed": SELECTION_SEED,
            "n_requested": 0,
            "candidates": 0,
            "selected": 0,
            "page_cap_removed": 0,
            "rank_cutoff_removed": 0,
        },
        "stages": [],
        "exclusions": {key: 0 for key in _EXCLUSION_KEYS},
        "multi_reason": {},
        "by_topic": {topic: 0 for topic in TOPIC_TAXONOMY},
        "families": {
            "n": 0,
            "size_histogram": {},
            "max_size": 0,
            "near_dup_edges": 0,
            "near_dup_members_retained": 0,
            "page_edges": 0,
        },
        "output": {"path": "", "sha256": None, "count": 0},
        "reconciliation": {"input": input_rows, "removed": 0, "selected": 0, "balanced": False},
        "warnings": [],
    }


def _finalize_reconciliation(audit: dict[str, Any]) -> None:
    removed = sum(int(stage["removed"]) for stage in audit["stages"])
    selected = int(audit["reconciliation"]["selected"])
    balanced = int(audit["reconciliation"]["input"]) - removed == selected
    audit["reconciliation"]["removed"] = removed
    audit["reconciliation"]["balanced"] = balanced


# --------------------------------------------------------------------------
# Full build
# --------------------------------------------------------------------------
def run_build(
    *,
    input_path: Path,
    out_path: Path,
    audit_path: Path,
    n: int = 5000,
    limit: int | None = None,
    client: Any = None,
    offline: bool = False,
    refresh: bool = False,
    cache_dir: Path | None = None,
    retrieved_at: str | None = None,
) -> tuple[dict[str, Any], int]:
    """Run stages 1-17, write the audit, and (when possible) the manifest.

    Returns ``(audit, exit_code)``. ``WikipediaOfflineError`` is propagated so the
    CLI can fail rather than silently treating a cache miss as an API error.
    """
    input_path = Path(input_path)
    rows = read_input(input_path, limit=limit)
    input_sha = _sha256_file(input_path)
    retrieved_at = retrieved_at or _today()
    exclusions: collections.Counter[str] = collections.Counter()
    multi_reason: collections.Counter[str] = collections.Counter()

    audit = _new_audit(
        input_path=input_path,
        input_rows=len(rows),
        input_pages=len({row["page_title"] for row in rows}),
        input_sha=input_sha,
        retrieved_at=retrieved_at,
    )
    audit["selection"]["n_requested"] = n

    survivors, stages = run_offline_stages(rows, exclusions)
    audit["stages"].extend(stages)

    if client is None:
        client = WikipediaClient(
            cache_dir=cache_dir,
            offline=offline,
            refresh=refresh,
        )

    verified = _run_api_stages(survivors, client, audit["stages"], exclusions, multi_reason)

    audit["selection"]["candidates"] = len(verified)
    try:
        chosen, page_cap_removed, rank_cutoff_removed = prompts._rank_and_select(
            verified, n, SELECTION_SEED, MAX_PROMPTS_PER_PAGE
        )
        exit_code = 0
        insufficient = False
    except InsufficientPoolError as exc:
        chosen = []
        page_cap_removed = 0
        rank_cutoff_removed = len(verified)
        exit_code = 3
        insufficient = True
        audit["status"] = "insufficient_pool"
        audit["shortfall"] = exc.shortfall
        audit["warnings"].append(
            f"verified pool {len(verified)} < requested {n}; manifest not written"
        )

    _record_stage(audit["stages"], "page_cap", len(verified), len(verified) - page_cap_removed)
    _record_stage(
        audit["stages"],
        "rank_cutoff",
        len(verified) - page_cap_removed,
        len(verified) - page_cap_removed - rank_cutoff_removed,
    )
    exclusions["page_cap"] += page_cap_removed
    exclusions["rank_cutoff"] += rank_cutoff_removed
    audit["selection"]["selected"] = len(chosen)
    audit["selection"]["page_cap_removed"] = page_cap_removed
    audit["selection"]["rank_cutoff_removed"] = rank_cutoff_removed

    manifest_rows: list[dict[str, Any]] = []
    for candidate in chosen:
        snapshot = candidate["snapshot"]
        revision_id = snapshot.revision_id
        prompt_hash = str(candidate["prompt_hash"])
        url = make_source_url(revision_id)
        manifest_rows.append(
            {
                "prompt_id": make_prompt_id(revision_id, prompt_hash),
                "source_title": snapshot.title,
                "source_url": url,
                "revision_id": revision_id,
                "retrieved_at": retrieved_at,
                "source_licence": SOURCE_LICENCE,
                "attribution": make_attribution(
                    snapshot.title, revision_id, retrieved_at, url
                ),
                "prompt_text": candidate["prompt_text"],
                "prompt_hash": prompt_hash,
                "topic_category": candidate["topic_category"],
                "content_safety_status": "ok",
                "page_id": snapshot.page_id,
                "words": candidate["words"],
                "script_latin": True,
                "topic_rule": candidate["topic_rule"],
            }
        )

    audit["exclusions"].update(dict(exclusions))
    audit["multi_reason"] = dict(sorted(multi_reason.items()))
    for candidate in chosen:
        audit["by_topic"][candidate["topic_category"]] += 1

    # Family IDs are assigned after selection (§B) and before the manifest is
    # written. Families group near-duplicates and same-page prompts; nothing is
    # removed here.
    families, family_stats = build_families_with_stats(manifest_rows)
    for row in manifest_rows:
        row["family_id"] = families[row["prompt_id"]]
    audit["families"] = family_stats

    if insufficient:
        output_sha = None
        audit["output"] = {"path": str(out_path), "sha256": None, "count": 0}
    else:
        output_sha = write_manifest(out_path, manifest_rows)
        audit["output"] = {
            "path": str(out_path),
            "sha256": output_sha,
            "count": len(manifest_rows),
        }

    audit["reconciliation"]["selected"] = len(manifest_rows)
    _finalize_reconciliation(audit)

    if not audit["reconciliation"]["balanced"] and exit_code == 0:
        audit["warnings"].append("reconciliation failed: input - removed != selected")
        exit_code = 4

    _write_json(audit_path, audit)
    return audit, exit_code


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the E2-C1 prompt manifest (offline dry run supported)."
    )
    parser.add_argument("--input", default=DEFAULT_INPUT, help="evidence JSONL")
    parser.add_argument("--out", default=DEFAULT_OUT, help="prompt manifest output")
    parser.add_argument("--audit", default=DEFAULT_AUDIT, help="audit JSON output")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="run stages 1-5 only; print the stage table; write nothing",
    )
    parser.add_argument(
        "--offline", action="store_true", help="read cached API responses only"
    )
    parser.add_argument(
        "--refresh", action="store_true", help="ignore the API response cache"
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="read at most N input rows (smoke)"
    )
    parser.add_argument(
        "--n", type=int, default=5000, help="number of prompts to select"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)

    if args.dry_run:
        try:
            rows = read_input(Path(args.input), limit=args.limit)
        except FileNotFoundError as exc:
            print(f"input not found: {exc}", file=sys.stderr)
            return 2
        exclusions: collections.Counter[str] = collections.Counter()
        _survivors, stages = run_offline_stages(rows, exclusions)
        print(format_stage_table(stages))
        print(f"offline survivors: {stages[-1]['out']}")
        return 0

    try:
        audit, exit_code = run_build(
            input_path=Path(args.input),
            out_path=Path(args.out),
            audit_path=Path(args.audit),
            n=args.n,
            limit=args.limit,
            offline=args.offline,
            refresh=args.refresh,
        )
    except WikipediaOfflineError as exc:
        print(f"offline cache miss: {exc}", file=sys.stderr)
        return 2

    print(format_stage_table(audit["stages"]))
    print(
        f"status={audit['status']} selected={audit['selection']['selected']} "
        f"balanced={audit['reconciliation']['balanced']}"
    )
    if audit["status"] == "insufficient_pool":
        print(f"shortfall: {audit.get('shortfall', 0)}", file=sys.stderr)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
