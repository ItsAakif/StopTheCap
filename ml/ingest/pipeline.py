"""Build pipeline: manifests -> canonical records -> splits -> audit.

Run order is fixed and each step is auditable:

    manifests -> read -> normalise+flags -> dedup -> group -> split -> write -> audit

Nothing here trains a model. The pipeline's only job is to produce a dataset whose
provenance and split integrity can be defended.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import yaml

from ..audit.audit import AuditResult, run_audit, write_reports
from ..data.dedup import DuplicatePair, find_duplicates
from ..data.splits import (
    DEFAULT_SALT,
    apply_plan,
    build_split_plan,
    group_counts,
    split_counts,
    split_digest,
)
from ..ingest.readers.base import DatasetManifest, RawSample, build_record, quality_flags
from ..ingest.readers.ghostbuster import GhostbusterReader
from ..ingest.readers.hc3 import HC3Reader
from ..ingest.readers.raid import RaidReader
from ..ingest.readers.video import GenVidBenchInventoryReader
from ..ingest.schema import CanonicalRecord, Modality, Split

REPO = Path(__file__).resolve().parents[2]
RAW = REPO / "datasets" / "raw"
MANIFESTS = REPO / "datasets" / "manifests"
CANONICAL = REPO / "datasets" / "canonical"
SPLITS = REPO / "datasets" / "splits"
REPORTS = REPO / "reports"


@dataclass
class BuildConfig:
    salt: str = DEFAULT_SALT
    raid_n: int = 2000
    raid_seed: int = 20261029
    generator_holdout: str | None = None
    domain_holdout: str | None = "news_reuters"
    write: bool = True
    #: Per-dataset record caps. Milestone 1 proves pipeline correctness on a
    #: representative sample; full corpora are a later, explicitly gated step.
    #: Every cap is recorded in COMBINED.yaml so no result can silently depend on it.
    max_per_dataset: dict[str, int] = field(default_factory=lambda: {
        "hc3": 8000,          # 85k records available; eval-only, sample is sufficient
        "genvidbench": 20000,  # 142k inventory rows; sample proves split behaviour
    })


@dataclass
class BuildResult:
    records: list[CanonicalRecord]
    duplicates: list[DuplicatePair]
    audit: AuditResult | None = None
    per_dataset: dict[str, int] = field(default_factory=dict)
    excluded: dict[str, int] = field(default_factory=dict)


def load_manifests(directory: Path = MANIFESTS) -> dict[str, DatasetManifest]:
    out: dict[str, DatasetManifest] = {}
    for p in sorted(directory.glob("*.yaml")):
        if p.name == "COMBINED.yaml":
            continue
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        if not data.get("enabled", True):
            continue
        if "dataset_id" not in data:
            continue
        out[data["dataset_id"]] = DatasetManifest.load(p)
    return out


def _read_dataset(ds_id: str, cfg: BuildConfig) -> Iterable[RawSample]:
    if ds_id == "ghostbuster":
        return GhostbusterReader().read(RAW / "ghostbuster")
    if ds_id == "hc3":
        return HC3Reader().read(RAW / "hc3")
    if ds_id == "raid":
        return RaidReader(cache=RAW / "raid" / "sample_train.jsonl",
                          n=cfg.raid_n, seed=cfg.raid_seed).read(RAW / "raid")
    if ds_id == "genvidbench":
        return GenVidBenchInventoryReader().read(RAW / "genvidbench")
    return []


def build(cfg: BuildConfig | None = None) -> BuildResult:
    import sys
    import time

    t_start = time.time()

    def _log(msg: str) -> None:
        print(f"[build {time.time() - t_start:6.1f}s] {msg}", file=sys.stderr, flush=True)

    cfg = cfg or BuildConfig()
    manifests = load_manifests()
    _log(f"manifests: {sorted(manifests)}")

    records: list[CanonicalRecord] = []
    per_dataset: Counter = Counter()
    excluded: Counter = Counter()
    capped: dict[str, dict] = {}
    total_available: Counter = Counter()

    for ds_id, man in manifests.items():
        raws = list(_read_dataset(ds_id, cfg))
        if not raws:
            continue
        total_available[ds_id] = len(raws)
        cap = cfg.max_per_dataset.get(ds_id)
        if cap and len(raws) > cap:
            # Deterministic stride sample, not a head slice: source files are
            # ordered by variant/author, so a head slice would take only one.
            step = len(raws) / float(cap)
            raws = [raws[int(i * step)] for i in range(cap)]
            capped[ds_id] = {"available": total_available[ds_id], "used": cap}
        for raw in raws:
            rec = build_record(raw, man, split=Split.EXCLUDED)
            flags = quality_flags(rec)
            if man.unresolved:
                flags.append("licence_unresolved")
            meta = dict(rec.metadata)
            meta["quality_flags"] = flags
            meta["_eval_only"] = not man.permitted_train
            if "too_short" in flags:
                meta["exclusion_reason"] = "too_short"
            elif "empty_text" in flags:
                meta["exclusion_reason"] = "empty_text"
            # NOTE: `licence_unresolved` is recorded as a flag but does NOT exclude the
            # record. For genvidbench the unresolved point concerns redistribution of
            # the *media*, not whether we may hold an inventory row; training is already
            # blocked by permitted_uses.train=false, so the row is safe in QUARANTINE.
            rec = rec.model_copy(update={"metadata": meta})
            records.append(rec)
            per_dataset[ds_id] += 1
            for f in flags:
                excluded[f] += 1

    # ---------- dedup (text only) ----------
    _log(f"read {len(records)} records; starting dedup")
    # Only records that are still candidates participate: already-excluded records
    # (too short / empty / unresolved licence) would otherwise produce meaningless
    # "duplicate" pairs among empty strings and inflate the report.
    text_idx = [
        i for i, r in enumerate(records)
        if r.modality is Modality.TEXT
        and not r.metadata.get("exclusion_reason")
        and int(r.metadata.get("word_count", 0)) >= 40
    ]
    texts = [records[i].text or "" for i in text_idx]
    hashes = [records[i].content_hash for i in text_idx]
    groups = [records[i].group_id for i in text_idx]

    pairs = find_duplicates(texts, hashes, groups) if texts else []

    # union duplicate pairs into one group so they cannot straddle splits
    group_rewrite: dict[str, str] = {}
    drop_reason: dict[int, str] = {}
    for p in pairs:
        a, b = text_idx[p.i], text_idx[p.j]
        if a > b:
            a, b = b, a
        ga, gb = records[a].group_id, records[b].group_id
        keep_g = min(ga, gb)
        for g in (ga, gb):
            if g != keep_g:
                group_rewrite[g] = keep_g
        if b not in drop_reason:
            drop_reason[b] = f"duplicate_of:{records[a].sample_id}"

    if group_rewrite:
        merged: list[CanonicalRecord] = []
        for r in records:
            g = group_rewrite.get(r.group_id)
            merged.append(r.model_copy(update={"group_id": g}) if g else r)
        records = merged

    for idx, reason in drop_reason.items():
        meta = dict(records[idx].metadata)
        meta["exclusion_reason"] = reason
        records[idx] = records[idx].model_copy(
            update={"metadata": meta, "split": Split.EXCLUDED})

    # ---------- split ----------
    train_allowed = [ds for ds, m in manifests.items() if m.permitted_train]
    plan = build_split_plan(
        records,
        train_allowed_datasets=train_allowed,
        salt=cfg.salt,
        generator_holdout=cfg.generator_holdout,
        domain_holdout=cfg.domain_holdout,
    )
    assigned = apply_plan(records, plan)

    # force eval-only to quarantine (defence in depth)
    final: list[CanonicalRecord] = []
    for r in assigned:
        if r.metadata.get("_eval_only") and r.split.is_trainable:
            r = r.model_copy(update={"split": Split.QUARANTINE})
        final.append(r)

    result = BuildResult(records=final,
                         duplicates=[DuplicatePair(text_idx[p.i], text_idx[p.j],
                                                   p.jaccard, p.kind) for p in pairs],
                         per_dataset=dict(per_dataset), excluded=dict(excluded))

    if cfg.write:
        _write_outputs(final, plan, manifests, cfg, result, capped, dict(total_available))
    return result


def audit_build(cfg: BuildConfig | None = None) -> tuple[BuildResult, AuditResult]:
    cfg = cfg or BuildConfig()
    res = build(cfg)
    expected = None
    combined = MANIFESTS / "COMBINED.yaml"
    if combined.is_file():
        expected = (yaml.safe_load(combined.read_text(encoding="utf-8")) or {}).get("split_digest")
    audit = run_audit(res.records, duplicates=res.duplicates,
                      salt=cfg.salt, expected_split_digest=expected)
    res.audit = audit
    if cfg.write:
        write_reports(audit, REPORTS)
        _write_ingest_report(res)
    return res, audit


# ------------------------------------------------------------------ outputs

def _write_outputs(records, plan, manifests, cfg, result, capped=None, available=None) -> None:
    CANONICAL.mkdir(parents=True, exist_ok=True)
    SPLITS.mkdir(parents=True, exist_ok=True)

    by_ds: dict[str, list[CanonicalRecord]] = defaultdict(list)
    for r in records:
        by_ds[r.source_dataset].append(r)
    for ds, recs in by_ds.items():
        with (CANONICAL / f"{ds}.jsonl").open("w", encoding="utf-8") as fh:
            for r in recs:
                fh.write(r.model_dump_json() + "\n")

    with (SPLITS / "assignments.jsonl").open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps({"sample_id": r.sample_id, "group_id": r.group_id,
                                 "dataset": r.source_dataset,
                                 "split": r.split.value}) + "\n")

    counts = split_counts(records)
    groups = group_counts(records)
    doc = {
        "combined_version": f"milestone1-{cfg.salt}",
        "split_salt": cfg.salt,
        "split_digest": split_digest(records, cfg.salt),
        "generator_holdout": cfg.generator_holdout,
        "domain_holdout": cfg.domain_holdout,
        "holdout_note": (
            "No generator is held out from the trainable corpus. Holding out a "
            "generator is incompatible with pair-grouping: because every machine "
            "variant of a prompt shares one group with its human source, excluding "
            "one generator removes every group it appears in. Unseen-generator "
            "evaluation is provided instead by RAID, which is never trained on."
        ),
        "sources": {
            ds: {"version": m.version, "licence": m.licence_spdx,
                 "train_permitted": m.permitted_train,
                 "records": result.per_dataset.get(ds, 0)}
            for ds, m in manifests.items()
        },
        "counts": {"records": counts, "groups": groups},
        "per_dataset": result.per_dataset,
        "records_available_before_caps": available or {},
        "sample_caps_applied": capped or {},
        "quality_flags": result.excluded,
        "duplicate_pairs": len(result.duplicates),
        "reproduction": ["python -m ml.cli build", "python -m ml.cli audit"],
    }
    (MANIFESTS / "COMBINED.yaml").write_text(
        yaml.safe_dump(doc, sort_keys=False, allow_unicode=True), encoding="utf-8")


def _write_ingest_report(result: BuildResult) -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    lines = ["# Ingest summary", "", "| Dataset | Records |", "|---|---:|"]
    for ds, n in sorted(result.per_dataset.items()):
        lines.append(f"| {ds} | {n} |")
    lines += ["", "## Quality flags", "", "| Flag | Count |", "|---|---:|"]
    for f, n in sorted(result.excluded.items()):
        lines.append(f"| {f} | {n} |")
    lines += ["", f"**Duplicate pairs:** {len(result.duplicates)}", ""]
    (REPORTS / "ingest.md").write_text("\n".join(lines), encoding="utf-8")
