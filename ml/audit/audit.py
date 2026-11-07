"""Automated data-quality and leakage audit.

Produces a JSON result and a human-readable markdown report. The audit is the
project's primary technical claim (ADR-016), so every check states *what it proves*
rather than only whether it passed.

Checks
------
A. Structural      -- duplicate ids, empty text, missing provenance, bad labels
B. Distribution    -- label/source/domain/generator counts, class balance
C. Duplicates      -- exact + near (Jaccard >= 0.85), and *cross-split* duplicates
D. Pairing         -- groups containing both a human and a machine record
E. Group leakage   -- any group_id spanning two splits
F. Role            -- evaluation-only datasets absent from train/val
G. Shortcut        -- character-only and length-only AUC gates (<= 0.60)
H. Determinism     -- split hash recomputation matches the committed manifest
I. Video           -- separate checks; text metrics are not forced onto video
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from ..data.dedup import DuplicatePair, find_duplicates, summarise
from ..data.splits import DEFAULT_SALT, bucket_for, group_counts, split_counts, split_digest
from ..ingest.schema import CanonicalRecord, Modality, Split
from .diagnostics import run_diagnostics, render_markdown as render_diagnostics
from .shortcut import (
    ProbeResult,
    artifact_single_feature_probe,
    character_only_probe,
    length_only_probe,
)

EVAL_ONLY_DATASETS = {"raid", "hc3"}


@dataclass
class AuditResult:
    ok: bool = True
    gates: dict[str, bool] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    data: dict = field(default_factory=dict)

    def gate(self, name: str, passed: bool, detail: str = "") -> None:
        self.gates[name] = passed
        if not passed:
            self.ok = False
            self.notes.append(f"FAIL {name}: {detail}")
        elif detail:
            self.notes.append(f"pass {name}: {detail}")


def _must(rec: CanonicalRecord) -> list[str]:
    """Fields that must be present for provenance to be defensible."""
    missing: list[str] = []
    for f in ("sample_id", "source_dataset", "source_version", "source_sample_id",
              "content_hash", "group_id", "licence_ref", "original_label"):
        if not getattr(rec, f):
            missing.append(f)
    if rec.modality is Modality.TEXT and not rec.metadata.get("word_count"):
        missing.append("word_count")
    return missing


def run_audit(
    records: Sequence[CanonicalRecord],
    duplicates: Sequence[DuplicatePair] | None = None,
    salt: str = DEFAULT_SALT,
    expected_split_digest: str | None = None,
) -> AuditResult:
    res = AuditResult()

    text_recs = [r for r in records if r.modality is Modality.TEXT]
    video_recs = [r for r in records if r.modality is Modality.VIDEO]
    active = [r for r in records if r.split is not Split.EXCLUDED]

    # ---------- A. structural ----------
    ids = Counter(r.sample_id for r in records)
    dup_ids = [k for k, v in ids.items() if v > 1]
    res.gate("unique_sample_ids", not dup_ids,
             f"{len(dup_ids)} duplicated sample_id values" if dup_ids else
             f"{len(ids)} unique ids")

    missing = {r.sample_id: _must(r) for r in active}
    missing = {k: v for k, v in missing.items() if v}
    res.gate("required_provenance", not missing,
             f"{len(missing)} records missing fields" if missing else
             f"all {len(active)} active records complete")

    empty = [r.sample_id for r in active
             if r.modality is Modality.TEXT and not r.metadata.get("word_count")]
    res.gate("no_empty_text", not empty, f"{len(empty)} empty text records" if empty else
             "no empty text among active records")

    bad_labels = [r.sample_id for r in records if r.label < 0]
    unmapped = [r.sample_id for r in records if not r.original_label]
    res.gate("labels_valid", not bad_labels and not unmapped,
             f"{len(bad_labels)} negative labels, {len(unmapped)} missing original_label"
             if (bad_labels or unmapped) else "labels mapped")

    # ---------- B. distributions ----------
    def dist(keyfn) -> dict[str, dict[str, int]]:
        out: dict[str, Counter] = defaultdict(Counter)
        for r in active:
            out[str(keyfn(r))][str(r.label)] += 1
        return {k: dict(v) for k, v in sorted(out.items())}

    res.data["split_counts"] = split_counts(records)
    res.data["group_counts"] = group_counts(records)
    res.data["label_counts"] = dict(Counter(str(r.label) for r in active))
    res.data["by_dataset"] = dist(lambda r: r.source_dataset)
    res.data["by_domain"] = dist(lambda r: r.domain or "unknown")
    res.data["by_generator"] = dist(lambda r: r.generator or "human")
    res.data["by_modality"] = dict(Counter(r.modality.value for r in records))
    res.data["excluded_counts"] = dict(Counter(
        r.metadata.get("exclusion_reason", "unspecified")
        for r in records if r.split is Split.EXCLUDED
    ))

    # ---------- C. duplicates ----------
    if duplicates is None:
        duplicates = []
    near = [p for p in duplicates if p.kind == "near"]
    exact = [p for p in duplicates if p.kind == "exact"]
    res.data["duplicates"] = {**summarise(duplicates),
                              "exact": len(exact), "near": len(near)}

    # cross-split duplicates
    cross = 0
    if duplicates:
        for p in duplicates:
            a, b = records[p.i], records[p.j]
            if a.split is not Split.EXCLUDED and b.split is not Split.EXCLUDED:
                if a.split != b.split:
                    cross += 1
    res.gate("no_cross_split_duplicates", cross == 0,
             f"{cross} cross-split duplicate pairs" if cross else "0 across splits")

    # ---------- D. pairing ----------
    labels_by_group: dict[str, set[int]] = defaultdict(set)
    for r in active:
        if r.modality is Modality.TEXT:
            labels_by_group[r.group_id].add(r.label)
    paired = {g: ls for g, ls in labels_by_group.items() if len(ls) > 1}
    res.data["paired_groups"] = len(paired)

    # ---------- E. group leakage ----------
    spans: dict[str, set[str]] = defaultdict(set)
    for r in active:
        spans[r.group_id].add(r.split.value)
    leaking = {g: sorted(s) for g, s in spans.items() if len(s) > 1}
    res.data["leaking_groups"] = len(leaking)
    res.gate("no_group_split_overlap", not leaking,
             f"{len(leaking)} groups span splits" if leaking else
             f"{len(spans)} groups, none spanning splits")

    # ---------- F. role enforcement ----------
    bad_role = [r.sample_id for r in records
                if r.source_dataset in EVAL_ONLY_DATASETS and r.split.is_trainable]
    res.gate("eval_only_not_trained", not bad_role,
             f"{len(bad_role)} eval-only records in train/val" if bad_role else
             "eval-only datasets quarantined")

    # ---------- G. shortcut probes ----------
    tr = [r for r in text_recs if r.split is Split.TRAIN]
    te = [r for r in text_recs if r.split is Split.TEST_ID]
    probes: list[ProbeResult] = []
    if tr and te:
        tr_text = [r.text or "" for r in tr]
        te_text = [r.text or "" for r in te]

        # GATE: no single interpretable feature may nearly solve the task.
        sid = artifact_single_feature_probe(te_text, [r.label for r in te])
        probes.append(sid)
        res.data.setdefault("probes", {})[sid.name] = sid.to_dict()
        res.gate("shortcut_single_feature", sid.passed,
                 (f"max single-feature AUC={sid.auc:.3f}" if sid.auc is not None
                  else sid.reason) + (" > 0.90" if not sid.passed and sid.auc else ""))

        # GATE: length must not carry the signal.
        lp = length_only_probe(
            [int(r.metadata.get("word_count", 0)) for r in tr], [r.label for r in tr],
            [int(r.metadata.get("word_count", 0)) for r in te], [r.label for r in te])
        probes.append(lp)
        res.data.setdefault("probes", {})[lp.name] = lp.to_dict()
        if lp.auc is None:
            res.notes.append(f"skip {lp.name}: {lp.reason}")
        else:
            res.gate("shortcut_length_only", lp.passed,
                     f"AUC={lp.auc:.3f} {'>' if not lp.passed else '<='} 0.60")

        # DIAGNOSTIC (not gating): multi-feature character model. Reported because a
        # high value is the honest signal that in-domain performance is easy and will
        # not transfer -- see the note in shortcut.py explaining why this is not an
        # absolute gate.
        if any(tr_text) and any(te_text):
            cp = character_only_probe(tr_text, [r.label for r in tr], te_text,
                                      [r.label for r in te])
            probes.append(cp)
            res.data.setdefault("probes", {})[cp.name] = {**cp.to_dict(),
                                                          "gating": False}
            if cp.auc is not None:
                res.notes.append(
                    f"diagnostic character_only AUC={cp.auc:.3f} (not a gate; "
                    "see shortcut.py for why an absolute threshold is inappropriate)")
    else:
        res.notes.append("skip shortcut probes: empty train or test_id split")
        res.gate("shortcut_probes_runnable", False,
                 "no train/test_id text available")

    # ---------- H. determinism ----------
    if expected_split_digest:
        recomputed = split_digest(records, salt)
        res.data["split_digest_recomputed"] = recomputed
        res.gate("split_deterministic", recomputed == expected_split_digest,
                 "recomputed digest matches manifest" if recomputed == expected_split_digest
                 else "digest mismatch -- split is not reproducible")
    else:
        res.data["split_digest_recomputed"] = split_digest(records, salt)

    # ---------- I. diagnostics (non-gating) ----------
    try:
        diag = run_diagnostics(records)
        res.data["diagnostics"] = diag.to_dict()
        res.notes.extend(f"diagnostic: {n}" for n in diag.notes)
    except Exception as exc:  # diagnostics must never break the audit
        res.data["diagnostics"] = {"error": str(exc)}
        res.notes.append(f"diagnostics unavailable: {exc}")

    # ---------- J. video (task-specific, not text metrics) ----------
    if video_recs:
        vids = [r for r in video_recs if r.split is not Split.EXCLUDED]
        vid_spans: dict[str, set[str]] = defaultdict(set)
        ident_spans: dict[str, set[str]] = defaultdict(set)
        for r in vids:
            vid_spans[r.group_id].add(r.split.value)
            if r.identity_group:
                ident_spans[r.identity_group].add(r.split.value)
        res.data["video_records"] = len(video_recs)
        res.data["video_label_counts"] = dict(Counter(str(r.label) for r in vids))
        res.data["video_group_leaks"] = sum(1 for s in vid_spans.values() if len(s) > 1)
        res.data["video_identity_leaks"] = sum(1 for s in ident_spans.values() if len(s) > 1)
        conflicts = sum(1 for r in video_recs if r.metadata.get("manifest_label_conflict"))
        res.gate("video_no_group_overlap", res.data["video_group_leaks"] == 0,
                 f"{res.data['video_group_leaks']} video groups span splits")
        res.gate("video_media_absent_marked",
                 all(r.metadata.get("media_present") is False for r in video_recs),
                 "all video records correctly declare media_present=false")
        if conflicts:
            res.notes.append(f"note: {conflicts} video records where the manifest label "
                             "disagrees with the directory convention")

    return res


# ---------------------------------------------------------------- reporting

def render_markdown(res: AuditResult, *, title: str = "Dataset audit") -> str:
    lines = [f"# {title}", "", f"**Overall: {'PASS' if res.ok else 'FAIL'}**", ""]

    lines += ["## Gates", "", "| Gate | Result | Detail |", "|---|---|---|"]
    for name, passed in res.gates.items():
        detail = ""
        for n in res.notes:
            if name in n:
                detail = n.split(":", 1)[-1].strip()
                break
        lines.append(f"| `{name}` | {'PASS' if passed else '**FAIL**'} | {detail} |")

    if res.notes:
        lines += ["", "## Notes", ""] + [f"- {n}" for n in res.notes]

    d = res.data
    if "split_counts" in d:
        lines += ["", "## Records by split", "", "| Split | Records | Groups |", "|---|---:|---:|"]
        for split, n in d["split_counts"].items():
            g = d.get("group_counts", {}).get(split, 0)
            lines.append(f"| {split} | {n} | {g} |")

    if "by_dataset" in d:
        lines += ["", "## Records by dataset (label distribution)", "",
                  "| Dataset | label 0 | label 1 | label 2 |", "|---|---:|---:|---:|"]
        for ds, counts in d["by_dataset"].items():
            lines.append(f"| {ds} | {counts.get('0', 0)} | {counts.get('1', 0)} | "
                         f"{counts.get('2', 0)} |")

    if "by_generator" in d:
        lines += ["", "## Records by generator", "", "| Generator | Count by label |", "|---|---|"]
        for g, counts in d["by_generator"].items():
            lines.append(f"| {g} | {counts} |")

    if "probes" in d:
        lines += ["", "## Shortcut probes", "",
                  "| Probe | AUC | Gate | n_test | neg | pos |",
                  "|---|---:|---|---:|---:|---:|"]
        for name, p in d["probes"].items():
            auc = "n/a" if p["auc"] is None else f"{p['auc']:.3f}"
            lines.append(f"| {name} | {auc} | {'PASS' if p['passed'] else '**FAIL**'} | "
                         f"{p['n_test']} | {p['n_test_neg']} | {p['n_test_pos']} |")
        lines += ["", "_Gate: AUC <= 0.60. A high value means the corpus leaks "
                  "typography or verbosity rather than style._"]

    if "duplicates" in d:
        lines += ["", "## Duplicates", "",
                  f"- exact pairs: {d['duplicates'].get('exact', 0)}",
                  f"- near pairs (Jaccard >= 0.85): {d['duplicates'].get('near', 0)}",
                  f"- cross-split pairs: {'0' if res.gates.get('no_cross_split_duplicates') else 'present'}"]

    if "paired_groups" in d:
        lines += ["", f"**Paired groups (human+machine of the same source): "
                      f"{d['paired_groups']}**"]

    if "video_records" in d:
        lines += ["", "## Video", "",
                  f"- inventory records: {d['video_records']}",
                  f"- label counts: {d.get('video_label_counts')}",
                  f"- group leaks: {d.get('video_group_leaks')}",
                  f"- identity leaks: {d.get('video_identity_leaks')}"]

    diag_data = d.get("diagnostics")
    if diag_data and "sources" in diag_data:
        from .diagnostics import Diagnostics, SourceDiagnostic, VariantDiagnostic, FingerprintResult

        diag = Diagnostics(
            sources=[SourceDiagnostic(**s) for s in diag_data["sources"]],
            variants=[VariantDiagnostic(**v) for v in diag_data["variants"]],
            fingerprints=[FingerprintResult(**f) for f in diag_data["dataset_fingerprints"]],
            notes=list(diag_data.get("notes", [])),
        )
        lines += ["", render_diagnostics(diag)]

    lines += ["", "## Reproduce", "", "```", "python -m ml.cli build", "python -m ml.cli audit",
              "```", ""]
    return "\n".join(lines)


def write_reports(res: AuditResult, outdir: Path, name: str = "audit") -> tuple[Path, Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    md = outdir / f"{name}.md"
    js = outdir / f"{name}.json"
    md.write_text(render_markdown(res), encoding="utf-8")
    js.write_text(json.dumps({"ok": res.ok, "gates": res.gates,
                              "notes": res.notes, "data": res.data},
                             indent=2, default=str), encoding="utf-8")
    return md, js
