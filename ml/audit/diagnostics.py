"""Source-, variant- and dataset-level diagnostics.

These are **not gates**. They answer questions the whole-corpus probes cannot:

* **Per-source** — is one dataset carrying the task, or hiding a weak one?
* **Per-variant** — within Ghostbuster, does each generator variant behave the same way?
  If one variant is trivially separable and others are not, the model is learning that
  variant, not AI authorship.
* **Dataset fingerprint** — can a model tell *which dataset* a record came from? If two
  composed sources are perfectly distinguishable from the text alone, then a detector
  trained across them can win by recognising the corpus rather than the authorship. This
  is the "dataset-specific shortcut" risk.

Design constraints:

* Nothing here trains or tunes the real model.
* Datasets that the project treats as evaluation-only (RAID, HC3) get an *internal*
  diagnostic holdout derived purely from their own group ids. That holdout is used only to
  characterise the corpus and must never be used for model selection — it is labelled
  ``internal_diagnostic`` in every result so it cannot be mistaken for a project split.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from ..ingest.schema import CanonicalRecord, Modality, Split

MIN_ROWS = 120          # per class, for a diagnostic to be reported at all
MIN_ROWS_PER_CLASS = 40
MAX_DOCS = 4000         # cap for the fingerprint probe (runtime)
CHAR_MAX_FEATURES = 20000


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _char_auc(train_texts, train_labels, test_texts, test_labels) -> float | None:
    y = np.asarray(test_labels)
    if len(y) == 0 or len(np.unique(y)) < 2:
        return None
    if min((y == 1).sum(), (y == 0).sum()) < MIN_ROWS_PER_CLASS:
        return None
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 3),
                          max_features=CHAR_MAX_FEATURES, sublinear_tf=True)
    Xtr = vec.fit_transform(train_texts)
    Xte = vec.transform(test_texts)
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(Xtr, train_labels)
    return float(roc_auc_score(y, clf.predict_proba(Xte)[:, 1]))


def _length_auc(test_word_counts, test_labels) -> float | None:
    y = np.asarray(test_labels)
    if len(y) == 0 or len(np.unique(y)) < 2:
        return None
    if min((y == 1).sum(), (y == 0).sum()) < MIN_ROWS_PER_CLASS:
        return None
    auc = float(roc_auc_score(y, np.asarray(test_word_counts, dtype=float)))
    return max(auc, 1.0 - auc)


def _internal_holdout(records: Sequence[CanonicalRecord], frac: float = 0.30):
    """Deterministic 70/30 split by group id, for diagnostics on quarantine data only."""
    train, test = [], []
    for r in records:
        b = int(hashlib.sha256(r.group_id.encode()).hexdigest()[:8], 16) % 100
        (test if b < frac * 100 else train).append(r)
    return train, test


@dataclass
class SourceDiagnostic:
    source: str
    n_train: int
    n_test: int
    split_basis: str            # "project_split" | "internal_diagnostic"
    char_auc: float | None
    length_auc: float | None
    note: str = ""


@dataclass
class VariantDiagnostic:
    variant: str
    n_train_machine: int
    n_test_machine: int
    n_test_human: int
    char_auc: float | None


@dataclass
class FingerprintResult:
    pair: str
    n_a: int
    n_b: int
    auc: float | None


@dataclass
class Diagnostics:
    sources: list[SourceDiagnostic] = field(default_factory=list)
    variants: list[VariantDiagnostic] = field(default_factory=list)
    fingerprints: list[FingerprintResult] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "sources": [asdict(s) for s in self.sources],
            "variants": [asdict(v) for v in self.variants],
            "dataset_fingerprints": [asdict(f) for f in self.fingerprints],
            "notes": self.notes,
        }


def run_diagnostics(records: Sequence[CanonicalRecord]) -> Diagnostics:
    out = Diagnostics()
    text = [r for r in records
            if r.modality is Modality.TEXT and r.split is not Split.EXCLUDED]

    # ---------------- per source ----------------
    by_source: dict[str, list[CanonicalRecord]] = {}
    for r in text:
        by_source.setdefault(r.source_dataset, []).append(r)

    for source, recs in sorted(by_source.items()):
        labels = {r.label for r in recs}
        if len(labels) < 2:
            out.sources.append(SourceDiagnostic(source, 0, len(recs), "n/a", None, None,
                                                "single-class source"))
            continue

        tr = [r for r in recs if r.split is Split.TRAIN]
        te = [r for r in recs if r.split is Split.TEST_ID]
        basis = "project_split"
        if len(tr) < MIN_ROWS or len(te) < MIN_ROWS:
            tr, te = _internal_holdout(recs)
            basis = "internal_diagnostic"

        if len({r.label for r in tr}) < 2 or len({r.label for r in te}) < 2:
            out.sources.append(SourceDiagnostic(
                source, len(tr), len(te), basis, None, None, "single class in a side"))
            continue

        auc = _char_auc([r.text or "" for r in tr], [r.label for r in tr],
                        [r.text or "" for r in te], [r.label for r in te])
        lauc = _length_auc([int(r.metadata.get("word_count", 0)) for r in te],
                           [r.label for r in te])
        out.sources.append(SourceDiagnostic(source, len(tr), len(te), basis, auc, lauc))

    # ---------------- per variant (Ghostbuster) ----------------
    gb = [r for r in text if r.source_dataset == "ghostbuster"]
    if gb:
        humans_tr = [r for r in gb if r.label == 0 and r.split is Split.TRAIN]
        humans_te = [r for r in gb if r.label == 0 and r.split is Split.TEST_ID]
        variants: dict[str, list[CanonicalRecord]] = {}
        for r in gb:
            if r.label == 1:
                v = str(r.metadata.get("variant", "unknown"))
                variants.setdefault(v, []).append(r)
        for v, recs in sorted(variants.items()):
            m_tr = [r for r in recs if r.split is Split.TRAIN]
            m_te = [r for r in recs if r.split is Split.TEST_ID]
            auc = None
            if m_tr and m_te and humans_tr and humans_te:
                auc = _char_auc(
                    [r.text or "" for r in humans_tr] + [r.text or "" for r in m_tr],
                    [0] * len(humans_tr) + [1] * len(m_tr),
                    [r.text or "" for r in humans_te] + [r.text or "" for r in m_te],
                    [0] * len(humans_te) + [1] * len(m_te),
                )
            out.variants.append(VariantDiagnostic(v, len(m_tr), len(m_te),
                                                  len(humans_te), auc))

    # ---------------- dataset fingerprints ----------------
    names = sorted(by_source)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a_name, b_name = names[i], names[j]
            a = by_source[a_name][:MAX_DOCS]
            b = by_source[b_name][:MAX_DOCS]
            if len(a) < MIN_ROWS or len(b) < MIN_ROWS:
                continue
            a_tr, a_te = _internal_holdout(a)
            b_tr, b_te = _internal_holdout(b)
            if not a_tr or not b_tr or not a_te or not b_te:
                continue
            auc = _char_auc(
                [r.text or "" for r in a_tr] + [r.text or "" for r in b_tr],
                [0] * len(a_tr) + [1] * len(b_tr),
                [r.text or "" for r in a_te] + [r.text or "" for r in b_te],
                [0] * len(a_te) + [1] * len(b_te),
            )
            out.fingerprints.append(FingerprintResult(
                f"{a_name} vs {b_name}", len(a), len(b), auc))

    if any(f.auc is not None and f.auc >= 0.95 for f in out.fingerprints):
        out.notes.append(
            "At least one dataset pair is separated at AUC >= 0.95 from text alone. "
            "A model trained across these sources can identify the corpus rather than "
            "the authorship. Report metrics per source and never present a pooled "
            "number as if the sources were exchangeable.")
    if any(v.char_auc is not None and v.char_auc >= 0.99 for v in out.variants):
        out.notes.append(
            "At least one Ghostbuster variant is separable from human text at "
            "AUC >= 0.99. Per-variant results must be reported so a strong pooled "
            "number cannot hide a variant-specific shortcut.")

    return out


def render_markdown(diag: Diagnostics) -> str:
    lines = ["## Diagnostics (non-gating)", ""]

    if diag.sources:
        lines += ["### Per source", "",
                  "| Source | Basis | n train | n test | char AUC | length AUC | Note |",
                  "|---|---|---:|---:|---:|---:|---|"]
        for s in diag.sources:
            ca = "n/a" if s.char_auc is None else f"{s.char_auc:.3f}"
            la = "n/a" if s.length_auc is None else f"{s.length_auc:.3f}"
            lines.append(f"| {s.source} | {s.split_basis} | {s.n_train} | {s.n_test} | "
                         f"{ca} | {la} | {s.note} |")

    if diag.variants:
        lines += ["", "### Per Ghostbuster variant (human vs variant)", "",
                  "| Variant | n train (machine) | n test (machine) | n test (human) | char AUC |",
                  "|---|---:|---:|---:|---:|"]
        for v in diag.variants:
            ca = "n/a" if v.char_auc is None else f"{v.char_auc:.3f}"
            lines.append(f"| {v.variant} | {v.n_train_machine} | {v.n_test_machine} | "
                         f"{v.n_test_human} | {ca} |")

    if diag.fingerprints:
        lines += ["", "### Dataset fingerprints (can text reveal the source?)", "",
                  "| Pair | n A | n B | AUC |", "|---|---:|---:|---:|"]
        for f in diag.fingerprints:
            ca = "n/a" if f.auc is None else f"{f.auc:.3f}"
            lines.append(f"| {f.pair} | {f.n_a} | {f.n_b} | {ca} |")

    if diag.notes:
        lines += ["", "**Diagnostic notes**", ""] + [f"- {n}" for n in diag.notes]

    lines += ["", "_These diagnostics are reported, not enforced. They exist so a pooled "
                  "score cannot hide which source or variant produced it._", ""]
    return "\n".join(lines)
