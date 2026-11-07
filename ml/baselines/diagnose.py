"""Cross-dataset failure diagnosis.

Goal: explain **why** the baseline collapses out of domain, without tuning anything on
RAID or HC3 labels.

What counts as "not tuning": no hyperparameter search, no feature selection, no threshold
choice may consult RAID/HC3 labels. Those labels are used **only** to report a metric
after the model and threshold are frozen. Everything below obeys that.

Three independent lines of evidence:

1. **Domain shift inside the licensed corpus.** Ghostbuster holds out `news_reuters`
   entirely (`test_ood_domain`). If performance drops there too, the failure is
   domain sensitivity, not generator shift. This uses only training-corpus labels.
2. **Covariate shift, label-free.** Out-of-vocabulary rate of each evaluation corpus
   against the training vocabulary. No labels are read at all.
3. **Decomposition on an external corpus.** For RAID, report AUC by domain and the
   detection rate by generator, after freezing. This is reporting, not tuning.

The distinction matters: if (1) also collapses, then "unseen generator" is not the
explanation and no amount of generator diversity would fix it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from ..ingest.schema import CanonicalRecord, Modality, Split

REPO = Path(__file__).resolve().parents[2]
REPORTS = REPO / "reports"

TARGET_FPR = 0.05


@dataclass
class DomainResult:
    name: str
    n: int
    n_pos: int
    n_neg: int
    roc_auc: float | None
    oov_rate: float
    mean_score: float


@dataclass
class Diagnosis:
    threshold: float
    results: list[DomainResult] = field(default_factory=list)
    raid_by_domain: list[DomainResult] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "threshold": self.threshold,
            "results": [r.__dict__ for r in self.results],
            "raid_by_domain": [r.__dict__ for r in self.raid_by_domain],
            "notes": self.notes,
        }


def _auc(y: np.ndarray, s: np.ndarray) -> float | None:
    if len(y) == 0 or len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, s))


def diagnose(records: Sequence[CanonicalRecord] | None = None) -> Diagnosis:
    from .tfidf import load_text_records

    recs = list(records) if records is not None else load_text_records()
    text = [r for r in recs if r.modality is Modality.TEXT]

    train = [r for r in text if r.split is Split.TRAIN]
    val = [r for r in text if r.split is Split.VAL]
    if not train or not val:
        raise RuntimeError("need train and val splits; run `python -m ml.cli build`")

    vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2,
                          max_features=200_000, sublinear_tf=True, lowercase=True)
    Xtr = vec.fit_transform([r.text or "" for r in train])
    ytr = np.array([r.label for r in train])
    clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced",
                             solver="liblinear").fit(Xtr, ytr)

    vocab = set(vec.vocabulary_)

    def oov_rate(texts: Sequence[str]) -> float:
        """Fraction of tokens not present in the training vocabulary (label-free)."""
        total = unseen = 0
        for t in texts:
            for tok in (t or "").lower().split():
                total += 1
                if tok not in vocab:
                    unseen += 1
        return unseen / total if total else 0.0

    # threshold frozen on val (same rule as the baseline)
    Xva = vec.transform([r.text or "" for r in val])
    val_scores = clf.predict_proba(Xva)[:, 1]
    yva = np.array([r.label for r in val])
    thr = float(np.quantile(val_scores[yva == 0], 1 - TARGET_FPR))

    diag = Diagnosis(threshold=thr)

    def evaluate(name: str, subset: Sequence[CanonicalRecord]) -> DomainResult:
        texts = [r.text or "" for r in subset]
        y = np.array([r.label for r in subset])
        s = clf.predict_proba(vec.transform(texts))[:, 1]
        return DomainResult(
            name=name, n=len(subset), n_pos=int((y == 1).sum()),
            n_neg=int((y == 0).sum()), roc_auc=_auc(y, s),
            oov_rate=oov_rate(texts), mean_score=float(s.mean()) if len(s) else 0.0,
        )

    # 1. inside the licensed corpus
    diag.results.append(evaluate("ghostbuster test_id (in-domain)",
                                 [r for r in text if r.split is Split.TEST_ID]))
    diag.results.append(evaluate("ghostbuster test_ood_domain (news_reuters)",
                                 [r for r in text if r.split is Split.TEST_OOD_DOMAIN]))
    diag.results.append(evaluate("ghostbuster val", [r for r in text if r.split is Split.VAL]))

    # 2/3. external, reporting only
    for src in ("raid", "hc3"):
        sub = [r for r in text if r.source_dataset == src]
        if sub:
            diag.results.append(evaluate(f"{src} (external)", sub))

    # RAID decomposition by domain (reporting only)
    raid = [r for r in text if r.source_dataset == "raid"]
    for dom in sorted({r.domain or "unknown" for r in raid}):
        sub = [r for r in raid if (r.domain or "unknown") == dom]
        if len(sub) >= 60:
            diag.raid_by_domain.append(evaluate(f"raid:{dom}", sub))

    # 4. generator transfer INSIDE the licensed corpus.
    #    Train on human + gpt*, test on human + claude. Both sides come from Ghostbuster,
    #    so this isolates generator change from corpus change -- and it uses only
    #    training-corpus labels, never RAID/HC3.
    gb = [r for r in text if r.source_dataset == "ghostbuster"]
    train_g = [r for r in gb if r.split is Split.TRAIN
               and str(r.metadata.get("variant", "")) != "claude"]
    test_g = [r for r in gb if r.split in (Split.TEST_ID, Split.QUARANTINE)
              and str(r.metadata.get("variant", "")) in ("human", "claude")]
    if len(train_g) > 100 and len({r.label for r in test_g}) == 2:
        v2 = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2,
                             max_features=200_000, sublinear_tf=True, lowercase=True)
        X2 = v2.fit_transform([r.text or "" for r in train_g])
        y2 = np.array([r.label for r in train_g])
        c2 = LogisticRegression(max_iter=2000, class_weight="balanced",
                                solver="liblinear").fit(X2, y2)
        yt = np.array([r.label for r in test_g])
        st = c2.predict_proba(v2.transform([r.text or "" for r in test_g]))[:, 1]
        auc = _auc(yt, st)
        diag.results.append(DomainResult(
            name="ghostbuster: train gpt* -> test claude (generator shift)",
            n=len(test_g), n_pos=int((yt == 1).sum()), n_neg=int((yt == 0).sum()),
            roc_auc=auc, oov_rate=0.0, mean_score=float(st.mean())))
        if auc is not None:
            diag.notes.append(
                f"Generator transfer inside Ghostbuster: training on gpt* and testing on "
                f"claude gives AUC {auc:.3f}. "
                + ("Generator change alone causes a large drop."
                   if auc < 0.85 else
                   "Generator change alone does not explain the external collapse; the "
                   "difference is corpus/era rather than a single generator."))

    in_dom = next((r for r in diag.results if "test_id" in r.name), None)
    ood = next((r for r in diag.results if "ood_domain" in r.name), None)
    if in_dom and ood and in_dom.roc_auc and ood.roc_auc is not None:
        drop = in_dom.roc_auc - ood.roc_auc
        diag.notes.append(
            f"Within the licensed corpus, moving to an unseen domain drops AUC by "
            f"{drop:.3f} ({in_dom.roc_auc:.3f} -> {ood.roc_auc:.3f}). "
            + ("This is a domain-sensitivity result independent of generator shift."
               if drop > 0.15 else
               "Domain shift alone does not explain the external collapse."))

    return diag


def render_markdown(d: Diagnosis) -> str:
    lines = ["# Cross-dataset failure diagnosis", "",
             f"Threshold frozen on `val` at a {TARGET_FPR:.0%} human FPR budget: "
             f"{d.threshold:.4f}. No hyperparameter, feature or threshold decision used "
             "RAID or HC3 labels.", "",
             "| Evaluation set | n | pos | neg | ROC-AUC | OOV rate (label-free) | mean score |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for r in d.results:
        auc = "n/a" if r.roc_auc is None else f"{r.roc_auc:.3f}"
        lines.append(f"| {r.name} | {r.n} | {r.n_pos} | {r.n_neg} | {auc} | "
                     f"{r.oov_rate:.1%} | {r.mean_score:.3f} |")

    if d.raid_by_domain:
        lines += ["", "## RAID by domain (reporting only)", "",
                  "| Domain | n | pos | neg | ROC-AUC | OOV rate | mean score |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for r in d.raid_by_domain:
            auc = "n/a" if r.roc_auc is None else f"{r.roc_auc:.3f}"
            lines.append(f"| {r.name} | {r.n} | {r.n_pos} | {r.n_neg} | {auc} | "
                         f"{r.oov_rate:.1%} | {r.mean_score:.3f} |")

    if d.notes:
        lines += ["", "## Interpretation", ""] + [f"- {n}" for n in d.notes]

    lines += ["", "## What this rules in and out", "",
              "- **Domain sensitivity** is measured inside the licensed corpus, so it cannot "
              "be blamed on RAID.",
              "- **Covariate shift** is estimated without labels: a high out-of-vocabulary rate "
              "means the evaluation text uses vocabulary the model never saw.",
              "- Neither line of evidence licenses a claim of generalisation. They exist to "
              "explain a failure, not to tune it away.", ""]
    return "\n".join(lines)


def write_report(d: Diagnosis, outdir: Path = REPORTS) -> tuple[Path, Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    md = outdir / "diagnose_cross_dataset.md"
    js = outdir / "diagnose_cross_dataset.json"
    md.write_text(render_markdown(d), encoding="utf-8")
    js.write_text(json.dumps(d.to_dict(), indent=2), encoding="utf-8")
    return md, js
