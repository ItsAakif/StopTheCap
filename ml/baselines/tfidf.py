"""TF-IDF + logistic-regression baseline.

Purpose: establish the floor that any encoder must beat, and produce the first honest
numbers for the project.

Rules this script obeys (research/03, research/10):

* **Fit on `train` only.** Nothing else contributes to the fitted vocabulary or weights.
* **Threshold comes from `val` only**, chosen to hit a human false-positive budget.
  The test split is scored once, after the threshold is frozen.
* **RAID and HC3 are evaluation-only.** They are loaded from QUARANTINE and never fitted,
  never used for feature selection, and never used to choose a threshold.
* **Human false-positive rate is the primary reported metric**, with the negative-sample
  count and a Wilson interval. Where the negative count is too small to support a
  threshold metric, the report says so instead of printing a fragile number.

The 5.9:1 machine:human imbalance is handled with ``class_weight='balanced'``, which
reweights the loss so each class contributes equally. The alternative (subsampling the
majority class) discards data; oversampling duplicates human texts and can leak via
near-duplicates, so reweighting is the justified choice here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

from ..ingest.schema import CanonicalRecord, Modality, Split

REPO = Path(__file__).resolve().parents[2]
CANONICAL = REPO / "datasets" / "canonical"
REPORTS = REPO / "reports"

#: Target human false-positive rate used to pick the decision threshold on `val`.
TARGET_FPR = 0.05
#: Below this many negatives a threshold metric is reported as insufficient.
MIN_NEGATIVES_FOR_THRESHOLD_METRIC = 1000

CONFIGS = {
    "word": dict(analyzer="word", ngram_range=(1, 2), min_df=2, max_features=200_000,
                 sublinear_tf=True, lowercase=True),
    "char": dict(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=200_000,
                 sublinear_tf=True, lowercase=True),
}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    n = len(y)
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        m = (p >= lo) & (p < hi if i < bins - 1 else p <= hi)
        if not m.any():
            continue
        ece += abs(p[m].mean() - y[m].mean()) * m.sum() / n
    return float(ece)


def load_text_records(directory: Path = CANONICAL) -> list[CanonicalRecord]:
    out: list[CanonicalRecord] = []
    for p in sorted(directory.glob("*.jsonl")):
        with p.open(encoding="utf-8") as fh:
            for line in fh:
                r = CanonicalRecord.model_validate_json(line)
                if r.modality is Modality.TEXT and r.split is not Split.EXCLUDED:
                    out.append(r)
    return out


def _xy(recs: Sequence[CanonicalRecord]) -> tuple[list[str], np.ndarray]:
    return [r.text or "" for r in recs], np.array([r.label for r in recs])


@dataclass
class SplitMetrics:
    name: str
    n: int
    n_pos: int
    n_neg: int
    roc_auc: float | None
    pr_auc: float | None
    accuracy: float | None
    precision: float | None
    recall: float | None
    f1: float | None
    fpr: float | None
    fpr_ci: tuple[float, float] | None
    brier: float | None
    ece: float | None
    threshold_metrics_usable: bool
    note: str = ""


def evaluate(name: str, y: np.ndarray, score: np.ndarray, threshold: float) -> SplitMetrics:
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    pred = (score >= threshold).astype(int)

    usable = n_neg >= MIN_NEGATIVES_FOR_THRESHOLD_METRIC
    note = "" if usable else (
        f"only {n_neg} negatives: threshold metrics are indicative, not measurements")

    if len(np.unique(y)) < 2:
        return SplitMetrics(name, len(y), n_pos, n_neg, None, None, None, None, None,
                            None, None, None, None, None, False,
                            "single-class split; metrics undefined")

    prec, rec, f1, _ = precision_recall_fscore_support(
        y, pred, average="binary", zero_division=0)
    fp = int(((pred == 1) & (y == 0)).sum())
    fpr = fp / n_neg if n_neg else None
    ci = wilson(fp, n_neg) if n_neg else None

    return SplitMetrics(
        name=name,
        n=len(y),
        n_pos=n_pos,
        n_neg=n_neg,
        roc_auc=float(roc_auc_score(y, score)),
        pr_auc=float(average_precision_score(y, score)),
        accuracy=float((pred == y).mean()),
        precision=float(prec),
        recall=float(rec),
        f1=float(f1),
        fpr=fpr,
        fpr_ci=ci,
        brier=float(np.mean((score - y) ** 2)),
        ece=expected_calibration_error(y, score),
        threshold_metrics_usable=usable,
        note=note,
    )


@dataclass
class BaselineResult:
    config: str
    threshold: float
    threshold_basis: str
    metrics: dict[str, SplitMetrics] = field(default_factory=dict)
    per_source_quarantine: dict[str, SplitMetrics] = field(default_factory=dict)
    per_generator_raid: dict[str, dict] = field(default_factory=dict)
    vocab_size: int = 0
    train_class_counts: dict[str, int] = field(default_factory=dict)


def run_baseline(config: str = "word", records: Sequence[CanonicalRecord] | None = None
                 ) -> BaselineResult:
    if config not in CONFIGS:
        raise ValueError(f"unknown config {config!r}; choose from {sorted(CONFIGS)}")
    recs = list(records) if records is not None else load_text_records()

    train = [r for r in recs if r.split is Split.TRAIN]
    val = [r for r in recs if r.split is Split.VAL]
    test = [r for r in recs if r.split is Split.TEST_ID]
    quarantine = [r for r in recs if r.split is Split.QUARANTINE]

    if not train or not val:
        raise RuntimeError("empty train/val split -- run `python -m ml.cli build` first")

    Xtr, ytr = _xy(train)
    Xva, yva = _xy(val)

    vec = TfidfVectorizer(**CONFIGS[config])
    Xtr_v = vec.fit_transform(Xtr)
    Xva_v = vec.transform(Xva)

    clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced",
                             solver="liblinear")
    clf.fit(Xtr_v, ytr)

    val_score = clf.predict_proba(Xva_v)[:, 1]

    # ---- threshold from VAL only, at the human FPR budget ----
    human_val = val_score[yva == 0]
    if len(human_val) == 0:
        raise RuntimeError("no human records in val; cannot set a false-positive budget")
    threshold = float(np.quantile(human_val, 1.0 - TARGET_FPR))
    basis = (f"quantile({1 - TARGET_FPR:.2f}) of {len(human_val)} human val scores "
             f"-> FPR budget {TARGET_FPR:.0%}")

    res = BaselineResult(
        config=config,
        threshold=threshold,
        threshold_basis=basis,
        vocab_size=len(vec.vocabulary_),
        train_class_counts={"human": int((ytr == 0).sum()), "machine": int((ytr == 1).sum())},
    )

    res.metrics["val"] = evaluate("val", yva, val_score, threshold)

    if test:
        Xte, yte = _xy(test)
        res.metrics["test_id"] = evaluate(
            "test_id", yte, clf.predict_proba(vec.transform(Xte))[:, 1], threshold)

    # ---- independent evaluation (never fitted on) ----
    for source in sorted({r.source_dataset for r in quarantine}):
        sub = [r for r in quarantine if r.source_dataset == source]
        if not sub or len({r.label for r in sub}) < 2:
            continue
        Xq, yq = _xy(sub)
        res.per_source_quarantine[source] = evaluate(
            source, yq, clf.predict_proba(vec.transform(Xq))[:, 1], threshold)

    raid = [r for r in quarantine if r.source_dataset == "raid"]
    if raid:
        Xr, yr = _xy(raid)
        s = clf.predict_proba(vec.transform(Xr))[:, 1]
        gens = sorted({r.generator or "human" for r in raid})
        for g in gens:
            mask = np.array([(r.generator or "human") == g for r in raid])
            if mask.sum() < 20 or len(np.unique(yr[mask])) < 2:
                continue
            res.per_generator_raid[g] = {
                "n": int(mask.sum()),
                "mean_score": float(s[mask].mean()),
                "detected_at_threshold": float((s[mask] >= threshold).mean()),
            }

    return res


def render_markdown(results: list[BaselineResult]) -> str:
    lines = ["# TF-IDF + logistic-regression baseline", ""]
    lines += [
        "Measured on the committed split manifest "
        "(`datasets/manifests/COMBINED.yaml`). No number here is an estimate.", "",
        f"Threshold selected on **val only** at a {TARGET_FPR:.0%} human false-positive "
        "budget, then frozen before the test split was scored.",
        "",
        "Class imbalance (machine:human) is handled with `class_weight='balanced'`.",
        "",
    ]

    for r in results:
        lines += [f"## Configuration: `{r.config}`", "",
                  f"- vocabulary size: {r.vocab_size:,}",
                  f"- train class counts: {r.train_class_counts}",
                  f"- threshold: {r.threshold:.4f} ({r.threshold_basis})", "",
                  "| Split | n | pos | neg | ROC-AUC | PR-AUC | acc | prec | rec | F1 | "
                  "**human FPR (95% CI)** | Brier | ECE |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|"]

        for name, m in r.metrics.items():
            lines.append(_metric_row(name, m))
        for name, m in r.per_source_quarantine.items():
            lines.append(_metric_row(f"quarantine:{name}", m))

        for name, m in {**r.metrics, **r.per_source_quarantine}.items():
            if m.note:
                lines.append(f"| ↳ _{name}_ | | | | | | | | | | {m.note} | | |")

        if r.per_generator_raid:
            lines += ["", "### RAID by generator (independent, never fitted on)", "",
                      "| Generator | n | mean score | fraction flagged at threshold |",
                      "|---|---:|---:|---:|"]
            for g, d in sorted(r.per_generator_raid.items()):
                lines.append(f"| {g} | {d['n']} | {d['mean_score']:.3f} | "
                             f"{d['detected_at_threshold']:.1%} |")
        lines.append("")

    lines += [
        "## How to read this", "",
        "- **Human FPR is the primary metric.** A false accusation on human writing is the "
        "worst product failure; FPR is reported with its Wilson interval and the negative "
        "count.",
        "- **`test_id` is in-domain**: same sources and generators as training, different "
        "prompts/authors. It is the optimistic number.",
        "- **`quarantine:*` is independent.** RAID and HC3 were never fitted on.",
        "- Where a split has fewer than "
        f"{MIN_NEGATIVES_FOR_THRESHOLD_METRIC} negatives, threshold metrics are marked "
        "indicative. Do not cite them as measurements.",
        "- A high in-domain AUC alongside a low independent AUC is the expected shape for "
        "this corpus and is the finding, not a bug.",
        "",
    ]
    return "\n".join(lines)


def _metric_row(name: str, m: SplitMetrics) -> str:
    def f(x, nd=3):
        return "n/a" if x is None else f"{x:.{nd}f}"

    fpr = "n/a"
    if m.fpr is not None and m.fpr_ci is not None:
        fpr = f"{m.fpr:.3f} [{m.fpr_ci[0]:.3f}, {m.fpr_ci[1]:.3f}]"
    return (f"| {name} | {m.n} | {m.n_pos} | {m.n_neg} | {f(m.roc_auc)} | {f(m.pr_auc)} | "
            f"{f(m.accuracy)} | {f(m.precision)} | {f(m.recall)} | {f(m.f1)} | {fpr} | "
            f"{f(m.brier)} | {f(m.ece)} |")


def write_reports(results: list[BaselineResult], outdir: Path = REPORTS) -> tuple[Path, Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    md = outdir / "baseline_tfidf.md"
    js = outdir / "baseline_tfidf.json"
    md.write_text(render_markdown(results), encoding="utf-8")

    payload = []
    for r in results:
        payload.append({
            "config": r.config,
            "threshold": r.threshold,
            "threshold_basis": r.threshold_basis,
            "vocab_size": r.vocab_size,
            "train_class_counts": r.train_class_counts,
            "metrics": {k: _asdict(v) for k, v in r.metrics.items()},
            "per_source_quarantine": {k: _asdict(v) for k, v in r.per_source_quarantine.items()},
            "per_generator_raid": r.per_generator_raid,
        })
    js.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return md, js


def _asdict(m: SplitMetrics) -> dict:
    d = m.__dict__.copy()
    if d["fpr_ci"] is not None:
        d["fpr_ci"] = list(d["fpr_ci"])
    return d
