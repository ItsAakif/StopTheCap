"""E2-C1 encoder fine-tune (distilbert-base-cased).

The E2-C1 corpus is the option-C build: generated E2-C1 passages paired 1:1 with
chunked Ghostbuster human essays.  This trainer mirrors the E1 architecture and
schedule and adds the full evaluation contract:

* train only on ``partition == "train"``;
* choose the decision threshold on ``val`` at a 5% human false-positive budget,
  freeze it, then score ``val``, ``test_holdout``, ``hc3`` and ``raid``;
* report ROC-AUC, PR-AUC, human FPR with a Wilson interval, precision/recall,
  Brier, ECE (15 bins), per-generator and per-topic tables;
* write ``E2C1.json``, ``E2C1_eval.json``, ``E2C1_eval.md`` and
  ``E2C1_predictions.jsonl`` under ``--out``;
* checkpoint ``last.pt`` every 200 optimizer steps and resume when present.

Heavy dependencies (torch/transformers) are imported inside :func:`train`, so the
metric helpers and ``--help`` work on a machine without torch.

Usage
-----
    python -m ml.train.train_e2c1 --help
    python -m ml.train.train_e2c1 --device cuda --out /content/drive/MyDrive/StopTheCap_reports/E2C1
    python -m ml.train.train_e2c1 --device cpu --smoke 5 --max-length 64
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
CONFIG = Path(__file__).resolve().parent / "e2c1_config.yaml"
CORPUS = REPO / "datasets" / "canonical" / "e2c1.jsonl"
MANIFEST_DIR = REPO / "datasets" / "manifests"
MANIFEST = MANIFEST_DIR / "e2_c1_training.json"
CORPUS_MANIFEST = MANIFEST_DIR / "e2_c1_corpus.json"

# Canonical corpus stem -> manifest filename under ``datasets/manifests/``.  New
# corpora can be added here; anything else falls back to ``<stem>.json``.
MANIFEST_BY_CORPUS_STEM: dict[str, str] = {
    "e2c1": MANIFEST.name,
    "e2c1_essays": "e2_c1_essays.json",
}
HC3 = REPO / "datasets" / "canonical" / "hc3.jsonl"
RAID = REPO / "datasets" / "canonical" / "raid.jsonl"
OUT_DEFAULT = REPO / "reports" / "E2C1"

PARTITIONS: tuple[str, ...] = ("train", "val", "test_holdout")
REQUIRED_FIELDS: tuple[str, ...] = (
    "sample_id",
    "text",
    "label",
    "partition",
    "source_dataset",
    "group_id",
    "generator",
    "topic_category",
    "content_hash",
)
TARGET_FPR = 0.05
ECE_BINS = 15


# ---------------------------------------------------------------------------
# Configuration and corpus loading (pure, offline).
# ---------------------------------------------------------------------------


def load_config(path: Path = CONFIG) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


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


def load_corpus(path: Path = CORPUS) -> list[dict[str, Any]]:
    """Load and validate the E2-C1 training corpus."""
    rows = read_jsonl(path)
    if not rows:
        raise ValueError(f"{path}: corpus is empty")
    for index, row in enumerate(rows, start=1):
        missing = [field for field in REQUIRED_FIELDS if field not in row]
        if missing:
            raise ValueError(f"{path}: row {index} missing field(s): {missing}")
        if int(row["label"]) not in (0, 1):
            raise ValueError(f"{path}: row {index} has label {row['label']!r}")
        if str(row["partition"]) not in PARTITIONS:
            raise ValueError(
                f"{path}: row {index} has partition {row['partition']!r}"
            )
        if not str(row["sample_id"]).strip():
            raise ValueError(f"{path}: row {index} has an empty sample_id")
    return rows


def load_canonical(path: Path) -> list[dict[str, Any]]:
    """Load an independent canonical evaluation set (hc3/raid).

    Canonical records carry far more fields than the trainer needs.  Only the
    fields used for scoring and reporting are normalised, and empty texts are
    retained (the tokenizer handles them) so the reported n matches the file.
    """
    rows = read_jsonl(path)
    out: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        for field in ("sample_id", "text", "label"):
            if field not in row:
                raise ValueError(f"{path}: row {index} missing field {field!r}")
        if int(row["label"]) not in (0, 1):
            raise ValueError(f"{path}: row {index} has label {row['label']!r}")
        metadata = row.get("metadata") or {}
        out.append(
            {
                "sample_id": str(row["sample_id"]),
                "text": str(row.get("text") or ""),
                "label": int(row["label"]),
                "source_dataset": str(row.get("source_dataset") or path.stem),
                "generator": row.get("generator"),
                "topic_category": metadata.get("subset") or metadata.get("domain")
                or row.get("domain")
                or "unknown",
            }
        )
    return out


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_corpus_digest(path: Path = CORPUS, expected: str | None = None) -> str:
    """Return the corpus SHA-256, raising when a pinned digest does not match."""
    digest = sha256_file(path)
    if expected and digest != expected:
        raise ValueError(
            f"corpus SHA-256 mismatch: {digest} != pinned {expected} ({path})"
        )
    return digest


def default_manifest_path(corpus_path: Path = CORPUS) -> Path:
    """Derive the manifest path for a canonical corpus file.

    ``e2c1.jsonl`` keeps its historical manifest and
    ``e2c1_essays.jsonl`` maps to ``e2_c1_essays.json``; any other corpus falls
    back to ``datasets/manifests/<stem>.json``.
    """
    stem = Path(corpus_path).stem
    return MANIFEST_DIR / MANIFEST_BY_CORPUS_STEM.get(stem, f"{stem}.json")


def load_manifest(path: Path, *, required: bool = False) -> dict[str, Any]:
    """Read a JSON manifest, optionally requiring that it exists."""
    path = Path(path)
    if not path.is_file():
        if required:
            raise FileNotFoundError(f"manifest not found: {path}")
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: manifest is not a JSON object")
    return data


def load_and_verify_manifest(
    corpus_path: Path, manifest_path: Path, *, required: bool = False
) -> tuple[dict[str, Any], str]:
    """Load a manifest and verify the corpus digest it pins.

    When ``required`` is true (an explicit ``--manifest``), the manifest must
    exist and must contain ``corpus_sha256``.  The legacy default keeps its old
    behaviour: a missing manifest simply means no digest is pinned.
    """
    manifest = load_manifest(manifest_path, required=required)
    if required and not manifest.get("corpus_sha256"):
        raise ValueError(f"{manifest_path}: manifest is missing corpus_sha256")
    digest = verify_corpus_digest(Path(corpus_path), manifest.get("corpus_sha256"))
    return manifest, digest


def partition_rows(
    rows: Sequence[Mapping[str, Any]], partition: str
) -> list[dict[str, Any]]:
    return [dict(row) for row in rows if str(row["partition"]) == partition]


def class_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return {
        "human": sum(1 for row in rows if int(row["label"]) == 0),
        "machine": sum(1 for row in rows if int(row["label"]) == 1),
    }


def class_weights(rows: Sequence[Mapping[str, Any]]) -> tuple[float, float]:
    """Inverse-frequency weights for ``[human, machine]`` (balanced contribution)."""
    counts = class_counts(rows)
    total = counts["human"] + counts["machine"]
    if not counts["human"] or not counts["machine"]:
        raise RuntimeError("train split is single-class; refusing to train")
    return (
        total / (2 * counts["human"]),
        total / (2 * counts["machine"]),
    )


def planned_steps(n_rows: int, batch_size: int, grad_accum: int, epochs: int) -> dict:
    """Optimizer-step plan for a row count and an effective batch size."""
    if batch_size <= 0 or grad_accum <= 0 or epochs <= 0:
        raise ValueError("batch_size, grad_accum and epochs must be positive")
    micro_batches = math.ceil(n_rows / batch_size)
    steps_per_epoch = max(1, math.ceil(micro_batches / grad_accum))
    return {
        "rows": int(n_rows),
        "micro_batches_per_epoch": int(micro_batches),
        "steps_per_epoch": int(steps_per_epoch),
        "total_steps": int(steps_per_epoch * epochs),
    }


# ---------------------------------------------------------------------------
# Metrics (pure numpy; no torch).
# ---------------------------------------------------------------------------


def select_threshold(human_scores: Iterable[float], target_fpr: float = TARGET_FPR) -> float:
    """Quantile cut that flags roughly ``target_fpr`` of human validation rows."""
    scores = np.asarray(list(human_scores), dtype=float)
    if scores.size == 0:
        raise ValueError("cannot select a threshold without human scores")
    if not 0.0 < target_fpr < 1.0:
        raise ValueError("target_fpr must be in (0, 1)")
    return float(np.quantile(scores, 1.0 - target_fpr, method="linear"))


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def roc_auc(y: Iterable[int], scores: Iterable[float]) -> float | None:
    """Rank-based ROC-AUC, tie-aware.  Returns None for a single-class split."""
    y_arr = np.asarray(list(y), dtype=int)
    s_arr = np.asarray(list(scores), dtype=float)
    if y_arr.size != s_arr.size:
        raise ValueError("y and scores must have the same length")
    positive = y_arr == 1
    n_pos = int(positive.sum())
    n_neg = int((~positive).sum())
    if n_pos == 0 or n_neg == 0:
        return None

    order = np.argsort(s_arr, kind="mergesort")
    sorted_scores = s_arr[order]
    ranks = np.empty(len(s_arr), dtype=float)
    start = 0
    while start < len(sorted_scores):
        end = start
        while end + 1 < len(sorted_scores) and sorted_scores[end + 1] == sorted_scores[start]:
            end += 1
        ranks[order[start:end + 1]] = (start + end) / 2.0 + 1.0
        start = end + 1
    return float(
        (ranks[positive].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    )


def pr_auc(y: Iterable[int], scores: Iterable[float]) -> float | None:
    """Average precision (step-wise area under the precision/recall curve)."""
    y_arr = np.asarray(list(y), dtype=int)
    s_arr = np.asarray(list(scores), dtype=float)
    n_pos = int((y_arr == 1).sum())
    if n_pos == 0:
        return None
    order = np.argsort(-s_arr, kind="mergesort")
    y_sorted = y_arr[order]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)
    precision = tp / np.maximum(tp + fp, 1)
    return float(precision[y_sorted == 1].sum() / n_pos)


def brier(y: Iterable[int], scores: Iterable[float]) -> float:
    y_arr = np.asarray(list(y), dtype=float)
    s_arr = np.asarray(list(scores), dtype=float)
    return float(np.mean((s_arr - y_arr) ** 2))


def ece(
    y: Iterable[int], scores: Iterable[float], bins: int = ECE_BINS
) -> float:
    """Expected calibration error over equal-width probability bins."""
    y_arr = np.asarray(list(y), dtype=float)
    s_arr = np.asarray(list(scores), dtype=float)
    if y_arr.size == 0:
        raise ValueError("cannot compute ECE on an empty split")
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (s_arr >= lo) & (s_arr < hi if i < bins - 1 else s_arr <= hi)
        if not mask.any():
            continue
        total += abs(s_arr[mask].mean() - y_arr[mask].mean()) * mask.sum() / len(y_arr)
    return float(total)


def reliability_bins(
    y: Iterable[int], scores: Iterable[float], bins: int = ECE_BINS
) -> list[dict[str, Any]]:
    y_arr = np.asarray(list(y), dtype=float)
    s_arr = np.asarray(list(scores), dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    out: list[dict[str, Any]] = []
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (s_arr >= lo) & (s_arr < hi if i < bins - 1 else s_arr <= hi)
        out.append(
            {
                "lo": round(float(lo), 4),
                "hi": round(float(hi), 4),
                "n": int(mask.sum()),
                "mean_pred": round(float(s_arr[mask].mean()), 4) if mask.any() else None,
                "frac_pos": round(float(y_arr[mask].mean()), 4) if mask.any() else None,
            }
        )
    return out


def precision_recall_at_threshold(
    y: Iterable[int], scores: Iterable[float], threshold: float
) -> dict[str, Any]:
    y_arr = np.asarray(list(y), dtype=int)
    s_arr = np.asarray(list(scores), dtype=float)
    pred = (s_arr >= threshold).astype(int)
    tp = int(((pred == 1) & (y_arr == 1)).sum())
    fp = int(((pred == 1) & (y_arr == 0)).sum())
    fn = int(((pred == 0) & (y_arr == 1)).sum())
    tn = int(((pred == 0) & (y_arr == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


def human_fpr(
    y: Iterable[int], scores: Iterable[float], threshold: float
) -> dict[str, Any]:
    y_arr = np.asarray(list(y), dtype=int)
    s_arr = np.asarray(list(scores), dtype=float)
    human = y_arr == 0
    n_human = int(human.sum())
    fp = int((s_arr[human] >= threshold).sum()) if n_human else 0
    fpr = fp / n_human if n_human else None
    ci = wilson(fp, n_human) if n_human else None
    return {
        "false_positives": fp,
        "n_human": n_human,
        "human_fpr": fpr,
        "human_fpr_ci": list(ci) if ci is not None else None,
    }


def evaluate_split(
    name: str,
    rows: Sequence[Mapping[str, Any]],
    scores: Sequence[float],
    threshold: float,
    *,
    ece_bins: int = ECE_BINS,
) -> dict[str, Any]:
    y = np.asarray([int(row["label"]) for row in rows], dtype=int)
    s = np.asarray(list(scores), dtype=float)
    if y.size != s.size:
        raise ValueError(f"{name}: {y.size} labels but {s.size} scores")
    n_human = int((y == 0).sum())
    n_machine = int((y == 1).sum())
    at = precision_recall_at_threshold(y, s, threshold)
    fpr = human_fpr(y, s, threshold)
    report: dict[str, Any] = {
        "split": name,
        "n": int(y.size),
        "n_human": n_human,
        "n_machine": n_machine,
        "roc_auc": roc_auc(y, s),
        "pr_auc": pr_auc(y, s),
        "human_fpr": fpr["human_fpr"],
        "human_fpr_ci": fpr["human_fpr_ci"],
        "false_positives": fpr["false_positives"],
        "precision": at["precision"],
        "recall": at["recall"],
        "f1": at["f1"],
        "confusion": at["confusion"],
        "brier": brier(y, s),
        "ece": ece(y, s, bins=ece_bins),
        "mean_score": float(s.mean()) if s.size else None,
    }
    if n_human < 1000:
        report["note"] = f"only {n_human} human rows: threshold metrics are indicative"
    return report


def per_generator_metrics(
    rows: Sequence[Mapping[str, Any]],
    scores: Sequence[float],
    threshold: float,
) -> dict[str, dict[str, Any]]:
    """Detection rate per E2-C1 generator (machine rows only)."""
    grouped: dict[str, list[tuple[Mapping[str, Any], float]]] = defaultdict(list)
    for row, score in zip(rows, scores):
        generator = row.get("generator")
        if generator is None:
            continue
        grouped[str(generator)].append((row, float(score)))
    out: dict[str, dict[str, Any]] = {}
    for generator, items in sorted(grouped.items()):
        y = np.asarray([int(row["label"]) for row, _ in items], dtype=int)
        s = np.asarray([score for _, score in items], dtype=float)
        detected = int((s >= threshold).sum())
        out[generator] = {
            "n": int(len(items)),
            "detected": detected,
            "detection_rate": float(detected / len(items)),
            "mean_score": float(s.mean()),
            "roc_auc": roc_auc(y, s),
        }
    return out


def per_topic_metrics(
    rows: Sequence[Mapping[str, Any]],
    scores: Sequence[float],
    threshold: float,
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[tuple[Mapping[str, Any], float]]] = defaultdict(list)
    for row, score in zip(rows, scores):
        grouped[str(row.get("topic_category") or "unknown")].append((row, float(score)))
    out: dict[str, dict[str, Any]] = {}
    for topic, items in sorted(grouped.items()):
        y = np.asarray([int(row["label"]) for row, _ in items], dtype=int)
        s = np.asarray([score for _, score in items], dtype=float)
        human = y == 0
        machine = y == 1
        out[topic] = {
            "n": int(len(items)),
            "n_human": int(human.sum()),
            "n_machine": int(machine.sum()),
            "human_fpr": float((s[human] >= threshold).mean()) if human.any() else None,
            "recall": float((s[machine] >= threshold).mean()) if machine.any() else None,
            "mean_score": float(s.mean()),
            "roc_auc": roc_auc(y, s),
        }
    return out


# ---------------------------------------------------------------------------
# Environment / reporting.
# ---------------------------------------------------------------------------


def environment_report() -> dict[str, Any]:
    import sklearn
    import torch
    import transformers

    cuda = bool(torch.cuda.is_available())
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda_available": cuda,
        "torch_cuda_device": torch.cuda.get_device_name(0) if cuda else None,
        "torch_cuda_vram_gb": (
            round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)
            if cuda
            else None
        ),
        "transformers": transformers.__version__,
        "sklearn": sklearn.__version__,
        "numpy": np.__version__,
    }


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# E2-C1 encoder evaluation",
        "",
        f"- experiment: **{result['experiment']}**",
        f"- model: `{result['model_name']}`",
        f"- device: **{result['device']}**",
        f"- max_length: {result['max_length']}",
        f"- corpus SHA-256: `{result['corpus_sha256']}`",
        f"- split digest: `{result.get('split_digest')}`",
        f"- frozen threshold: **{result['frozen_threshold']:.6f}** "
        f"({result['threshold_basis']})",
        f"- target human FPR: {result['target_fpr']:.0%}",
        "",
        "## Splits",
        "",
        "| Split | n | human | machine | ROC-AUC | PR-AUC | human FPR (95% CI) | "
        "Precision | Recall | F1 | Brier | ECE (15) |",
        "|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    for name in ("val", "test_holdout", "hc3", "raid"):
        report = result["splits"].get(name)
        if report is None:
            continue
        ci = report.get("human_fpr_ci")
        fpr = "n/a"
        if report["human_fpr"] is not None and ci:
            fpr = f"{report['human_fpr']:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]"
        lines.append(
            f"| {name} | {report['n']} | {report['n_human']} | "
            f"{report['n_machine']} | {_fmt(report['roc_auc'])} | "
            f"{_fmt(report['pr_auc'])} | {fpr} | {_fmt(report['precision'])} | "
            f"{_fmt(report['recall'])} | {_fmt(report['f1'])} | "
            f"{_fmt(report['brier'], 4)} | {_fmt(report['ece'], 4)} |"
        )

    generators = result.get("per_generator") or {}
    if generators:
        lines += [
            "",
            "## Per-generator detection (E2-C1 test_holdout)",
            "",
            "| Generator | n | detected | detection rate | mean score |",
            "|---|---:|---:|---:|---:|",
        ]
        for generator, values in generators.items():
            lines.append(
                f"| {generator} | {values['n']} | {values['detected']} | "
                f"{values['detection_rate']:.3f} | {values['mean_score']:.3f} |"
            )

    topics = result.get("per_topic") or {}
    if topics:
        lines += [
            "",
            "## Per-topic (E2-C1 test_holdout)",
            "",
            "| Topic | n | human | machine | human FPR | recall | ROC-AUC |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for topic, values in topics.items():
            lines.append(
                f"| {topic} | {values['n']} | {values['n_human']} | "
                f"{values['n_machine']} | {_fmt(values['human_fpr'])} | "
                f"{_fmt(values['recall'])} | {_fmt(values['roc_auc'])} |"
            )

    notes = [
        report["note"]
        for report in result["splits"].values()
        if report.get("note")
    ]
    if notes:
        lines += ["", "## Caveats"] + [f"- {note}" for note in notes]

    lines += [
        "",
        "## How to read this",
        "",
        "- `test_holdout` is in-domain but unseen; `hc3` and `raid` are independent,",
        "  out-of-distribution sets and were never used to choose the threshold.",
        "- The threshold came from `val` only and was frozen before any test split",
        "  was scored.",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Training (torch imported lazily).
# ---------------------------------------------------------------------------


def _stratified(
    rows: Sequence[Mapping[str, Any]], n: int
) -> list[dict[str, Any]]:
    by_label: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_label[int(row["label"])].append(dict(row))
    per_label = max(1, n // max(1, len(by_label)))
    out: list[dict[str, Any]] = []
    for label in sorted(by_label):
        out.extend(by_label[label][:per_label])
    return out


def _resolve_outputs(cfg: Mapping[str, Any], out_dir: Path) -> dict[str, Path]:
    output = cfg.get("output") or {}
    return {
        "run_log": out_dir / str(output.get("run_log", "E2C1.json")),
        "eval_json": out_dir / str(output.get("eval_json", "E2C1_eval.json")),
        "eval_md": out_dir / str(output.get("eval_md", "E2C1_eval.md")),
        "predictions": out_dir
        / str(output.get("predictions", "E2C1_predictions.jsonl")),
    }


def train(
    cfg: dict[str, Any],
    *,
    corpus_path: Path = CORPUS,
    out_dir: Path = OUT_DEFAULT,
    device: str | None = None,
    smoke_steps: int | None = None,
    max_length: int | None = None,
    manifest_path: Path | None = None,
    manifest_required: bool = False,
) -> int:
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )

    started = time.perf_counter()
    out_dir = Path(out_dir)
    outputs = _resolve_outputs(cfg, out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- corpus integrity -------------------------------------------------
    manifest_path = (
        Path(manifest_path)
        if manifest_path is not None
        else default_manifest_path(Path(corpus_path))
    )
    manifest, corpus_sha256 = load_and_verify_manifest(
        Path(corpus_path), manifest_path, required=manifest_required
    )
    corpus_manifest: dict[str, Any] = {}
    if "split_digest" in manifest:
        corpus_manifest = manifest
    elif Path(corpus_path).stem == "e2c1" and CORPUS_MANIFEST.is_file():
        corpus_manifest = json.loads(CORPUS_MANIFEST.read_text(encoding="utf-8"))

    rows = load_corpus(Path(corpus_path))
    train_rows = partition_rows(rows, "train")
    val_rows = partition_rows(rows, "val")
    test_rows = partition_rows(rows, "test_holdout")
    if not train_rows:
        print(f"ERROR: {len(rows)} rows loaded but none are in the train partition.",
              file=sys.stderr)
        return 4

    smoke = smoke_steps is not None
    if max_length is not None:
        cfg = {**cfg, "max_length": int(max_length)}
    if smoke:
        cfg = {**cfg, "max_length": min(int(cfg["max_length"]), 128),
               "batch_size": min(int(cfg["batch_size"]), 8)}
        train_rows = _stratified(train_rows, 2000)
        val_rows = _stratified(val_rows, 400)
        test_rows = _stratified(test_rows, 400)

    seed = int(cfg["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))

    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if resolved_device.startswith("cuda") and not torch.cuda.is_available():
        print(f"ERROR: requested {resolved_device} but CUDA is unavailable.",
              file=sys.stderr)
        return 3

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])

    class _TokenDataset(Dataset):
        def __init__(self, records: Sequence[Mapping[str, Any]]) -> None:
            self.records = list(records)

        def __len__(self) -> int:
            return len(self.records)

        def __getitem__(self, index: int) -> dict[str, Any]:
            record = self.records[index]
            encoded = tokenizer(
                str(record.get("text") or ""),
                truncation=True,
                max_length=int(cfg["max_length"]),
                padding="max_length",
                return_tensors="pt",
            )
            return {
                "input_ids": encoded["input_ids"][0],
                "attention_mask": encoded["attention_mask"][0],
                "labels": torch.tensor(int(record["label"]), dtype=torch.long),
            }

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["model_name"], num_labels=2
    ).to(resolved_device)

    human_weight, machine_weight = class_weights(train_rows)
    weight = torch.tensor(
        [human_weight, machine_weight], dtype=torch.float, device=resolved_device
    )
    loss_fn = torch.nn.CrossEntropyLoss(weight=weight)

    batch_size = int(cfg["batch_size"])
    grad_accum = int(cfg["grad_accum"])
    loader = DataLoader(
        _TokenDataset(train_rows),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=int(cfg.get("num_workers", 0)),
    )
    plan = planned_steps(len(train_rows), batch_size, grad_accum, int(cfg["epochs"]))
    total_steps = plan["total_steps"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg["weight_decay"]),
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * float(cfg["warmup_ratio"])),
        num_training_steps=total_steps,
    )

    use_amp = bool(cfg.get("fp16")) and resolved_device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    checkpoint_cfg = cfg.get("checkpoint") or {}
    checkpoint_dir = out_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    state_path = checkpoint_dir / "last.pt"
    start_epoch, global_step = 0, 0

    if checkpoint_cfg.get("resume", True) and state_path.is_file():
        state = torch.load(state_path, map_location=resolved_device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["opt"])
        scheduler.load_state_dict(state["sched"])
        start_epoch = int(state.get("epoch", 0))
        global_step = int(state.get("step", 0))
        print(f"resumed from {state_path} at epoch {start_epoch} step {global_step}")

    every_steps = int(checkpoint_cfg.get("every_steps", 200))
    stop_after = int(smoke_steps) if smoke else None
    pending = 0
    completed_epochs = start_epoch

    def _save(epoch: int) -> None:
        torch.save(
            {
                "model": model.state_dict(),
                "opt": optimizer.state_dict(),
                "sched": scheduler.state_dict(),
                "epoch": int(epoch),
                "step": int(global_step),
            },
            state_path,
        )

    def _optimizer_step() -> None:
        nonlocal global_step, pending
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        pending = 0
        global_step += 1
        if every_steps > 0 and global_step % every_steps == 0:
            _save(epoch)
            print(f"checkpoint @ epoch {epoch} step {global_step}", flush=True)

    for epoch in range(start_epoch, int(cfg["epochs"])):
        if stop_after is not None and global_step >= stop_after:
            break
        model.train()
        for batch in loader:
            batch = {key: value.to(resolved_device) for key, value in batch.items()}
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                ).logits
                loss = loss_fn(logits, batch["labels"]) / grad_accum
            scaler.scale(loss).backward()
            pending += 1
            if pending == grad_accum:
                _optimizer_step()
                if stop_after is not None and global_step >= stop_after:
                    break
        if pending:
            _optimizer_step()
        completed_epochs = epoch + 1
        if stop_after is not None and global_step >= stop_after:
            print(f"smoke stop at step {global_step}", flush=True)
            break

    _save(completed_epochs)

    # ---- scoring and frozen threshold ------------------------------------
    model.eval()

    def _score(records: Sequence[Mapping[str, Any]]) -> np.ndarray:
        if not records:
            return np.array([], dtype=float)
        out: list[float] = []
        score_loader = DataLoader(_TokenDataset(records), batch_size=batch_size)
        with torch.no_grad():
            for batch in score_loader:
                batch = {key: value.to(resolved_device) for key, value in batch.items()}
                with torch.amp.autocast("cuda", enabled=use_amp):
                    logits = model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                    ).logits
                probs = torch.softmax(logits.float(), dim=-1)[:, 1]
                out.extend(probs.cpu().numpy().tolist())
        return np.asarray(out, dtype=float)

    threshold_started = time.perf_counter()
    val_scores = _score(val_rows)
    val_labels = np.asarray([int(row["label"]) for row in val_rows], dtype=int)
    human_val_scores = val_scores[val_labels == 0]
    frozen_threshold = select_threshold(
        human_val_scores, float(cfg["threshold"]["target_human_fpr"])
    )
    threshold_frozen_at = time.perf_counter()
    print(
        f"frozen threshold {frozen_threshold:.6f} "
        f"(quantile {1 - float(cfg['threshold']['target_human_fpr']):.2f} of "
        f"{len(human_val_scores)} human val scores)",
        flush=True,
    )

    # ---- evaluation -------------------------------------------------------
    eval_sets: dict[str, list[dict[str, Any]]] = {
        "val": val_rows,
        "test_holdout": test_rows,
    }
    if HC3.is_file():
        hc3_rows = load_canonical(HC3)
        eval_sets["hc3"] = _stratified(hc3_rows, 500) if smoke else hc3_rows
    else:
        print(f"WARNING: independent set missing: {HC3}", file=sys.stderr)
    if RAID.is_file():
        raid_rows = load_canonical(RAID)
        eval_sets["raid"] = _stratified(raid_rows, 500) if smoke else raid_rows
    else:
        print(f"WARNING: independent set missing: {RAID}", file=sys.stderr)

    split_reports: dict[str, dict[str, Any]] = {}
    split_scores: dict[str, np.ndarray] = {}
    all_predictions: list[dict[str, Any]] = []
    for name, records in eval_sets.items():
        scores = _score(records)
        split_scores[name] = scores
        split_reports[name] = evaluate_split(name, records, scores, frozen_threshold)
        for record, score in zip(records, scores):
            all_predictions.append(
                {
                    "split": name,
                    "sample_id": record["sample_id"],
                    "dataset": record.get("source_dataset"),
                    "label": int(record["label"]),
                    "generator": record.get("generator"),
                    "topic_category": record.get("topic_category"),
                    "score": float(score),
                    "threshold": float(frozen_threshold),
                }
            )
        print(
            f"  {name:13} n={len(records):<6} "
            f"AUC={_fmt(split_reports[name]['roc_auc'])} "
            f"FPR={_fmt(split_reports[name]['human_fpr'])}",
            flush=True,
        )

    test_scores = split_scores.get("test_holdout", np.array([], dtype=float))
    per_generator = per_generator_metrics(test_rows, test_scores, frozen_threshold)
    per_topic = per_topic_metrics(test_rows, test_scores, frozen_threshold)

    eval_result = {
        "experiment": cfg["experiment"],
        "model_name": cfg["model_name"],
        "device": resolved_device,
        "max_length": int(cfg["max_length"]),
        "corpus": str(corpus_path),
        "corpus_sha256": corpus_sha256,
        "manifest": str(manifest_path),
        "split_digest": corpus_manifest.get("split_digest"),
        "frozen_threshold": frozen_threshold,
        "target_fpr": float(cfg["threshold"]["target_human_fpr"]),
        "threshold_basis": (
            f"quantile({1 - float(cfg['threshold']['target_human_fpr']):.2f}) "
            f"of {len(human_val_scores)} human val scores"
        ),
        "splits": split_reports,
        "per_generator": per_generator,
        "per_topic": per_topic,
    }

    # ---- write artefacts --------------------------------------------------
    run_log = {
        "experiment": cfg["experiment"],
        "status": "SMOKE" if smoke else "TRAINED",
        "smoke": smoke,
        "smoke_steps": smoke_steps,
        "config": cfg,
        "environment": environment_report(),
        "corpus": str(corpus_path),
        "corpus_sha256": corpus_sha256,
        "manifest": str(manifest_path),
        "split_digest": corpus_manifest.get("split_digest"),
        "partition_counts": {
            partition: len(partition_rows(rows, partition)) for partition in PARTITIONS
        },
        "train_class_counts": class_counts(train_rows),
        "train_class_weights": [human_weight, machine_weight],
        "step_plan": plan,
        "steps_completed": int(global_step),
        "frozen_threshold": frozen_threshold,
        "threshold_basis": eval_result["threshold_basis"],
        "checkpoint": str(state_path),
        "timing": {
            "threshold_pass_s": round(threshold_frozen_at - threshold_started, 2),
            "total_s": round(time.perf_counter() - started, 2),
        },
        "outputs": {key: str(path) for key, path in outputs.items()},
        "evaluation": {
            name: {
                "n": report["n"],
                "roc_auc": report["roc_auc"],
                "pr_auc": report["pr_auc"],
                "human_fpr": report["human_fpr"],
                "human_fpr_ci": report["human_fpr_ci"],
                "precision": report["precision"],
                "recall": report["recall"],
                "brier": report["brier"],
                "ece": report["ece"],
            }
            for name, report in split_reports.items()
        },
        "note": (
            "SMOKE run: reduced data/steps. Do not cite these metrics."
            if smoke
            else "Threshold selected on val only and frozen before evaluation."
        ),
    }

    outputs["run_log"].write_text(json.dumps(run_log, indent=2), encoding="utf-8")
    outputs["eval_json"].write_text(
        json.dumps(eval_result, indent=2), encoding="utf-8"
    )
    outputs["eval_md"].write_text(render_markdown(eval_result), encoding="utf-8")
    with outputs["predictions"].open("w", encoding="utf-8") as handle:
        for prediction in all_predictions:
            handle.write(json.dumps(prediction, ensure_ascii=False) + "\n")

    print(f"wrote {outputs['run_log']}")
    print(f"wrote {outputs['eval_json']}")
    print(f"wrote {outputs['eval_md']}")
    print(f"wrote {outputs['predictions']}")
    return 0


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ml.train.train_e2c1")
    parser.add_argument("--config", default=str(CONFIG))
    parser.add_argument("--corpus", default=str(CORPUS))
    parser.add_argument(
        "--manifest",
        default=None,
        help=(
            "corpus manifest whose corpus_sha256 is validated (default: derived "
            "under datasets/manifests/; e2c1.jsonl -> e2_c1_training.json, "
            "e2c1_essays.jsonl -> e2_c1_essays.json)"
        ),
    )
    parser.add_argument("--out", default=str(OUT_DEFAULT))
    parser.add_argument(
        "--device", default=None, help="cuda / cuda:0 / cpu (default: cuda if available)"
    )
    parser.add_argument(
        "--smoke", type=int, default=None, metavar="STEPS",
        help="run N optimizer steps only, for plumbing validation",
    )
    parser.add_argument(
        "--max-length", type=int, default=None,
        help="override the tokeniser length (use 64/128 for a CPU smoke run)",
    )
    args = parser.parse_args(argv)

    try:
        cfg = load_config(Path(args.config))
    except (OSError, yaml.YAMLError) as exc:
        print(f"ERROR: cannot read config {args.config}: {exc}", file=sys.stderr)
        return 2

    try:
        return train(
            cfg,
            corpus_path=Path(args.corpus),
            out_dir=Path(args.out),
            device=args.device,
            smoke_steps=args.smoke,
            max_length=args.max_length,
            manifest_path=(
                Path(args.manifest)
                if args.manifest
                else default_manifest_path(Path(args.corpus))
            ),
            manifest_required=args.manifest is not None,
        )
    except FileNotFoundError as exc:
        print(f"ERROR: input not found: {exc}", file=sys.stderr)
        return 2
    except (ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
