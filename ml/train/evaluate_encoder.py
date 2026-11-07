"""Evaluate an encoder checkpoint across every required split.

Produces the full metric set the sprint asks for: ROC-AUC, PR-AUC, human false-positive rate
with a Wilson interval, precision/recall/F1, confusion matrix, calibration (Brier, ECE,
reliability bins), and a bootstrap interval for AUC.

Rules enforced structurally, not by convention:

* The decision threshold is chosen on ``val`` at the human false-positive budget and is FROZEN
  before any test split is scored.
* HC3 and RAID are loaded only to be scored. No threshold, hyperparameter or feature decision
  reads their labels.
* Every result carries the split digest so it can be tied to an exact dataset version.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
CONFIG = Path(__file__).resolve().parent / "encoder_config.yaml"
REPORTS = REPO / "reports"

TARGET_FPR = 0.05
BOOTSTRAP = 1000


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        m = (p >= lo) & (p < hi if i < bins - 1 else p <= hi)
        if not m.any():
            continue
        total += abs(p[m].mean() - y[m].mean()) * m.sum() / len(y)
    return float(total)


def reliability_bins(y: np.ndarray, p: np.ndarray, bins: int = 10) -> list[dict]:
    edges = np.linspace(0.0, 1.0, bins + 1)
    out = []
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        m = (p >= lo) & (p < hi if i < bins - 1 else p <= hi)
        out.append({"lo": round(float(lo), 2), "hi": round(float(hi), 2),
                    "n": int(m.sum()),
                    "mean_pred": round(float(p[m].mean()), 4) if m.any() else None,
                    "frac_pos": round(float(y[m].mean()), 4) if m.any() else None})
    return out


def bootstrap_auc(y: np.ndarray, p: np.ndarray, n: int = BOOTSTRAP, seed: int = 42) -> dict:
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(seed)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    if len(pos) < 2 or len(neg) < 2:
        return {"point": None, "lo": None, "hi": None, "note": "insufficient rows per class"}
    vals = []
    for _ in range(n):
        idx = np.concatenate([rng.choice(pos, len(pos), replace=True),
                              rng.choice(neg, len(neg), replace=True)])
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(roc_auc_score(y[idx], p[idx]))
    return {"point": float(roc_auc_score(y, p)),
            "lo": float(np.percentile(vals, 2.5)), "hi": float(np.percentile(vals, 97.5)),
            "n_boot": len(vals)}


@dataclass
class SplitReport:
    name: str
    n: int
    n_pos: int
    n_neg: int
    roc_auc: float | None
    pr_auc: float | None
    auc_ci: dict
    human_fpr: float | None
    human_fpr_ci: tuple[float, float] | None
    precision: float | None
    recall: float | None
    f1: float | None
    confusion: dict | None
    brier: float | None
    ece: float | None
    reliability: list[dict] = field(default_factory=list)
    note: str = ""


def evaluate_split(name: str, y: np.ndarray, score: np.ndarray, threshold: float) -> SplitReport:
    from sklearn.metrics import (average_precision_score, confusion_matrix,
                                 precision_recall_fscore_support, roc_auc_score)

    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    if len(np.unique(y)) < 2:
        return SplitReport(name, len(y), n_pos, n_neg, None, None, {}, None, None,
                           None, None, None, None, None, None, [], "single-class split")

    pred = (score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    prec, rec, f1, _ = precision_recall_fscore_support(y, pred, average="binary",
                                                       zero_division=0)
    fpr = fp / n_neg if n_neg else None
    note = "" if n_neg >= 1000 else f"only {n_neg} negatives: threshold metrics are indicative"
    return SplitReport(
        name=name, n=len(y), n_pos=n_pos, n_neg=n_neg,
        roc_auc=float(roc_auc_score(y, score)),
        pr_auc=float(average_precision_score(y, score)),
        auc_ci=bootstrap_auc(y, score),
        human_fpr=fpr,
        human_fpr_ci=wilson(int(fp), n_neg) if n_neg else None,
        precision=float(prec), recall=float(rec), f1=float(f1),
        confusion={"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        brier=float(np.mean((score - y) ** 2)),
        ece=ece(y, score),
        reliability=reliability_bins(y, score),
        note=note,
    )


def load_model(config: dict, checkpoint: Path):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(config["model_name"])
    model = AutoModelForSequenceClassification.from_pretrained(
        config["model_name"], num_labels=2).to(device)

    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    model.load_state_dict(state)
    model.eval()
    return model, tok, device


def score_records(model, tok, device, records, max_length: int, batch_size: int) -> np.ndarray:
    import torch

    out: list[float] = []
    with torch.no_grad():
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            enc = tok([r.text or "" for r in batch], truncation=True, padding=True,
                      max_length=max_length, return_tensors="pt").to(device)
            logits = model(**enc).logits
            out.extend(torch.softmax(logits, dim=-1)[:, 1].cpu().numpy().tolist())
    return np.array(out)


def run(config_path: Path, checkpoint: Path, out_stem: str = "E1_eval",
        max_length: int | None = None) -> dict:
    from ml.ingest.schema import Split
    from ml.baselines.tfidf import load_text_records

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if max_length:
        config["max_length"] = max_length
    records = load_text_records()

    model, tok, device = load_model(config, checkpoint)

    def pick(pred):
        return [r for r in records if pred(r)]

    val = pick(lambda r: r.split is Split.VAL)
    splits = {
        "val": val,
        "test_id": pick(lambda r: r.split is Split.TEST_ID),
        "test_ood_domain": pick(lambda r: r.split is Split.TEST_OOD_DOMAIN),
        "hc3": pick(lambda r: r.source_dataset == "hc3"),
        "raid": pick(lambda r: r.source_dataset == "raid"),
    }

    t0 = time.perf_counter()
    val_scores = score_records(model, tok, device, val, config["max_length"], config["batch_size"])
    val_y = np.array([r.label for r in val])
    humans = val_scores[val_y == 0]
    threshold = float(np.quantile(humans, 1 - TARGET_FPR))
    frozen_at = time.perf_counter()

    reports = []
    for name, recs in splits.items():
        if not recs:
            continue
        s = score_records(model, tok, device, recs, config["max_length"], config["batch_size"])
        reports.append(evaluate_split(name, np.array([r.label for r in recs]), s, threshold))

    combined = yaml.safe_load((REPO / "datasets" / "manifests" / "COMBINED.yaml")
                              .read_text(encoding="utf-8")) if (
        REPO / "datasets" / "manifests" / "COMBINED.yaml").is_file() else {}

    result = {
        "checkpoint": str(checkpoint),
        "model": config["model_name"],
        "device": device,
        "max_length": config["max_length"],
        "frozen_threshold": threshold,
        "threshold_basis": f"quantile({1 - TARGET_FPR:.2f}) of {len(humans)} human val scores",
        "split_digest": combined.get("split_digest"),
        "timing": {"threshold_pass_s": round(frozen_at - t0, 2),
                   "total_s": round(time.perf_counter() - t0, 2)},
        "splits": [asdict(r) for r in reports],
    }

    REPORTS.mkdir(parents=True, exist_ok=True)
    js = REPORTS / f"{out_stem}.json"
    js.write_text(json.dumps(result, indent=2), encoding="utf-8")
    (REPORTS / f"{out_stem}.md").write_text(render(result), encoding="utf-8")
    return result


def render(result: dict) -> str:
    lines = ["# Encoder evaluation", "",
             f"- model: `{result['model']}`",
             f"- checkpoint: `{result['checkpoint']}`",
             f"- device: **{result['device']}**",
             f"- max_length: {result['max_length']}",
             f"- frozen threshold: **{result['frozen_threshold']:.4f}** "
             f"({result['threshold_basis']})",
             f"- split digest: `{result.get('split_digest')}`", "",
             "| Split | n | pos | neg | ROC-AUC (95% CI) | PR-AUC | human FPR (95% CI) | P | R | F1 | Brier | ECE |",
             "|---|---:|---:|---:|---|---:|---|---:|---:|---:|---:|---:|"]

    def f(x, nd=3):
        return "n/a" if x is None else f"{x:.{nd}f}"

    for s in result["splits"]:
        ci = s.get("auc_ci") or {}
        auc = f(s["roc_auc"])
        if ci.get("lo") is not None:
            auc = f"{s['roc_auc']:.3f} [{ci['lo']:.3f}, {ci['hi']:.3f}]"
        fpr = "n/a"
        if s["human_fpr"] is not None and s["human_fpr_ci"]:
            fpr = f"{s['human_fpr']:.3f} [{s['human_fpr_ci'][0]:.3f}, {s['human_fpr_ci'][1]:.3f}]"
        lines.append(f"| {s['name']} | {s['n']} | {s['n_pos']} | {s['n_neg']} | {auc} | "
                     f"{f(s['pr_auc'])} | {fpr} | {f(s['precision'])} | {f(s['recall'])} | "
                     f"{f(s['f1'])} | {f(s['brier'])} | {f(s['ece'])} |")

    lines += ["", "## Confusion matrices (at the frozen threshold)", ""]
    for s in result["splits"]:
        c = s.get("confusion")
        if c:
            lines.append(f"- **{s['name']}**: TN {c['tn']} · FP {c['fp']} · FN {c['fn']} · TP {c['tp']}")

    notes = [s["note"] for s in result["splits"] if s.get("note")]
    if notes:
        lines += ["", "## Caveats"] + [f"- {n}" for n in notes]

    lines += ["", "## How to read this", "",
              "- `test_id` is in-domain and is the optimistic number.",
              "- `test_ood_domain`, `hc3` and `raid` are out-of-distribution. Do not describe a high "
              "in-domain AUC as general detection ability.",
              "- The threshold came from `val` only and was frozen before any test split was scored.",
              ""]
    return "\n".join(lines)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="ml.train.evaluate_encoder")
    p.add_argument("--config", default=str(CONFIG))
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", default="E1_eval")
    p.add_argument("--max-length", type=int, default=None,
                   help="override tokeniser length (use 128 to match a smoke checkpoint)")
    args = p.parse_args(argv)

    res = run(Path(args.config), Path(args.checkpoint), args.out, args.max_length)
    print(f"device={res['device']} threshold={res['frozen_threshold']:.4f}")
    for s in res["splits"]:
        auc = "n/a" if s["roc_auc"] is None else f"{s['roc_auc']:.3f}"
        fpr = "n/a" if s["human_fpr"] is None else f"{s['human_fpr']:.3f}"
        print(f"  {s['name']:16} n={s['n']:<6} AUC={auc} FPR={fpr}")
    print(f"\nwrote {REPORTS / (args.out + '.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
