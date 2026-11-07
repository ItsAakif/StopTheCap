"""Calibration, abstention and threshold-reliability analysis for E1.

Everything is fitted on `val` only. The external splits (test_id, test_ood_domain, hc3, raid) are
scored and reported, never used to fit a calibration parameter or choose an abstention policy.

Pipeline:
    python -m ml.train.calibrate score    # cache per-record scores (resumable)
    python -m ml.train.calibrate analyse  # fit temperature, compute all tables

Why a prediction cache: scoring 18k records on CPU takes over an hour. Caching means the analysis
can be re-run and extended without re-scoring, and it makes every reported number reproducible from
a committed JSONL.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
CFG = Path(__file__).resolve().parent / "encoder_config.yaml"
PRED = REPO / "reports" / "e1" / "predictions.jsonl"
OUT_JSON = REPO / "reports" / "e1" / "calibration.json"
OUT_MD = REPO / "reports" / "e1" / "calibration.md"

#: Big splits are strided down so the analysis finishes in reasonable time. Every downstream table
#: states the sample size so a capped split cannot be mistaken for a full one.
CAPS = {"test_ood_domain": 2000, "hc3": 2000}
SPLIT_ORDER = ["val", "test_id", "test_ood_domain", "hc3", "raid"]


# --------------------------------------------------------------------------- scoring


def _split_records():
    from ml.ingest.schema import Split
    from ml.baselines.tfidf import load_text_records

    wanted = {
        "val": Split.VAL,
        "test_id": Split.TEST_ID,
        "test_ood_domain": Split.TEST_OOD_DOMAIN,
    }
    recs = load_text_records()
    out = {k: [r for r in recs if r.split is v] for k, v in wanted.items()}
    out["hc3"] = [r for r in recs if r.source_dataset == "hc3"]
    out["raid"] = [r for r in recs if r.source_dataset == "raid"]
    return out


def _stride(recs, cap):
    if cap is None or len(recs) <= cap:
        return recs
    step = len(recs) / cap
    return [recs[int(i * step)] for i in range(cap)]


def cmd_score(args) -> int:
    import torch
    from ml.train.evaluate_encoder import load_model

    cfg = yaml.safe_load(CFG.read_text(encoding="utf-8"))
    torch.set_num_threads(args.threads)
    model, tok, device = load_model(cfg, Path(args.checkpoint))
    all_recs = _split_records()

    PRED.parent.mkdir(parents=True, exist_ok=True)
    done: dict[str, int] = {}
    if PRED.is_file():
        for line in PRED.read_text(encoding="utf-8").splitlines():
            if line.strip():
                d = json.loads(line)
                done[d["split"]] = done.get(d["split"], 0) + 1

    for split in SPLIT_ORDER:
        recs = _stride(all_recs[split], CAPS.get(split))
        if done.get(split) == len(recs):
            print(f"  {split:16} already scored ({len(recs)})", flush=True)
            continue
        t0 = time.perf_counter()
        with PRED.open("a", encoding="utf-8") as fh, torch.inference_mode():
            for i in range(0, len(recs), args.batch):
                batch = recs[i:i + args.batch]
                enc = tok([r.text or "" for r in batch], truncation=True, padding=True,
                          max_length=cfg["max_length"], return_tensors="pt").to(device)
                logits = model(**enc).logits
                probs = torch.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy()
                for r, p in zip(batch, probs):
                    fh.write(json.dumps({"split": split, "dataset": r.source_dataset,
                                         "label": int(r.label), "score": float(p)}) + "\n")
                fh.flush()
                if (i // args.batch) % 20 == 0:
                    print(f"  {split:16} {i + len(batch)}/{len(recs)}", flush=True)
        dt = time.perf_counter() - t0
        print(f"  {split:16} scored {len(recs)} in {dt/60:.1f} min", flush=True)
    print("scoring complete")
    return 0


# --------------------------------------------------------------------------- metrics


def brier(y, p) -> float:
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def ece(y, p, bins: int = 10) -> float:
    y = np.asarray(y); p = np.asarray(p)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        m = (p >= lo) & (p < hi if i < bins - 1 else p <= hi)
        if m.any():
            total += abs(p[m].mean() - y[m].mean()) * m.sum() / len(y)
    return float(total)


def reliability(y, p, bins: int = 10) -> list[dict]:
    y = np.asarray(y); p = np.asarray(p)
    edges = np.linspace(0, 1, bins + 1)
    out = []
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        m = (p >= lo) & (p < hi if i < bins - 1 else p <= hi)
        out.append({"lo": round(float(lo), 2), "hi": round(float(hi), 2), "n": int(m.sum()),
                    "mean_pred": round(float(p[m].mean()), 4) if m.any() else None,
                    "frac_pos": round(float(y[m].mean()), 4) if m.any() else None})
    return out


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def fit_temperature(scores: np.ndarray, labels: np.ndarray) -> dict:
    """Single-parameter temperature scaling fitted by minimising NLL on `val` only."""
    import torch

    z = torch.tensor(np.clip(scores, 1e-7, 1 - 1e-7), dtype=torch.float64)
    logit = torch.log(z) - torch.log1p(-z)          # logit of the sigmoid output
    y = torch.tensor(labels, dtype=torch.float64)
    logT = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([logT], lr=0.1, max_iter=200, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        p = torch.sigmoid(logit / torch.exp(logT))
        loss = torch.nn.functional.binary_cross_entropy(p, y)
        loss.backward()
        return loss

    opt.step(closure)
    T = float(torch.exp(logT).item())
    with torch.no_grad():
        p = torch.sigmoid(logit / T).numpy()
    nll = float(-np.mean(labels * np.log(np.clip(p, 1e-12, 1)) +
                         (1 - labels) * np.log(np.clip(1 - p, 1e-12, 1))))
    return {"temperature": round(T, 6), "nll": round(nll, 6),
            "n_samples": int(len(labels)),
            "n_human": int((labels == 0).sum()), "n_machine": int((labels == 1).sum())}


def apply_temperature(scores: np.ndarray, T: float) -> np.ndarray:
    z = np.clip(scores, 1e-7, 1 - 1e-7)
    logit = np.log(z) - np.log1p(-z)
    return 1.0 / (1.0 + np.exp(-logit / T))


def at_threshold(y, p, thr) -> dict:
    y = np.asarray(y); p = np.asarray(p)
    pred = (p >= thr).astype(int)
    neg = y == 0
    pos = y == 1
    fp = int(((pred == 1) & neg).sum())
    fn = int(((pred == 0) & pos).sum())
    n_neg = int(neg.sum()); n_pos = int(pos.sum())
    ci = wilson(fp, n_neg) if n_neg else (0.0, 0.0)
    return {"threshold": float(thr), "n": int(len(y)), "n_neg": n_neg, "n_pos": n_pos,
            "false_positives": fp, "false_negatives": fn,
            "fpr": fp / n_neg if n_neg else None, "fpr_ci": list(ci),
            "fnr": fn / n_pos if n_pos else None,
            "accuracy": float((pred == y).mean())}


def risk_coverage(y, p, thr, margins) -> list[dict]:
    """Selective classification: accept when the calibrated score is far enough from the threshold."""
    y = np.asarray(y); p = np.asarray(p)
    pred = (p >= thr).astype(int)
    correct = (pred == y)
    dist = np.abs(p - thr)
    out = []
    for m in margins:
        acc = dist >= m
        n_acc = int(acc.sum())
        if n_acc == 0:
            out.append({"margin": float(m), "coverage": 0.0, "n_accepted": 0})
            continue
        y_acc, pred_acc = y[acc], pred[acc]
        neg = y_acc == 0
        fp = int(((pred_acc == 1) & neg).sum())
        fn = int(((pred_acc == 0) & (y_acc == 1)).sum())
        out.append({
            "margin": round(float(m), 6),
            "coverage": round(n_acc / len(y), 4),
            "n_accepted": n_acc,
            "selective_risk": round(float((pred_acc != y_acc).mean()), 4),
            "fpr_accepted_humans": round(fp / int(neg.sum()), 4) if neg.any() else None,
            "fnr_accepted_ai": round(fn / int((y_acc == 1).sum()), 4) if (y_acc == 1).any() else None,
        })
    return out


def margin_for_coverage(y, p, thr, target_cov) -> float:
    """Pick the margin on `val` that achieves a target coverage."""
    dist = np.abs(np.asarray(p) - thr)
    q = np.quantile(dist, 1 - target_cov)
    return float(q)

# --------------------------------------------------------------------------- analysis


def load_predictions(path: Path = PRED) -> dict[str, dict]:
    rows: dict[str, list] = {k: [] for k in SPLIT_ORDER}
    bad = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            bad += 1          # a trailing partial line while the scorer is still writing
            continue
        rows.setdefault(d["split"], []).append(d)
    if bad:
        print(f"  note: skipped {bad} malformed line(s) in {path.name}")
    out = {}
    for split, items in rows.items():
        if not items:
            continue
        out[split] = {
            "y": np.array([i["label"] for i in items]),
            "p": np.array([i["score"] for i in items]),
            "dataset": items,
        }
    return out


def threshold_reliability(val_human_scores: np.ndarray, target_fpr: float = 0.05) -> dict:
    """How much does the frozen threshold depend on individual human examples?"""
    s = np.sort(np.asarray(val_human_scores))          # ascending
    n = len(s)
    thr = float(np.quantile(s, 1 - target_fpr))
    fp = int((s >= thr).sum())
    lo, hi = wilson(fp, n)

    # Jackknife: drop one human at a time, recompute the same quantile.
    jack = []
    for i in range(n):
        rest = np.delete(s, i)
        jack.append(float(np.quantile(rest, 1 - target_fpr)))
    jack = np.array(jack)

    # Which examples actually determine the cut? The quantile lands between the k-th and
    # (k+1)-th largest scores, where k = floor(n * target_fpr).
    k = int(np.floor(n * target_fpr))
    top = s[::-1]
    return {
        "target_fpr": target_fpr,
        "n_human_val": n,
        "threshold": thr,
        "humans_at_or_above_threshold": fp,
        "fpr": fp / n,
        "fpr_ci": list(wilson(fp, n)),
        "expected_count_at_5pct": round(n * target_fpr, 2),
        "determining_rank_k": k,
        "score_of_kth_largest": float(top[k - 1]) if k >= 1 else None,
        "score_of_k1th_largest": float(top[k]) if k < n else None,
        "gap_between_determining_scores": (float(top[k - 1] - top[k]) if 0 < k < n else None),
        "jackknife_min": float(jack.min()),
        "jackknife_max": float(jack.max()),
        "jackknife_std": float(jack.std()),
        "jackknife_max_abs_shift": float(np.abs(jack - thr).max()),
        "jackknife_relative_shift_pct": float(np.abs(jack - thr).max() / thr * 100),
        "needs_more_humans": bool(n < 1000),
    }


def cmd_analyse(args) -> int:
    preds = load_predictions(Path(args.predictions) if args.predictions else PRED)
    if "val" not in preds:
        print("no val predictions - run the score step first", file=sys.stderr)
        return 2

    run_log = REPO / "reports" / "e1" / "E1.json"
    frozen = None
    if run_log.is_file():
        frozen = json.loads(run_log.read_text(encoding="utf-8")).get("frozen_threshold")

    val = preds["val"]
    cal = fit_temperature(val["p"], val["y"])
    T = cal["temperature"]

    raw_thr = float(np.quantile(val["p"][val["y"] == 0], 1 - args.target_fpr))
    val_cal = apply_temperature(val["p"], T)
    cal_thr = float(np.quantile(val_cal[val["y"] == 0], 1 - args.target_fpr))

    report = {
        "temperature": cal,
        "frozen_threshold_original": frozen,
        "threshold_recomputed_raw": raw_thr,
        "threshold_matches_run_log": (frozen is not None and abs(raw_thr - frozen) < 1e-6),
        "calibrated_threshold": cal_thr,
        "target_fpr": args.target_fpr,
        "splits": {},
        "abstention": {},
        "threshold_reliability": threshold_reliability(val["p"][val["y"] == 0], args.target_fpr),
        "sample_sizes": {k: int(len(v["y"])) for k, v in preds.items()},
    }

    for split, data in preds.items():
        y, p = data["y"], data["p"]
        pc = apply_temperature(p, T)
        report["splits"][split] = {
            "n": int(len(y)), "n_neg": int((y == 0).sum()), "n_pos": int((y == 1).sum()),
            "raw": {"brier": brier(y, p), "ece": ece(y, p),
                    "at_threshold": at_threshold(y, p, raw_thr)},
            "calibrated": {"brier": brier(y, pc), "ece": ece(y, pc),
                           "at_threshold": at_threshold(y, pc, cal_thr),
                           "reliability": reliability(y, pc)},
            "raw_reliability": reliability(y, p),
        }

    # Abstention policy chosen on val, then applied unchanged to every other split.
    levels = [1.0, 0.9, 0.75, 0.5, 0.25]
    margins = {str(c): margin_for_coverage(val["y"], val_cal, cal_thr, c) for c in levels}
    report["abstention"]["margins_from_val"] = margins
    report["abstention"]["levels"] = {}
    for c in levels:
        m = margins[str(c)]
        report["abstention"]["levels"][str(c)] = {
            split: next((r for r in risk_coverage(d["y"], apply_temperature(d["p"], T),
                                                  cal_thr, [m])), None)
            for split, d in preds.items()
        }

    OUT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    OUT_MD.write_text(render_markdown(report), encoding="utf-8")
    print(f"temperature = {T:.4f}  (fitted on {cal['n_samples']} val rows)")
    print(f"raw threshold {raw_thr:.4f} | calibrated threshold {cal_thr:.4f}")
    for split, s in report["splits"].items():
        print(f"  {split:16} ECE {s['raw']['ece']:.3f} -> {s['calibrated']['ece']:.3f} | "
              f"Brier {s['raw']['brier']:.4f} -> {s['calibrated']['brier']:.4f} | "
              f"FPR {s['calibrated']['at_threshold']['fpr']:.3f}")
    print(f"\nwrote {OUT_JSON}")
    return 0


def render_markdown(r: dict) -> str:
    L = ["# E1 calibration and abstention", "",
         f"Temperature fitted on **val only** ({r['temperature']['n_samples']} rows, "
         f"{r['temperature']['n_human']} human / {r['temperature']['n_machine']} machine): "
         f"**T = {r['temperature']['temperature']}**", "",
         f"- frozen threshold (E1, unchanged): **{r['frozen_threshold_original']}**",
         f"- recomputed from raw val scores: {r['threshold_recomputed_raw']:.6f} "
         f"(matches run log: {r['threshold_matches_run_log']})",
         f"- calibrated threshold: **{r['calibrated_threshold']:.6f}**",
         f"- target human FPR: {r['target_fpr']:.0%}", "",
         "## Calibration effect", "",
         "| Split | n | ECE raw | ECE calibrated | Brier raw | Brier calibrated |",
         "|---|---:|---:|---:|---:|---:|"]
    for s, d in r["splits"].items():
        L.append(f"| {s} | {d['n']} | {d['raw']['ece']:.3f} | {d['calibrated']['ece']:.3f} | "
                 f"{d['raw']['brier']:.4f} | {d['calibrated']['brier']:.4f} |")

    L += ["", "## Human FPR at threshold", "",
          "| Split | negatives | FPR (raw thr) | FPR (calibrated thr) [95% CI] |",
          "|---|---:|---|---|"]
    for s, d in r["splits"].items():
        raw = d["raw"]["at_threshold"]; c = d["calibrated"]["at_threshold"]
        ci = c["fpr_ci"]
        L.append(f"| {s} | {raw['n_neg']} | {raw['fpr']:.3f} | {c['fpr']:.3f} "
                 f"[{ci[0]:.3f}, {ci[1]:.3f}] |")

    L += ["", "## Abstention (policy chosen on val, applied unchanged elsewhere)", ""]
    for level, per_split in r["abstention"]["levels"].items():
        m = r["abstention"]["margins_from_val"].get(str(level))
        L.append(f"### Target coverage {float(level):.0%} (margin {m:.4f})" if m is not None
                 else f"### Target coverage {level}")
        L.append("")
        L.append("| Split | actual coverage | selective risk | FPR among accepted humans | FNR among accepted AI |")
        L.append("|---|---:|---:|---:|---:|")
        for s, d in per_split.items():
            if not d or d.get("n_accepted", 0) == 0:
                L.append(f"| {s} | {0.0} | n/a | n/a | n/a |")
                continue
            fpr = "n/a" if d["fpr_accepted_humans"] is None else f"{d['fpr_accepted_humans']:.3f}"
            fnr = "n/a" if d["fnr_accepted_ai"] is None else f"{d['fnr_accepted_ai']:.3f}"
            L.append(f"| {s} | {d['coverage']:.1%} | {d['selective_risk']:.3f} | {fpr} | {fnr} |")
        L.append("")

    tr = r["threshold_reliability"]
    L += ["## Threshold reliability", "",
          f"- human validation examples: **{tr['n_human_val']}**",
          f"- flagged at the threshold: **{tr['humans_at_or_above_threshold']}** "
          f"(FPR {tr['fpr']:.3f}, Wilson 95% CI [{tr['fpr_ci'][0]:.3f}, {tr['fpr_ci'][1]:.3f}])",
          f"- at a 5% budget we expect {tr['expected_count_at_5pct']:.1f} flags; "
          f"the cut is decided by examples ranked {tr['determining_rank_k']} and "
          f"{tr['determining_rank_k'] + 1} of 202",
          f"- score gap between those two examples: {tr['gap_between_determining_scores']}",
          f"- leave-one-out: threshold moves by at most **{tr['jackknife_max_abs_shift']:.6f}** "
          f"({tr['jackknife_relative_shift_pct']:.1f}% of the threshold), "
          f"sd {tr['jackknife_std']:.6f}",
          f"- more human validation data needed: **{tr['needs_more_humans']}**", ""]
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="ml.train.calibrate")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("--checkpoint", required=True)
    s.add_argument("--batch", type=int, default=32)
    s.add_argument("--threads", type=int, default=16)
    a = sub.add_parser("analyse")
    a.add_argument("--predictions", default=None)
    a.add_argument("--target-fpr", type=float, default=0.05)
    args = ap.parse_args(argv)
    return cmd_score(args) if args.cmd == "score" else cmd_analyse(args)


if __name__ == "__main__":
    sys.exit(main())
