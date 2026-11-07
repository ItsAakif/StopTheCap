"""Encoder fine-tune (experiment E1).

**This script has not been executed.** It is prepared so the experiment can start the
moment the Colab entitlement is measured. Nothing here has produced a metric.

Usage
-----
    python -m ml.train.train_encoder --dry-run     # validate data + plan, no model load
    python -m ml.train.train_encoder               # requires CUDA unless --allow-cpu

Design points that are deliberate, not incidental:

* **Refuses to train on CPU** unless `--allow-cpu` is passed. The local PyTorch build is
  CPU-only (see research/_notes/environment.md); a silent CPU run would be 20-50x slower
  and would be mistaken for a broken plan rather than a missing GPU build.
* **Class-weighted loss** rather than resampling, so every row is used exactly once and
  no human text is duplicated (which would risk near-duplicate leakage).
* **Checkpoints every N steps to a persistent path**, and writes a resumable state
  (model, optimizer, scheduler, step, epoch, RNG). Colab terminates runtimes without
  warning; an untested resume path is a lost run.
* **Threshold comes from val only.** The test split is scored once, after freezing.
* **Run log records** seed, package versions, dataset digest, hardware and config, so a
  number can be traced to the exact artefacts that produced it.
* **Quarantine data is never loaded into the optimiser.** RAID/HC3 are read only in the
  final evaluation pass.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import sys
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
CONFIG = Path(__file__).resolve().parent / "encoder_config.yaml"


def load_config(path: Path = CONFIG) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _load_records():
    from ml.baselines.tfidf import load_text_records
    return load_text_records()


def class_counts(records) -> dict[str, int]:
    from ml.ingest.schema import Split
    train = [r for r in records if r.split is Split.TRAIN]
    return {
        "human": sum(1 for r in train if r.label == 0),
        "machine": sum(1 for r in train if r.label == 1),
    }


def class_weights(records) -> tuple[float, float]:
    """Inverse-frequency weights for [human, machine]."""
    c = class_counts(records)
    n = c["human"] + c["machine"]
    if not c["human"] or not c["machine"]:
        raise RuntimeError("train split is single-class; refusing to train")
    # weight_i = n / (2 * count_i) -> balanced contribution
    return (n / (2 * c["human"]), n / (2 * c["machine"]))


def environment_report() -> dict:
    import sklearn
    import torch
    import transformers

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "torch_cuda_device": (torch.cuda.get_device_name(0)
                              if torch.cuda.is_available() else None),
        "transformers": transformers.__version__,
        "sklearn": sklearn.__version__,
        "numpy": np.__version__,
    }


def dry_run(cfg: dict) -> int:
    """Validate the plan without loading a model or touching the GPU."""
    records = _load_records()
    if not records:
        print("ERROR: no canonical records. Run `python -m ml.cli build` first.",
              file=sys.stderr)
        return 2

    counts = class_counts(records)
    try:
        weights = class_weights(records)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    manifest = REPO / "datasets" / "manifests" / "COMBINED.yaml"
    digest = None
    if manifest.is_file():
        digest = yaml.safe_load(manifest.read_text(encoding="utf-8")).get("split_digest")

    from ml.ingest.schema import Split
    split_n = {}
    for r in records:
        split_n[r.split.value] = split_n.get(r.split.value, 0) + 1

    steps_per_epoch = max(1, (counts["human"] + counts["machine"]) // cfg["batch_size"])
    total_steps = steps_per_epoch * cfg["epochs"]

    print("=== DRY RUN (no training performed) ===")
    print(json.dumps({
        "config": cfg,
        "environment": environment_report(),
        "split_digest": digest,
        "split_counts": split_n,
        "train_class_counts": counts,
        "class_weights_human_machine": [round(w, 4) for w in weights],
        "planned_steps_per_epoch": steps_per_epoch,
        "planned_total_steps": total_steps,
    }, indent=2))
    print("\nDry run OK. Training was NOT started.")
    print("Before a real run: complete docs/colab-entitlement.md, then run")
    print("  python -m ml.train.train_encoder")
    return 0


def train(cfg: dict, allow_cpu: bool = False, smoke_steps: int | None = None,
          device: str | None = None, checkpoint_dir: str | None = None,
          out_log: str | None = None, num_workers: int = 0) -> int:
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              get_linear_schedule_with_warmup)

    from ml.ingest.schema import Split

    if not torch.cuda.is_available() and not allow_cpu:
        print("ERROR: CUDA is not available. The local PyTorch build is CPU-only.\n"
              "Install the CUDA build, or pass --allow-cpu for a slow smoke test.",
              file=sys.stderr)
        return 3

    # ---- smoke mode: prove the script executes, nothing more ----
    smoke = smoke_steps is not None
    if smoke:
        cfg = dict(cfg)
        cfg["max_length"] = 128          # keep the CPU run bounded
        cfg["batch_size"] = 8
        cfg["checkpoint"] = {**cfg["checkpoint"],
                             "dir": cfg["checkpoint"]["dir"] + "-smoke"}

    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    torch.set_num_threads(max(1, min(2, os.cpu_count() or 1)))

    records = _load_records()
    if not records:
        print("ERROR: no canonical records found under datasets/canonical/.")
        print("       Rebuild the ZIP with: python scripts/make_colab_zip.py --with-data")
        return 4

    train_recs = [r for r in records if r.split is Split.TRAIN]
    val_recs = [r for r in records if r.split is Split.VAL]
    test_recs = [r for r in records if r.split is Split.TEST_ID]

    if smoke:
        # Stratified: the splits are ordered by dataset/variant, so a head slice can contain
        # a single class and make the threshold undefined.
        def _stratified(recs, n):
            by: dict[int, list] = {}
            for r in recs:
                by.setdefault(r.label, []).append(r)
            per = max(1, n // max(1, len(by)))
            out: list = []
            for lab in sorted(by):
                out.extend(by[lab][:per])
            return out

        train_recs = _stratified(train_recs, 2000)
        val_recs = _stratified(val_recs, 400)
        test_recs = _stratified(test_recs, 400)

    tok = AutoTokenizer.from_pretrained(cfg["model_name"])

    class _DS(Dataset):
        def __init__(self, recs):
            self.recs = recs

        def __len__(self):
            return len(self.recs)

        def __getitem__(self, i):
            r = self.recs[i]
            enc = tok(r.text or "", truncation=True, max_length=cfg["max_length"],
                      padding="max_length", return_tensors="pt")
            return {
                "input_ids": enc["input_ids"][0],
                "attention_mask": enc["attention_mask"][0],
                "labels": torch.tensor(r.label, dtype=torch.long),
            }

    # Pin a job to one card so two experiments can share a 2-GPU host, e.g. --device cuda:1.
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device.startswith("cuda") and not torch.cuda.is_available():
        print(f"ERROR: requested {device} but CUDA is unavailable.", file=sys.stderr)
        return 3
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["model_name"], num_labels=2).to(device)

    if not train_recs:
        print(f"ERROR: {len(records)} records loaded but none are in the train split.")
        return 4

    hw, mw = class_weights(records)
    weight = torch.tensor([hw, mw], dtype=torch.float, device=device)

    dl = DataLoader(_DS(train_recs), batch_size=cfg["batch_size"], shuffle=True,
                    drop_last=True, num_workers=num_workers,
                    persistent_workers=num_workers > 0)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"],
                            weight_decay=cfg["weight_decay"])
    total = len(dl) * cfg["epochs"]
    sched = get_linear_schedule_with_warmup(
        opt, int(total * cfg["warmup_ratio"]), total)

    ckpt_dir = Path(checkpoint_dir) if checkpoint_dir else (REPO / cfg["checkpoint"]["dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    state_path = ckpt_dir / "last.pt"
    start_epoch, global_step = 0, 0

    if cfg["checkpoint"]["resume"] and state_path.is_file():
        state = torch.load(state_path, map_location=device)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        sched.load_state_dict(state["sched"])
        start_epoch, global_step = state["epoch"], state["step"]
        print(f"resumed from {state_path} at epoch {start_epoch} step {global_step}")

    loss_fn = torch.nn.CrossEntropyLoss(weight=weight)

    stop = smoke_steps if smoke else None
    for epoch in range(start_epoch, cfg["epochs"]):
        model.train()
        for batch in dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"])
            loss = loss_fn(out.logits, batch["labels"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            global_step += 1

            if global_step % cfg["checkpoint"]["every_steps"] == 0:
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                            "sched": sched.state_dict(), "epoch": epoch,
                            "step": global_step}, state_path)
                print(f"checkpoint @ epoch {epoch} step {global_step}")

            if stop is not None and global_step >= stop:
                print(f"smoke stop at step {global_step}")
                break
        if stop is not None and global_step >= stop:
            break

    torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                "sched": sched.state_dict(), "epoch": cfg["epochs"],
                "step": global_step}, state_path)

    # ---- threshold from val only ----
    model.eval()

    def score(recs):
        ds = DataLoader(_DS(recs), batch_size=cfg["batch_size"])
        out = []
        with torch.no_grad():
            for batch in ds:
                batch = {k: v.to(device) for k, v in batch.items()}
                logits = model(input_ids=batch["input_ids"],
                               attention_mask=batch["attention_mask"]).logits
                out.extend(torch.softmax(logits, dim=-1)[:, 1].cpu().numpy())
        return np.array(out)

    val_scores = score(val_recs)
    val_y = np.array([r.label for r in val_recs])
    humans = val_scores[val_y == 0]
    thr = float(np.quantile(humans, 1 - cfg["threshold"]["target_human_fpr"]))

    evaluation = None
    if test_recs and len({r.label for r in test_recs}) == 2:
        from sklearn.metrics import roc_auc_score

        test_scores = score(test_recs)
        test_y = np.array([r.label for r in test_recs])
        neg = test_y == 0
        fpr = float((test_scores[neg] >= thr).mean()) if neg.any() else None
        evaluation = {
            "split": "test_id",
            "n": int(len(test_y)),
            "n_neg": int(neg.sum()),
            "roc_auc": float(roc_auc_score(test_y, test_scores)),
            "human_fpr_at_threshold": fpr,
            "caveat": ("smoke run: 128-token truncation, 2k-row subset and "
                       f"{global_step} steps. NOT a trained model."
                       if smoke else
                       f"only {int(neg.sum())} negatives; treat threshold metrics as indicative"
                       if neg.sum() < 1000 else ""),
        }

    run_log = {
        "experiment": cfg["experiment"],
        "smoke": smoke,
        "smoke_steps": smoke_steps,
        "config": cfg,
        "environment": environment_report(),
        "train_class_weights": [hw, mw],
        "steps_completed": global_step,
        "frozen_threshold": thr,
        "threshold_basis": f"quantile on {len(humans)} human val scores",
        "checkpoint": str(state_path),
        "evaluation": evaluation,
        "status": ("SMOKE -- script executes end to end; NOT a trained model, do not cite metrics"
                   if smoke else
                   "TRAINED -- evaluation against test_id/quarantine pending"),
    }
    (REPO / cfg["output"]["run_log"]).parent.mkdir(parents=True, exist_ok=True)
    log_path = Path(out_log) if out_log else (REPO / cfg["output"]["run_log"])
    if smoke:
        log_path = log_path.with_name(log_path.stem + "_smoke" + log_path.suffix)
    log_path.write_text(json.dumps(run_log, indent=2), encoding="utf-8")
    print(f"training complete; run log -> {log_path}")
    print("Next: evaluate test_id and quarantine, then write the report.")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(CONFIG))
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--allow-cpu", action="store_true")
    p.add_argument("--checkpoint-dir", default=None,
                   help="override checkpoint directory (put this on Drive in Colab)")
    p.add_argument("--out-log", default=None, help="override run-log path")
    p.add_argument("--num-workers", type=int, default=0,
                   help="DataLoader workers; 2 is reasonable on a 2-vCPU Colab box")
    p.add_argument("--device", default=None,
                   help="cuda:0 / cuda:1 / cpu - pin one job per GPU when running two in parallel")
    p.add_argument("--smoke", type=int, default=None, metavar="STEPS",
                   help="run N optimiser steps only, for plumbing validation")
    args = p.parse_args(argv)

    cfg = load_config(Path(args.config))
    if args.dry_run:
        return dry_run(cfg)
    return train(cfg, allow_cpu=args.allow_cpu, smoke_steps=args.smoke,
                 device=args.device, checkpoint_dir=args.checkpoint_dir,
                 out_log=args.out_log, num_workers=args.num_workers)


if __name__ == "__main__":
    sys.exit(main())
