"""Measure parallel throughput across multiple GPUs, without any distributed machinery.

Why not DDP: our encoder is 66M parameters at batch 16 / 512 tokens - it fits on one T4 with
several GB to spare. Data-parallel training would add NCCL, sampler and rank-0 plumbing, and on a
Kaggle box with only 2 vCPUs the dataloader becomes the bottleneck before the second GPU does any
work.

What actually uses two GPUs well here is **independent parallelism**: one job per card.
- Concurrent experiments (e.g. distilbert on GPU 0, bert-base on GPU 1).
- Two Wan generation workers, one per card, which halves wall clock for the video pilot.

This module measures whether that parallelism delivers, by running the same single-GPU benchmark
as separate processes pinned to each device and comparing summed throughput against one device.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]


def run_one(device: int, model: str, steps: int, out: Path) -> dict[str, Any] | None:
    """Run the probe benchmark pinned to a single device via CUDA_VISIBLE_DEVICES."""
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(device)
    cmd = [sys.executable, "-m", "ml.cloud.probe",
           "--out", str(out), "--benchmark-model", model, "--steps", str(steps)]
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=str(REPO))
    if not out.is_file():
        return {"device": device, "error": (proc.stderr or proc.stdout or "")[-800:]}
    data = json.loads(out.read_text(encoding="utf-8"))
    bench = data.get("benchmark") or {}
    devices = data.get("gpu", {}).get("devices") or []
    return {
        "device": device,
        "gpu_name": devices[0]["name"] if devices else None,
        "examples_per_s": bench.get("examples_per_s"),
        "steps_per_s": bench.get("steps_per_s"),
        "peak_vram_gb": bench.get("peak_vram_gb"),
        "cuda_available": data.get("gpu", {}).get("cuda_available"),
    }


def scaling_summary(results: list[dict], n_devices: int) -> dict[str, Any]:
    """Compare summed parallel throughput against a single device."""
    ok = [r for r in results if r.get("examples_per_s")]
    if not ok:
        return {"usable": False, "reason": "no device produced a throughput figure"}
    baseline = ok[0]["examples_per_s"]
    total = sum(r["examples_per_s"] for r in ok)
    return {
        "usable": True,
        "devices_measured": len(ok),
        "devices_requested": n_devices,
        "single_device_examples_per_s": round(baseline, 2),
        "aggregate_examples_per_s": round(total, 2),
        "speedup": round(total / baseline, 2) if baseline else None,
        "efficiency": round((total / baseline) / len(ok), 3) if baseline else None,
        "peak_vram_gb_per_device": [r.get("peak_vram_gb") for r in ok],
    }


def measure(devices: list[int], model: str, steps: int) -> dict[str, Any]:
    tmp = Path(tempfile.mkdtemp(prefix="stc_multigpu_"))
    results = []
    for d in devices:
        results.append(run_one(d, model, steps, tmp / f"dev{d}.json") or {"device": d})

    # Baseline: the first device alone, to compare against the sum.
    solo = run_one(devices[0], model, steps, tmp / "solo.json") or {}
    summary = scaling_summary(results, len(devices))
    if solo.get("examples_per_s") and summary.get("usable"):
        summary["single_device_examples_per_s"] = round(solo["examples_per_s"], 2)
        summary["speedup"] = round(summary["aggregate_examples_per_s"] / solo["examples_per_s"], 2)
        summary["efficiency"] = round(summary["speedup"] / len(devices), 3)

    return {"model": model, "steps": steps, "per_device": results,
            "solo_baseline": solo, "summary": summary}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="ml.cloud.multigpu_bench")
    ap.add_argument("--devices", default="0,1")
    ap.add_argument("--model", default="distilbert-base-cased")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--out", default="reports/cloud/multigpu_measure.json")
    args = ap.parse_args(argv)

    devices = [int(x) for x in args.devices.split(",") if x.strip() != ""]
    result = measure(devices, args.model, args.steps)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(json.dumps(result["summary"], indent=2))
    for r in result["per_device"]:
        print(f"  device {r['device']}: {r.get('gpu_name')} "
              f"{r.get('examples_per_s')} ex/s peak {r.get('peak_vram_gb')} GB")
    print(f"\nwrote {out}")
    print("Interpretation: speedup ~= number of cards means real parallelism; "
          "speedup ~1.0 means the workload is already CPU-bound.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
