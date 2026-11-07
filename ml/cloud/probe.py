"""Cloud environment probe: measure what the account actually gives you.

Run in Colab or Kaggle. Writes machine-readable JSON so results can be committed rather than
transcribed by hand.

Nothing here assumes a GPU, a quota or a session limit. Every value is measured at runtime or
reported as null with a reason. A missing measurement is never filled with a guessed number.

Usage:
    python -m ml.cloud.probe --out reports/cloud/colab_measure.json
    python -m ml.cloud.probe --benchmark-model distilbert-base-cased --steps 30
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]


def _run(cmd: list[str], timeout: int = 20) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
        return (out.stdout or "").strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def detect_platform() -> dict[str, Any]:
    env = os.environ
    host = "unknown"
    if env.get("COLAB_RELEASE_TAG") or (Path("/content").is_dir() and env.get("COLAB_GPU") is not None):
        host = "colab"
    elif env.get("KAGGLE_KERNEL_RUN_TYPE") or Path("/kaggle").is_dir():
        host = "kaggle"
    return {
        "host": host,
        "colab_release": env.get("COLAB_RELEASE_TAG"),
        "colab_gpu_env": env.get("COLAB_GPU"),
        "kaggle_run_type": env.get("KAGGLE_KERNEL_RUN_TYPE"),
        "note": "host detection is heuristic; trust the GPU fields over this label",
    }


def gpu_info() -> dict[str, Any]:
    import torch

    info: dict[str, Any] = {
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": getattr(torch.version, "cuda", None),
        "device_count": 0,
        "devices": [],
    }
    if not torch.cuda.is_available():
        info["reason"] = "torch.cuda.is_available() is False - CPU-only build or no GPU assigned"
        return info

    info["device_count"] = torch.cuda.device_count()
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        info["devices"].append({
            "index": i,
            "name": props.name,
            "total_vram_gb": round(props.total_memory / 1e9, 2),
            "major": props.major,
            "minor": props.minor,
            "multi_processor_count": props.multi_processor_count,
        })

    smi = _run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader"])
    info["nvidia_smi"] = smi.splitlines() if smi else None
    return info


def cpu_info() -> dict[str, Any]:
    import torch

    return {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "cpu_count": os.cpu_count(),
        "torch_threads": torch.get_num_threads(),
    }


def memory_info() -> dict[str, Any]:
    out: dict[str, Any] = {"total_gb": None, "available_gb": None, "source": None}
    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    out["total_gb"] = round(int(line.split()[1]) / 1e6, 2)
                elif line.startswith("MemAvailable:"):
                    out["available_gb"] = round(int(line.split()[1]) / 1e6, 2)
            out["source"] = "/proc/meminfo"
    except OSError:
        out["source"] = "unavailable (non-Linux host)"
    return out


def disk_info() -> dict[str, Any]:
    paths = {
        "cwd": Path.cwd(),
        "root": Path("/"),
        "content": Path("/content"),
        "kaggle_working": Path("/kaggle/working"),
        "drive": Path("/content/drive/MyDrive"),
    }
    out: dict[str, Any] = {}
    for name, p in paths.items():
        try:
            if not p.exists():
                continue
            usage = shutil.disk_usage(p)
            out[name] = {
                "path": str(p),
                "total_gb": round(usage.total / 1e9, 1),
                "free_gb": round(usage.free / 1e9, 1),
            }
        except OSError:
            continue
    return out


def benchmark(model_name: str = "distilbert-base-cased", steps: int = 30,
              batch_size: int = 8, seq_len: int = 128) -> dict[str, Any]:
    """Time real optimiser steps. Returns throughput and peak VRAM."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2).to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=5e-5)

    ids = torch.randint(0, tok.vocab_size, (batch_size, seq_len), device=device)
    mask = torch.ones_like(ids)
    labels = torch.randint(0, 2, (batch_size,), device=device)

    def one_step() -> None:
        opt.zero_grad(set_to_none=True)
        out = model(input_ids=ids, attention_mask=mask, labels=labels)
        out.loss.backward()
        opt.step()

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    one_step()
    if device == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(steps):
        one_step()
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    peak = (torch.cuda.max_memory_allocated() / 1e9) if device == "cuda" else None
    return {
        "model": model_name,
        "device": device,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "steps": steps,
        "seconds": round(elapsed, 3),
        "steps_per_s": round(steps / elapsed, 4) if elapsed else None,
        "examples_per_s": round(steps * batch_size / elapsed, 2) if elapsed else None,
        "peak_vram_gb": round(peak, 3) if peak is not None else None,
        "peak_vram_note": None if peak is not None else "CPU run: VRAM not applicable",
    }


def checkpoint_persistence(checkpoint_dir: Path) -> dict[str, Any]:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    probe = checkpoint_dir / "probe.txt"
    payload = f"probe {time.time_ns()}"
    probe.write_text(payload, encoding="utf-8")
    readback = probe.read_text(encoding="utf-8")
    return {
        "dir": str(checkpoint_dir),
        "writable": True,
        "readback_matches": readback == payload,
        "sha256_prefix": hashlib.sha256(readback.encode()).hexdigest()[:16],
        "survives_disconnect": None,
        "survives_disconnect_note": (
            "Re-run this cell after Runtime -> Disconnect to fill this in. "
            "A file present after disconnect means persistent storage works."
        ),
    }


def quota_hints() -> dict[str, Any]:
    return {
        "gpu_hours_remaining": None,
        "reason": "not exposed to the runtime; read it in the account UI",
        "colab_where": "Runtime -> View compute units / purchase panel",
        "kaggle_where": "Settings -> accelerator usage, or the notebook GPU quota indicator",
        "session_timeout_minutes": None,
        "session_timeout_note": "not exposed programmatically; record from the UI or observe",
    }


def run_probe(out: Path, benchmark_model: str | None = None, steps: int = 30,
              checkpoint_dir: Path | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "collected_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "platform": detect_platform(),
        "cpu": cpu_info(),
        "memory": memory_info(),
        "gpu": gpu_info(),
        "disk": disk_info(),
        "quota": quota_hints(),
    }
    if checkpoint_dir:
        result["checkpoint"] = checkpoint_persistence(checkpoint_dir)
    if benchmark_model:
        try:
            result["benchmark"] = benchmark(benchmark_model, steps=steps)
        except Exception as exc:  # probe must not crash the notebook
            result["benchmark"] = {"error": f"{type(exc).__name__}: {exc}"}

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="ml.cloud.probe")
    p.add_argument("--out", default="reports/cloud/probe.json")
    p.add_argument("--benchmark-model", default=None,
                   help="e.g. distilbert-base-cased; omit to skip the throughput run")
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--checkpoint-dir", default=None)
    args = p.parse_args(argv)

    res = run_probe(Path(args.out),
                    benchmark_model=args.benchmark_model,
                    steps=args.steps,
                    checkpoint_dir=Path(args.checkpoint_dir) if args.checkpoint_dir else None)

    print(json.dumps({k: res[k] for k in ("collected_at_utc", "platform", "gpu")}, indent=2))
    if "benchmark" in res:
        print(json.dumps(res["benchmark"], indent=2))
    print("\nwrote " + args.out)
    print("Still to record by hand: GPU hours remaining, session timeout.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
