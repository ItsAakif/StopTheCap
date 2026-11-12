"""E2-C1 three-class generator attribution (distilbert-base-cased).

The E2-C1 corpus contains machine passages from three generators.  This trainer
discards the human rows, maps the remaining ``generator`` values onto three
classes, fine-tunes the same encoder/schedule used by E1 and E2-C1, and reports
accuracy, macro-F1, per-class precision/recall/F1, a confusion matrix and the
top confusions.

* train only on ``partition == "train"`` AI rows;
* evaluate ``val`` and ``test_holdout``;
* write ``E2C1_attribution.json``, ``E2C1_attribution.md`` and
  ``E2C1_attribution_predictions.jsonl`` under ``--out``;
* checkpoint ``last.pt`` every 200 optimizer steps and resume when present.

Heavy dependencies (torch/transformers) are imported inside :func:`train`, so the
metric helpers and ``--help`` work on a machine without torch.

Usage
-----
    python -m ml.train.train_attribution --help
    python -m ml.train.train_attribution --device cuda --out /content/drive/MyDrive/StopTheCap_reports/E2C1_attribution
    python -m ml.train.train_attribution --device cpu --smoke 5 --max-length 64
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
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

REPO = Path(__file__).resolve().parents[2]
CORPUS = REPO / "datasets" / "canonical" / "e2c1.jsonl"
MANIFEST = REPO / "datasets" / "manifests" / "e2_c1_training.json"
OUT_DEFAULT = REPO / "reports" / "E2C1_attribution"

EXPERIMENT = "E2C1_attribution"
PARTITIONS: tuple[str, ...] = ("train", "val", "test_holdout")

# The class order is part of the on-disk contract: predictions and the JSON
# report use these indices.  Keep it stable across runs.
GENERATORS: tuple[str, ...] = (
    "deepseek_v41_flash",
    "muse_spark_1_3",
    "longcat_25",
)
LABELS = GENERATORS
NUM_LABELS = len(GENERATORS)
LABEL_TO_INDEX = {name: index for index, name in enumerate(GENERATORS)}
INDEX_TO_LABEL = {index: name for index, name in enumerate(GENERATORS)}

# Same architecture/schedule as E1 (ml/train/encoder_config.yaml).  The task is
# different, not the optimisation recipe.
MODEL_NAME = "distilbert-base-cased"
MAX_LENGTH = 512
EPOCHS = 3
LEARNING_RATE = 5.0e-5
BATCH_SIZE = 16
GRAD_ACCUM = 2
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.06
FP16 = True
SEED = 42
CHECKPOINT_EVERY = 200

DEFAULT_CONFIG: dict[str, Any] = {
    "experiment": EXPERIMENT,
    "model_name": MODEL_NAME,
    "max_length": MAX_LENGTH,
    "epochs": EPOCHS,
    "learning_rate": LEARNING_RATE,
    "batch_size": BATCH_SIZE,
    "grad_accum": GRAD_ACCUM,
    "weight_decay": WEIGHT_DECAY,
    "warmup_ratio": WARMUP_RATIO,
    "fp16": FP16,
    "seed": SEED,
    "num_workers": 0,
    "checkpoint": {"every_steps": CHECKPOINT_EVERY, "resume": True},
}

REQUIRED_FIELDS: tuple[str, ...] = (
    "sample_id",
    "text",
    "label",
    "partition",
    "generator",
)


# ---------------------------------------------------------------------------
# Corpus loading and label mapping (pure, offline).
# ---------------------------------------------------------------------------


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
    """Load and validate the E2-C1 corpus (human and machine rows)."""
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


def ai_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return only machine rows whose generator is one of the three classes."""
    out: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        if int(row["label"]) != 1:
            continue
        generator = row.get("generator")
        if generator not in LABEL_TO_INDEX:
            raise ValueError(
                f"row {index}: machine row has unknown generator {generator!r}; "
                f"expected one of {GENERATORS}"
            )
        item = dict(row)
        item["label_index"] = LABEL_TO_INDEX[str(generator)]
        out.append(item)
    return out


def partition_rows(
    rows: Sequence[Mapping[str, Any]], partition: str
) -> list[dict[str, Any]]:
    if partition not in PARTITIONS:
        raise ValueError(f"unknown partition {partition!r}")
    return [dict(row) for row in rows if str(row["partition"]) == partition]


def partition_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts = Counter(str(row["partition"]) for row in rows)
    return {partition: int(counts.get(partition, 0)) for partition in PARTITIONS}


def label_index(generator: str) -> int:
    try:
        return LABEL_TO_INDEX[generator]
    except KeyError as exc:
        raise ValueError(
            f"unknown generator {generator!r}; expected one of {GENERATORS}"
        ) from exc


def label_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts = Counter(str(row["generator"]) for row in rows)
    return {name: int(counts.get(name, 0)) for name in GENERATORS}


def class_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return label_counts(rows)


def class_weights(rows: Sequence[Mapping[str, Any]]) -> list[float]:
    """Inverse-frequency weights in :data:`GENERATORS` order.

    ``total / (num_classes * count)`` keeps every class's total contribution
    equal, so the 4% LongCat class is not drowned out by Muse/DeepSeek.
    """
    counts = label_counts(rows)
    total = sum(counts.values())
    if total == 0:
        raise RuntimeError("train split is empty; refusing to train")
    missing = [name for name, count in counts.items() if count == 0]
    if missing:
        raise RuntimeError(
            f"train split has no rows for {missing}; refusing to train"
        )
    return [total / (NUM_LABELS * counts[name]) for name in GENERATORS]


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_corpus_digest(path: Path = CORPUS, expected: str | None = None) -> str:
    digest = sha256_file(path)
    if expected and digest != expected:
        raise ValueError(
            f"corpus SHA-256 mismatch: {digest} != pinned {expected} ({path})"
        )
    return digest


# ---------------------------------------------------------------------------
# Metrics (pure numpy; no torch).
# ---------------------------------------------------------------------------


def confusion_matrix(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    num_labels: int = NUM_LABELS,
) -> list[list[int]]:
    """Row = true class, column = predicted class."""
    true = [int(value) for value in y_true]
    pred = [int(value) for value in y_pred]
    if len(true) != len(pred):
        raise ValueError("y_true and y_pred must have the same length")
    matrix = [[0 for _ in range(num_labels)] for _ in range(num_labels)]
    for actual, guessed in zip(true, pred):
        if not 0 <= actual < num_labels or not 0 <= guessed < num_labels:
            raise ValueError(
                f"label out of range for {num_labels} classes: "
                f"true={actual}, pred={guessed}"
            )
        matrix[actual][guessed] += 1
    return matrix


def per_class_metrics(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    num_labels: int = NUM_LABELS,
) -> list[dict[str, Any]]:
    matrix = confusion_matrix(y_true, y_pred, num_labels=num_labels)
    out: list[dict[str, Any]] = []
    for index in range(num_labels):
        tp = matrix[index][index]
        support = sum(matrix[index])
        predicted = sum(matrix[row][index] for row in range(num_labels))
        fp = predicted - tp
        fn = support - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        out.append(
            {
                "label_index": index,
                "label": INDEX_TO_LABEL[index],
                "support": int(support),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
            }
        )
    return out


def accuracy(y_true: Iterable[int], y_pred: Iterable[int]) -> float:
    true = [int(value) for value in y_true]
    pred = [int(value) for value in y_pred]
    if len(true) != len(pred):
        raise ValueError("y_true and y_pred must have the same length")
    if not true:
        raise ValueError("cannot compute accuracy on an empty split")
    return float(sum(a == b for a, b in zip(true, pred)) / len(true))


def macro_f1(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    num_labels: int = NUM_LABELS,
) -> float:
    """Unweighted mean F1 over all classes, including absent ones."""
    scores = per_class_metrics(y_true, y_pred, num_labels=num_labels)
    return float(sum(item["f1"] for item in scores) / num_labels)


def top_confusions(
    matrix: Sequence[Sequence[int]],
    k: int = 5,
    labels: Sequence[str] = GENERATORS,
) -> list[dict[str, Any]]:
    """Largest off-diagonal cells, ``true -> predicted``, descending by count."""
    candidates: list[dict[str, Any]] = []
    for actual, row in enumerate(matrix):
        for guessed, count in enumerate(row):
            if actual == guessed or count <= 0:
                continue
            candidates.append(
                {
                    "true_index": actual,
                    "true": labels[actual],
                    "predicted_index": guessed,
                    "predicted": labels[guessed],
                    "count": int(count),
                }
            )
    candidates.sort(
        key=lambda item: (-item["count"], item["true_index"], item["predicted_index"])
    )
    return candidates[:k]


def evaluate_predictions(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    *,
    split: str | None = None,
) -> dict[str, Any]:
    true = [int(value) for value in y_true]
    pred = [int(value) for value in y_pred]
    matrix = confusion_matrix(true, pred)
    per_class = per_class_metrics(true, pred)
    result: dict[str, Any] = {
        "n": len(true),
        "accuracy": accuracy(true, pred),
        "macro_f1": macro_f1(true, pred),
        "per_class": {item["label"]: item for item in per_class},
        "confusion_matrix": matrix,
        "top_confusions": top_confusions(matrix, k=5),
    }
    if split is not None:
        result = {"split": split, **result}
    return result


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render_markdown(result: Mapping[str, Any]) -> str:
    lines = [
        "# E2-C1 generator attribution",
        "",
        f"- experiment: **{result['experiment']}**",
        f"- model: `{result['model_name']}`",
        f"- device: **{result['device']}**",
        f"- max_length: {result['max_length']}",
        f"- corpus SHA-256: `{result['corpus_sha256']}`",
        f"- classes: {', '.join(f'`{name}`' for name in result['labels'])}",
        f"- train rows: {result['step_plan']['rows']}",
        f"- optimizer steps: {result['step_plan']['steps_per_epoch']}/epoch, "
        f"{result['step_plan']['total_steps']} total",
        "",
        "## Splits",
        "",
        "| Split | n | Accuracy | Macro-F1 |",
        "|---|---:|---:|---:|",
    ]
    for name in ("val", "test_holdout"):
        report = result["splits"].get(name)
        if report is None:
            continue
        lines.append(
            f"| {name} | {report['n']} | {_fmt(report['accuracy'])} | "
            f"{_fmt(report['macro_f1'])} |"
        )

    lines += [
        "",
        "## Per-class (test_holdout)",
        "",
        "| Generator | support | precision | recall | F1 |",
        "|---|---:|---:|---:|---:|",
    ]
    test = result["splits"].get("test_holdout", {})
    for name in result["labels"]:
        values = test.get("per_class", {}).get(name)
        if values is None:
            continue
        lines.append(
            f"| {name} | {values['support']} | {_fmt(values['precision'])} | "
            f"{_fmt(values['recall'])} | {_fmt(values['f1'])} |"
        )

    matrix = test.get("confusion_matrix")
    if matrix:
        lines += [
            "",
            "## Confusion matrix (test_holdout)",
            "",
            "Rows are true generators, columns are predictions.",
            "",
            "| true \\ predicted | " + " | ".join(result["labels"]) + " |",
            "|---|" + "|".join("---:" for _ in result["labels"]) + "|",
        ]
        for actual, row in enumerate(matrix):
            lines.append(
                f"| {result['labels'][actual]} | "
                + " | ".join(str(value) for value in row)
                + " |"
            )

    confusions = test.get("top_confusions") or []
    if confusions:
        lines += [
            "",
            "## Top-5 confusions (test_holdout)",
            "",
            "| true | predicted | count |",
            "|---|---|---:|",
        ]
        for item in confusions:
            lines.append(
                f"| {item['true']} | {item['predicted']} | {item['count']} |"
            )

    lines += [
        "",
        "## How to read this",
        "",
        "- Human rows are excluded: this is a three-way attribution task, not",
        "  human-vs-machine detection.",
        "- LongCat is ~4% of the corpus, so the loss is class-weighted and macro-F1",
        "  is reported alongside accuracy.",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Training (torch imported lazily).
# ---------------------------------------------------------------------------


def _stratified(
    rows: Sequence[Mapping[str, Any]], n: int
) -> list[dict[str, Any]]:
    by_label: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_label.setdefault(int(row["label_index"]), []).append(dict(row))
    per_label = max(1, n // max(1, len(by_label)))
    out: list[dict[str, Any]] = []
    for label in sorted(by_label):
        out.extend(by_label[label][:per_label])
    return out


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


def train(
    *,
    corpus_path: Path = CORPUS,
    out_dir: Path = OUT_DEFAULT,
    device: str | None = None,
    smoke_steps: int | None = None,
    max_length: int | None = None,
    manifest_path: Path = MANIFEST,
) -> int:
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )

    started = time.perf_counter()
    cfg = dict(DEFAULT_CONFIG)
    if max_length is not None:
        cfg["max_length"] = int(max_length)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    outputs = {
        "json": out_dir / "E2C1_attribution.json",
        "md": out_dir / "E2C1_attribution.md",
        "predictions": out_dir / "E2C1_attribution_predictions.jsonl",
    }

    # ---- corpus integrity -------------------------------------------------
    manifest: dict[str, Any] = {}
    if Path(manifest_path).is_file():
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    corpus_sha256 = verify_corpus_digest(
        Path(corpus_path), manifest.get("corpus_sha256")
    )

    all_rows = load_corpus(Path(corpus_path))
    rows = ai_rows(all_rows)
    if not rows:
        print("ERROR: corpus has no AI rows with a known generator.", file=sys.stderr)
        return 4
    train_rows = partition_rows(rows, "train")
    val_rows = partition_rows(rows, "val")
    test_rows = partition_rows(rows, "test_holdout")
    if not train_rows:
        print("ERROR: no AI rows in the train partition.", file=sys.stderr)
        return 4

    smoke = smoke_steps is not None
    if smoke:
        cfg["max_length"] = min(int(cfg["max_length"]), 128)
        cfg["batch_size"] = min(int(cfg["batch_size"]), 8)
        train_rows = _stratified(train_rows, 600)
        val_rows = _stratified(val_rows, 300)
        test_rows = _stratified(test_rows, 300)

    seed = int(cfg["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))

    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if resolved_device.startswith("cuda") and not torch.cuda.is_available():
        print(
            f"ERROR: requested {resolved_device} but CUDA is unavailable.",
            file=sys.stderr,
        )
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
                "labels": torch.tensor(
                    int(record["label_index"]), dtype=torch.long
                ),
            }

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["model_name"], num_labels=NUM_LABELS
    ).to(resolved_device)

    weights = class_weights(train_rows)
    weight = torch.tensor(weights, dtype=torch.float, device=resolved_device)
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
    plan = planned_steps(
        len(train_rows), batch_size, grad_accum, int(cfg["epochs"])
    )
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
        state = torch.load(
            state_path, map_location=resolved_device, weights_only=False
        )
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["opt"])
        scheduler.load_state_dict(state["sched"])
        start_epoch = int(state.get("epoch", 0))
        global_step = int(state.get("step", 0))
        print(f"resumed from {state_path} at epoch {start_epoch} step {global_step}")

    every_steps = int(checkpoint_cfg.get("every_steps", CHECKPOINT_EVERY))
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
            batch = {
                key: value.to(resolved_device) for key, value in batch.items()
            }
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

    # ---- scoring ----------------------------------------------------------
    model.eval()

    def _predict(
        records: Sequence[Mapping[str, Any]],
    ) -> tuple[np.ndarray, np.ndarray]:
        if not records:
            return (
                np.array([], dtype=int),
                np.empty((0, NUM_LABELS), dtype=float),
            )
        predictions: list[int] = []
        probabilities: list[list[float]] = []
        score_loader = DataLoader(_TokenDataset(records), batch_size=batch_size)
        with torch.no_grad():
            for batch in score_loader:
                batch = {
                    key: value.to(resolved_device) for key, value in batch.items()
                }
                with torch.amp.autocast("cuda", enabled=use_amp):
                    logits = model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                    ).logits
                probs = torch.softmax(logits.float(), dim=-1).cpu().numpy()
                predictions.extend(probs.argmax(axis=-1).tolist())
                probabilities.extend(probs.tolist())
        return (
            np.asarray(predictions, dtype=int),
            np.asarray(probabilities, dtype=float),
        )

    split_reports: dict[str, dict[str, Any]] = {}
    all_predictions: list[dict[str, Any]] = []
    for name, records in (("val", val_rows), ("test_holdout", test_rows)):
        y_pred, probs = _predict(records)
        y_true = np.asarray(
            [int(row["label_index"]) for row in records], dtype=int
        )
        report = evaluate_predictions(y_true, y_pred, split=name)
        split_reports[name] = report
        for record, actual, guessed, row_probs in zip(
            records, y_true, y_pred, probs
        ):
            all_predictions.append(
                {
                    "split": name,
                    "sample_id": record["sample_id"],
                    "true_label": int(actual),
                    "true_generator": INDEX_TO_LABEL[int(actual)],
                    "predicted_label": int(guessed),
                    "predicted_generator": INDEX_TO_LABEL[int(guessed)],
                    "probabilities": [float(value) for value in row_probs],
                }
            )
        print(
            f"  {name:13} n={report['n']:<6} "
            f"accuracy={_fmt(report['accuracy'])} "
            f"macro-F1={_fmt(report['macro_f1'])}",
            flush=True,
        )

    result = {
        "experiment": EXPERIMENT,
        "status": "SMOKE" if smoke else "TRAINED",
        "smoke": smoke,
        "smoke_steps": smoke_steps,
        "model_name": cfg["model_name"],
        "device": resolved_device,
        "max_length": int(cfg["max_length"]),
        "corpus": str(corpus_path),
        "corpus_sha256": corpus_sha256,
        "labels": list(GENERATORS),
        "label_mapping": dict(LABEL_TO_INDEX),
        "partition_counts": partition_counts(rows),
        "train_class_counts": label_counts(train_rows),
        "train_class_weights": dict(zip(GENERATORS, weights)),
        "step_plan": plan,
        "steps_completed": int(global_step),
        "checkpoint": str(state_path),
        "splits": split_reports,
        "environment": environment_report(),
        "timing": {"total_s": round(time.perf_counter() - started, 2)},
        "note": (
            "SMOKE run: reduced data/steps. Do not cite these metrics."
            if smoke
            else "Three-class attribution on AI rows only; human rows excluded."
        ),
    }

    outputs["json"].write_text(json.dumps(result, indent=2), encoding="utf-8")
    outputs["md"].write_text(render_markdown(result), encoding="utf-8")
    with outputs["predictions"].open("w", encoding="utf-8") as handle:
        for prediction in all_predictions:
            handle.write(json.dumps(prediction, ensure_ascii=False) + "\n")

    print(f"wrote {outputs['json']}")
    print(f"wrote {outputs['md']}")
    print(f"wrote {outputs['predictions']}")
    return 0


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ml.train.train_attribution")
    parser.add_argument("--corpus", default=str(CORPUS))
    parser.add_argument("--out", default=str(OUT_DEFAULT))
    parser.add_argument(
        "--device",
        default=None,
        help="cuda / cuda:0 / cpu (default: cuda if available)",
    )
    parser.add_argument(
        "--smoke",
        type=int,
        default=None,
        metavar="STEPS",
        help="run N optimizer steps only, for plumbing validation",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="override the tokeniser length (use 64/128 for a CPU smoke run)",
    )
    args = parser.parse_args(argv)

    try:
        return train(
            corpus_path=Path(args.corpus),
            out_dir=Path(args.out),
            device=args.device,
            smoke_steps=args.smoke,
            max_length=args.max_length,
        )
    except FileNotFoundError as exc:
        print(f"ERROR: input not found: {exc}", file=sys.stderr)
        return 2
    except (ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
