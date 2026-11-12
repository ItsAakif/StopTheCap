"""Export a trained E2-C1 encoder checkpoint to a Hugging Face model directory.

The E2-C1 trainer writes a resumable ``last.pt`` (model / optimizer / scheduler /
epoch / step).  This utility extracts the model weights, wraps them in the base
architecture, saves a ``save_pretrained`` directory plus tokenizer, and writes a
model card that states the intended use and the documented negative result.

Heavy dependencies (torch/transformers) are imported inside the functions so
``--help`` works without them.

Usage
-----
    python -m ml.train.export_checkpoint \\
        --checkpoint StopTheCap_reports/E2C1_essays/checkpoints/last.pt \\
        --out StopTheCap_reports/E2C1_essays/export
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

REPO = Path(__file__).resolve().parents[2]

MODEL_NAME = "E2-C1 essay detector"
DEFAULT_BASE_MODEL = "distilbert-base-cased"
DEFAULT_NUM_LABELS = 2
FROZEN_THRESHOLD = 0.0007257819990627467

#: Reported evaluation results (ROC-AUC, human false-positive rate).
RESULTS: tuple[tuple[str, float, float], ...] = (
    ("val", 1.000, 0.060),
    ("test_holdout", 1.000, 0.087),
    ("HC3", 0.447, 0.883),
    ("RAID", 0.377, 0.882),
)

#: Two short samples used to smoke-test the exported directory.
VERIFY_SAMPLES: tuple[str, ...] = (
    "The sun set behind the hills, and the children walked home.",
    "In conclusion, artificial intelligence represents a transformative paradigm shift.",
)


# ---------------------------------------------------------------------------
# Model card.
# ---------------------------------------------------------------------------


def render_model_card(base_model: str, num_labels: int) -> str:
    """Return the Markdown model card written next to the exported weights."""
    result_rows = "\n".join(
        f"| {split} | {auc:.3f} | {fpr:.1%} |" for split, auc, fpr in RESULTS
    )
    return f"""# {MODEL_NAME}

- **Model name:** {MODEL_NAME}
- **Base model:** `{base_model}`
- **Task:** binary classification — human vs machine-generated text
- **Labels:** `0` = human, `1` = machine ({num_labels} labels)
- **Training corpus:** `e2c1_essays.jsonl`, 2,178 rows: 805 Ghostbuster student
  essays + 1,373 API-generated essays from DeepSeek V4.1 Flash and Muse Spark 1.3.
- **Frozen decision threshold:** `{FROZEN_THRESHOLD!r}`

## Results

Scores are probabilities from `softmax(logits)[:, 1]`; the threshold was chosen
on `val` at a 5% human false-positive budget and frozen before any test split
was scored.

| Split | ROC-AUC | Human FPR |
|---|---:|---:|
{result_rows}

## Intended use

This is a **research artefact and a documented negative result**, not a
production detector.

- In-domain separation is a corpus artefact: the model learns shortcut features
  that happen to separate this particular corpus.
- Out-of-distribution human false-positive rate is roughly **88%**, so the model
  flags most human text from other sources as machine-generated.
- Do not deploy this model to make decisions about real users, students, or
  authors.

## How to load

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained("path/to/export")
tokenizer = AutoTokenizer.from_pretrained("path/to/export")

inputs = tokenizer("Some text to classify.", return_tensors="pt")
logits = model(**inputs).logits
predicted_class = logits.argmax(dim=-1).item()  # 0 = human, 1 = machine
```

## Limitations

- **Shortcut features:** paragraph density (newlines per 100 words) and digit
  density (digits per 100 words) carry most of the in-domain signal. These are
  documented in the `gate` block of
  `datasets/manifests/e2_c1_essays.json`, which records a failing gate
  (`newlines_per_100w` AUC 0.901; `digits_per_100w` AUC 0.097).
- The validation and holdout human splits are small, so threshold metrics are
  indicative rather than precise.
- The model does not generalise to HC3 or RAID; see the results table.
- The frozen threshold is extremely low because machine scores are near 1.0 and
  human scores are near 0.0 in-domain.
"""


# ---------------------------------------------------------------------------
# Export / verification.
# ---------------------------------------------------------------------------


def export(
    checkpoint: Path,
    out: Path,
    *,
    base_model: str = DEFAULT_BASE_MODEL,
    num_labels: int = DEFAULT_NUM_LABELS,
) -> Path:
    """Load ``checkpoint`` and write a Hugging Face model directory to ``out``."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    checkpoint = Path(checkpoint)
    out = Path(out)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    if num_labels <= 0:
        raise ValueError("num_labels must be positive")

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "model" not in payload:
        raise ValueError(f"{checkpoint}: checkpoint has no 'model' state_dict")

    model = AutoModelForSequenceClassification.from_pretrained(
        base_model, num_labels=num_labels
    )
    model.load_state_dict(payload["model"])
    model.eval()

    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.save_pretrained(out)

    (out / "README.md").write_text(
        render_model_card(base_model, num_labels), encoding="utf-8"
    )
    return out


def verify(out: Path) -> tuple[tuple[int, ...], list[int]]:
    """Reload the export, run a forward pass, and report logits/classes."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    out = Path(out)
    tokenizer = AutoTokenizer.from_pretrained(out)
    model = AutoModelForSequenceClassification.from_pretrained(out)
    model.eval()

    encoded = tokenizer(
        list(VERIFY_SAMPLES),
        truncation=True,
        padding=True,
        return_tensors="pt",
    )
    with torch.inference_mode():
        logits = model(**encoded).logits

    predicted = logits.argmax(dim=-1).tolist()
    shape = tuple(int(dim) for dim in logits.shape)
    print(f"reload check: logits shape {shape}")
    print(f"reload check: predicted classes {predicted}")
    for sample, label in zip(VERIFY_SAMPLES, predicted):
        print(f"  {label} <- {sample!r}")
    return shape, predicted


def directory_report(out: Path) -> tuple[list[Path], int]:
    """Print the exported file list and total size; return both."""
    out = Path(out)
    files = sorted(path for path in out.rglob("*") if path.is_file())
    total = sum(path.stat().st_size for path in files)
    print(
        f"exported {len(files)} file(s), total {total} bytes "
        f"({total / 1_000_000:.2f} MB)"
    )
    for path in files:
        print(f"  {path.relative_to(out)}  {path.stat().st_size} bytes")
    return files, total


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ml.train.export_checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--num-labels", type=int, default=DEFAULT_NUM_LABELS)
    args = parser.parse_args(argv)

    try:
        out = export(
            Path(args.checkpoint),
            Path(args.out),
            base_model=args.base_model,
            num_labels=args.num_labels,
        )
        verify(out)
        directory_report(out)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
