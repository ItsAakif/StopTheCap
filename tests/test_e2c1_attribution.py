"""Offline tests for the E2-C1 three-class attribution trainer and notebook.

No training is run and torch/transformers are never imported: the trainer
imports them lazily, so every helper exercised here is pure Python/numpy.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from ml.train.train_attribution import (
    CORPUS,
    DEFAULT_CONFIG,
    GENERATORS,
    LABEL_TO_INDEX,
    accuracy,
    ai_rows,
    class_counts,
    class_weights,
    confusion_matrix,
    evaluate_predictions,
    label_index,
    load_corpus,
    macro_f1,
    partition_counts,
    partition_rows,
    per_class_metrics,
    planned_steps,
    top_confusions,
)

REPO = Path(__file__).resolve().parents[1]
E1_CONFIG = REPO / "ml" / "train" / "encoder_config.yaml"
NOTEBOOK = REPO / "notebooks" / "colab_e2c1_attribution.ipynb"
CORPUS_SHA256 = "767853403bb77d72bbfa8a2a3e5dfd424ddfa5ebe9626e4cea3a6b1ccba36675"

EXPECTED_GENERATOR_COUNTS = {
    "deepseek_v41_flash": 5248,
    "muse_spark_1_3": 6131,
    "longcat_25": 508,
}
EXPECTED_PARTITIONS = {"train": 9344, "val": 1135, "test_holdout": 1408}


@pytest.fixture(scope="module")
def corpus() -> list[dict]:
    if not CORPUS.is_file():
        pytest.skip(f"corpus absent (gitignored, not distributed): {CORPUS}")
    return load_corpus(CORPUS)


@pytest.fixture(scope="module")
def ai(corpus) -> list[dict]:
    return ai_rows(corpus)


# ---------------------------------------------------------------------------
# Corpus filtering, label mapping, partitions and class weights.
# ---------------------------------------------------------------------------


def test_ai_only_filter_drops_every_human_row(corpus, ai):
    assert len(corpus) == 23774
    assert len(ai) == 11887
    assert all(row["label"] == 1 for row in ai)
    assert {row["generator"] for row in ai} == set(GENERATORS)
    assert len(ai) + sum(1 for row in corpus if row["label"] == 0) == len(corpus)


def test_ai_rows_rejects_an_unknown_generator():
    with pytest.raises(ValueError, match="unknown generator"):
        ai_rows([{"label": 1, "generator": "someone_else", "partition": "train"}])


def test_label_mapping_is_stable_and_invertible():
    assert GENERATORS == (
        "deepseek_v41_flash",
        "muse_spark_1_3",
        "longcat_25",
    )
    assert LABEL_TO_INDEX == {
        "deepseek_v41_flash": 0,
        "muse_spark_1_3": 1,
        "longcat_25": 2,
    }
    for name, index in LABEL_TO_INDEX.items():
        assert label_index(name) == index
    with pytest.raises(ValueError):
        label_index("unknown")


def test_ai_rows_carry_the_mapped_label_index(ai):
    for row in ai:
        assert row["label_index"] == LABEL_TO_INDEX[row["generator"]]


def test_partition_counts_for_ai_rows(ai):
    assert partition_counts(ai) == EXPECTED_PARTITIONS
    assert len(partition_rows(ai, "train")) == 9344
    assert len(partition_rows(ai, "val")) == 1135
    assert len(partition_rows(ai, "test_holdout")) == 1408


def test_class_counts_match_the_manifest(ai):
    assert class_counts(ai) == EXPECTED_GENERATOR_COUNTS


def test_class_weights_are_inverse_frequency(ai):
    train = partition_rows(ai, "train")
    assert class_counts(train) == {
        "deepseek_v41_flash": 4133,
        "muse_spark_1_3": 4822,
        "longcat_25": 389,
    }
    weights = class_weights(train)
    total = 9344
    expected = [
        total / (3 * 4133),
        total / (3 * 4822),
        total / (3 * 389),
    ]
    assert weights == pytest.approx(expected)
    # LongCat is ~4% of the data, so its loss weight is an order of magnitude larger.
    assert weights[2] > 5 * weights[0]
    assert weights[2] > 5 * weights[1]
    # Every class contributes the same total weight.
    for name, weight in zip(GENERATORS, weights):
        assert class_counts(train)[name] * weight == pytest.approx(total / 3)


def test_class_weights_reject_a_missing_class():
    rows = [
        {"generator": "deepseek_v41_flash", "label_index": 0},
        {"generator": "muse_spark_1_3", "label_index": 1},
    ]
    with pytest.raises(RuntimeError, match="no rows"):
        class_weights(rows)


# ---------------------------------------------------------------------------
# Metrics on known values.
# ---------------------------------------------------------------------------


def test_confusion_matrix_known_values():
    y_true = [0, 0, 1, 1, 2, 2]
    y_pred = [0, 1, 1, 1, 2, 0]
    assert confusion_matrix(y_true, y_pred) == [
        [1, 1, 0],
        [0, 2, 0],
        [1, 0, 1],
    ]


def test_accuracy_and_macro_f1_known_values():
    y_true = [0, 0, 1, 1, 2, 2]
    y_pred = [0, 1, 1, 1, 2, 0]
    assert accuracy(y_true, y_pred) == pytest.approx(4 / 6)
    # F1 per class: 0.5, 0.8, 2/3 -> macro = 0.655555...
    assert macro_f1(y_true, y_pred) == pytest.approx(
        (0.5 + 0.8 + (2 / 3)) / 3
    )


def test_per_class_precision_recall_f1_known_values():
    y_true = [0, 0, 1, 1, 2, 2]
    y_pred = [0, 1, 1, 1, 2, 0]
    metrics = per_class_metrics(y_true, y_pred)
    assert [item["label"] for item in metrics] == list(GENERATORS)
    assert [item["support"] for item in metrics] == [2, 2, 2]
    assert metrics[0]["precision"] == pytest.approx(0.5)
    assert metrics[0]["recall"] == pytest.approx(0.5)
    assert metrics[0]["f1"] == pytest.approx(0.5)
    assert metrics[1]["precision"] == pytest.approx(2 / 3)
    assert metrics[1]["recall"] == pytest.approx(1.0)
    assert metrics[1]["f1"] == pytest.approx(0.8)
    assert metrics[2]["precision"] == pytest.approx(1.0)
    assert metrics[2]["recall"] == pytest.approx(0.5)
    assert metrics[2]["f1"] == pytest.approx(2 / 3)


def test_macro_f1_includes_absent_classes_as_zero():
    # Only classes 0 and 1 are present; class 2 contributes F1=0.
    y_true = [0, 1]
    y_pred = [0, 1]
    assert macro_f1(y_true, y_pred) == pytest.approx(2 / 3)


def test_top_confusions_are_sorted_and_exclude_the_diagonal():
    matrix = [
        [5, 3, 1],
        [2, 6, 4],
        [0, 7, 8],
    ]
    top = top_confusions(matrix, k=5)
    assert [item["count"] for item in top] == [7, 4, 3, 2, 1]
    assert top[0] == {
        "true_index": 2,
        "true": "longcat_25",
        "predicted_index": 1,
        "predicted": "muse_spark_1_3",
        "count": 7,
    }
    assert all(item["true_index"] != item["predicted_index"] for item in top)


def test_evaluate_predictions_reports_the_full_contract():
    y_true = [0, 0, 1, 1, 2, 2]
    y_pred = [0, 1, 1, 1, 2, 0]
    report = evaluate_predictions(y_true, y_pred, split="val")
    assert report["split"] == "val"
    assert report["n"] == 6
    assert report["accuracy"] == pytest.approx(4 / 6)
    assert report["macro_f1"] == pytest.approx((0.5 + 0.8 + (2 / 3)) / 3)
    assert set(report["per_class"]) == set(GENERATORS)
    assert report["confusion_matrix"] == [[1, 1, 0], [0, 2, 0], [1, 0, 1]]
    assert report["top_confusions"]


def test_step_plan_for_ai_rows():
    # All 11,887 AI rows: 743 micro-batches; 372 optimizer steps/epoch; 1,116 total.
    plan = planned_steps(11887, 16, 2, 3)
    assert plan["micro_batches_per_epoch"] == 743
    assert plan["steps_per_epoch"] == 372
    assert plan["total_steps"] == 1116

    # The actual train partition: 9,344 rows -> 584 micro-batches; 292/epoch; 876 total.
    train_plan = planned_steps(9344, 16, 2, 3)
    assert train_plan["micro_batches_per_epoch"] == 584
    assert train_plan["steps_per_epoch"] == 292
    assert train_plan["total_steps"] == 876


# ---------------------------------------------------------------------------
# Config parity with E1.
# ---------------------------------------------------------------------------


def test_config_matches_e1_architecture_and_schedule():
    e1 = yaml.safe_load(E1_CONFIG.read_text(encoding="utf-8"))
    for key in (
        "model_name",
        "max_length",
        "epochs",
        "learning_rate",
        "batch_size",
        "grad_accum",
        "weight_decay",
        "warmup_ratio",
        "fp16",
        "seed",
    ):
        assert DEFAULT_CONFIG[key] == e1[key], key
    assert DEFAULT_CONFIG["experiment"] == "E2C1_attribution"
    assert DEFAULT_CONFIG["checkpoint"]["every_steps"] == 200
    assert DEFAULT_CONFIG["checkpoint"]["resume"] is True


# ---------------------------------------------------------------------------
# Notebook: static, offline checks.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def notebook() -> dict:
    assert NOTEBOOK.is_file(), f"notebook missing: {NOTEBOOK}"
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _source(cell: dict) -> str:
    return "".join(cell["source"])


def _all_source(nb: dict) -> str:
    return "\n".join(_source(cell) for cell in nb["cells"])


def test_notebook_has_12_to_16_cells(notebook):
    assert notebook["nbformat"] == 4
    assert 12 <= len(notebook["cells"]) <= 16


def test_notebook_contains_hash_and_private_drive_paths(notebook):
    text = _all_source(notebook)
    assert CORPUS_SHA256 in text
    assert "StopTheCap_E2C1.zip" in text
    assert "drive.mount" in text
    assert "/content/drive/MyDrive/StopTheCap_reports/E2C1_attribution" in text


def test_notebook_runs_the_attribution_trainer_on_cuda(notebook):
    text = _all_source(notebook)
    assert "ml.train.train_attribution" in text
    assert '"--device", "cuda"' in text
    assert '"--out", str(REPORT_DIR)' in text
    assert "torch.cuda.is_available()" in text
    assert "shutil.copy2" in text


def test_notebook_has_no_public_github_url(notebook):
    assert "github.com/" not in _all_source(notebook)
