"""Offline tests for the E2-C1 trainer, metrics and Colab notebook.

No training is run and torch/transformers are never imported: the trainer
imports them lazily, so every helper here is a pure numpy/Python function.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from ml.train.train_e2c1 import (
    CORPUS,
    ECE_BINS,
    class_counts,
    class_weights,
    ece,
    evaluate_split,
    human_fpr,
    load_config,
    load_corpus,
    per_generator_metrics,
    per_topic_metrics,
    planned_steps,
    pr_auc,
    precision_recall_at_threshold,
    roc_auc,
    select_threshold,
    sha256_file,
    wilson,
)
from scripts.make_colab_zip import (
    DATA_FILES,
    E2C1_TRAINING_FILES,
    build_zip,
    main as zip_main,
    verify_zip_members,
)

REPO = Path(__file__).resolve().parents[1]
E2C1_CONFIG = REPO / "ml" / "train" / "e2c1_config.yaml"
E1_CONFIG = REPO / "ml" / "train" / "encoder_config.yaml"
NOTEBOOK = REPO / "notebooks" / "colab_train_e2c1.ipynb"
CORPUS_SHA256 = "767853403bb77d72bbfa8a2a3e5dfd424ddfa5ebe9626e4cea3a6b1ccba36675"


# ---------------------------------------------------------------------------
# Corpus loader and class balance.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def corpus() -> list[dict]:
    return load_corpus(CORPUS)


def test_loader_reads_the_committed_corpus(corpus):
    assert len(corpus) == 23774
    assert sha256_file(CORPUS) == CORPUS_SHA256
    assert all({"sample_id", "text", "label", "partition"} <= set(row) for row in corpus)


def test_split_counts(corpus):
    counts = {partition: 0 for partition in ("train", "val", "test_holdout")}
    for row in corpus:
        counts[row["partition"]] += 1
    assert counts == {"train": 18688, "val": 2270, "test_holdout": 2816}


def test_label_balance_is_one_to_one_per_partition(corpus):
    for partition in ("train", "val", "test_holdout"):
        rows = [row for row in corpus if row["partition"] == partition]
        assert class_counts(rows) == {
            "human": len(rows) // 2,
            "machine": len(rows) // 2,
        }


def test_class_weights_are_balanced_for_the_e2c1_corpus(corpus):
    train = [row for row in corpus if row["partition"] == "train"]
    assert class_weights(train) == pytest.approx((1.0, 1.0))


def test_class_weights_correct_an_imbalance():
    rows = [{"label": 0}] * 100 + [{"label": 1}] * 300
    human, machine = class_weights(rows)
    assert human == pytest.approx(2.0)
    assert machine == pytest.approx(2.0 / 3.0)


def test_loader_rejects_an_invalid_label(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text(
        json.dumps(
            {
                "sample_id": "x",
                "text": "hello",
                "label": 2,
                "partition": "train",
                "source_dataset": "d",
                "group_id": "g",
                "generator": None,
                "topic_category": "other",
                "content_hash": "h",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="label"):
        load_corpus(path)


def test_step_plan_for_the_whole_corpus():
    # 23,774 rows / 16 = 1,486 micro-batches; / 2 = 743 optimizer steps/epoch.
    plan = planned_steps(23774, 16, 2, 3)
    assert plan["micro_batches_per_epoch"] == 1486
    assert plan["steps_per_epoch"] == 743
    assert plan["total_steps"] == 2229


# ---------------------------------------------------------------------------
# Threshold selection at a 5% human FPR.
# ---------------------------------------------------------------------------


def test_threshold_flags_five_percent_of_humans():
    humans = np.arange(100, dtype=float) / 100.0
    threshold = select_threshold(humans, 0.05)
    assert float((humans >= threshold).mean()) == pytest.approx(0.05)


def test_threshold_is_the_human_quantile_not_a_test_quantile():
    humans = np.array([0.0] * 95 + [0.9] * 5)
    threshold = select_threshold(humans, 0.05)
    assert threshold == pytest.approx(0.045)
    assert int((humans >= threshold).sum()) == 5


def test_human_fpr_uses_only_label_zero_rows():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    scores = np.array([0.1, 0.2, 0.9, 0.95, 0.99, 0.99, 0.99, 0.99])
    result = human_fpr(y, scores, 0.5)
    assert result["n_human"] == 4
    assert result["false_positives"] == 2
    assert result["human_fpr"] == pytest.approx(0.5)
    assert result["human_fpr_ci"] is not None


def test_threshold_rejects_empty_input():
    with pytest.raises(ValueError):
        select_threshold([], 0.05)


# ---------------------------------------------------------------------------
# Metrics on known values.
# ---------------------------------------------------------------------------


def test_roc_auc_known_values():
    assert roc_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == pytest.approx(1.0)
    assert roc_auc([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]) == pytest.approx(0.0)
    assert roc_auc([0, 1], [0.5, 0.5]) == pytest.approx(0.5)
    assert roc_auc([1, 1, 1], [0.1, 0.2, 0.3]) is None


def test_pr_auc_known_values():
    assert pr_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == pytest.approx(1.0)
    assert pr_auc([0, 1], [0.9, 0.1]) == pytest.approx(0.5)


def test_wilson_known_interval():
    lo, hi = wilson(5, 100)
    assert lo == pytest.approx(0.0215, abs=1e-3)
    assert hi == pytest.approx(0.1117, abs=1e-3)
    assert wilson(0, 0) == (0.0, 0.0)
    assert wilson(100, 100)[1] == pytest.approx(1.0)


def test_brier_known_values():
    from ml.train.train_e2c1 import brier

    assert brier([0, 1], [0.0, 1.0]) == pytest.approx(0.0)
    assert brier([0, 1], [0.5, 0.5]) == pytest.approx(0.25)


def test_ece_is_zero_when_perfect_and_detects_overconfidence():
    assert ece([0, 0, 1, 1], [0.0, 0.0, 1.0, 1.0]) == pytest.approx(0.0)
    assert ece([0] * 10, [0.9] * 10) == pytest.approx(0.9)
    assert ECE_BINS == 15


def test_precision_recall_and_confusion_are_consistent():
    y = [0, 0, 1, 1]
    scores = [0.1, 0.8, 0.6, 0.9]
    result = precision_recall_at_threshold(y, scores, 0.5)
    confusion = result["confusion"]
    assert sum(confusion.values()) == len(y)
    assert result["precision"] == pytest.approx(2 / 3)
    assert result["recall"] == pytest.approx(1.0)


def test_evaluate_split_reports_every_required_metric():
    rows = [
        {"label": 0, "generator": None, "topic_category": "other"},
        {"label": 0, "generator": None, "topic_category": "other"},
        {"label": 1, "generator": "g", "topic_category": "other"},
        {"label": 1, "generator": "g", "topic_category": "other"},
    ]
    report = evaluate_split("val", rows, [0.1, 0.2, 0.8, 0.9], 0.5)
    for field in ("roc_auc", "pr_auc", "human_fpr", "human_fpr_ci", "precision",
                  "recall", "f1", "brier", "ece", "confusion"):
        assert field in report
    assert report["n"] == 4
    assert report["n_human"] == 2
    assert report["n_machine"] == 2


def test_per_generator_and_per_topic_metrics():
    rows = [
        {"label": 0, "generator": None, "topic_category": "stem"},
        {"label": 0, "generator": None, "topic_category": "stem"},
        {"label": 1, "generator": "a", "topic_category": "stem"},
        {"label": 1, "generator": "a", "topic_category": "arts"},
        {"label": 1, "generator": "b", "topic_category": "arts"},
    ]
    scores = [0.1, 0.2, 0.9, 0.4, 0.8]
    generators = per_generator_metrics(rows, scores, 0.5)
    assert set(generators) == {"a", "b"}
    assert generators["a"]["detection_rate"] == pytest.approx(0.5)
    assert generators["b"]["detection_rate"] == pytest.approx(1.0)

    topics = per_topic_metrics(rows, scores, 0.5)
    assert topics["stem"]["human_fpr"] == pytest.approx(0.0)
    assert topics["arts"]["recall"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Config parity with E1.
# ---------------------------------------------------------------------------


def test_config_matches_e1_architecture_and_schedule():
    e1 = load_config(E1_CONFIG)
    e2 = load_config(E2C1_CONFIG)
    for key in ("model_name", "max_length", "epochs", "learning_rate", "batch_size",
                "grad_accum", "weight_decay", "warmup_ratio", "fp16", "seed"):
        assert e2[key] == e1[key], key
    assert e2["experiment"] == "E2C1"
    assert e2["threshold"]["selection_split"] == "val"
    assert e2["threshold"]["target_human_fpr"] == 0.05
    assert e2["checkpoint"]["every_steps"] == 200
    assert e2["checkpoint"]["resume"] is True


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


def test_notebook_has_15_to_20_cells(notebook):
    assert notebook["nbformat"] == 4
    assert 15 <= len(notebook["cells"]) <= 20


def test_notebook_contains_the_corpus_hash_and_split_digest(notebook):
    text = _all_source(notebook)
    assert CORPUS_SHA256 in text
    assert "adfe9df5900dac1342ff241ef1eff7ed5cf933cdcc728dfe4e69e0c00d94fb97" in text


def test_notebook_validates_required_files_and_sha256(notebook):
    text = _all_source(notebook)
    for token in (
        "REQUIRED",
        "datasets/canonical/e2c1.jsonl",
        "datasets/manifests/e2_c1_training.json",
        "datasets/manifests/e2_c1_corpus.json",
        "hashlib.sha256",
        "split_digest",
    ):
        assert token in text, token


def test_notebook_mounts_drive_and_runs_training_to_drive(notebook):
    text = _all_source(notebook)
    assert "drive.mount" in text
    assert "StopTheCap_E2C1.zip" in text
    assert "ml.train.train_e2c1" in text
    assert (
        "python -m ml.train.train_e2c1 --device cuda "
        "--out /content/drive/MyDrive/StopTheCap_reports/E2C1"
    ) in text
    assert '"--device", "cuda"' in text
    assert "/content/drive/MyDrive/StopTheCap_reports/E2C1" in text
    assert "%pip install" in text
    for package in ("transformers", "accelerate", "scikit-learn", "torch"):
        assert package in text, package


def test_notebook_checks_gpu_and_prints_step_estimate(notebook):
    text = _all_source(notebook)
    assert "torch.cuda.is_available()" in text
    assert "total_memory >= 14.5e9" in text
    assert "15.6" in text
    assert "steps_per_epoch" in text
    assert "TOTAL_SAMPLES = 23774" in text
    assert "BATCH_SIZE = 16" in text
    assert "GRAD_ACCUM = 2" in text
    assert "743" in text and "2,229" in text


def test_notebook_does_not_stop_early_or_use_a_public_url(notebook):
    text = _all_source(notebook)
    assert "github.com/" not in text
    assert "SystemExit" not in text
    assert "raise RuntimeError(f\"training failed" in text  # only on a real failure


# ---------------------------------------------------------------------------
# Colab ZIP packaging: the training allowlist.
# ---------------------------------------------------------------------------


def _write(root: Path, rel: str, text: str = "stub\n") -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _seed_training(root: Path) -> None:
    for rel in E2C1_TRAINING_FILES:
        _write(root, rel, "{}\n")
    _write(root, "datasets/canonical/hc3.jsonl", "{}\n")
    _write(root, "ml/__init__.py")


def _names(path: Path) -> set[str]:
    with zipfile.ZipFile(path) as zf:
        return set(zf.namelist())


def test_training_allowlist_is_exactly_the_three_inputs():
    assert E2C1_TRAINING_FILES == (
        "datasets/canonical/e2c1.jsonl",
        "datasets/manifests/e2_c1_training.json",
        "datasets/manifests/e2_c1_corpus.json",
    )


def test_build_zip_with_training_includes_corpus_but_not_other_canonical(tmp_path):
    root = tmp_path / "repo"
    _seed_training(root)
    out, _, _ = build_zip(tmp_path / "train.zip", root=root, with_e2c1_training=True)
    names = _names(out)
    for rel in E2C1_TRAINING_FILES:
        assert f"StopTheCap/{rel}" in names
    assert "StopTheCap/datasets/canonical/hc3.jsonl" not in names


def test_verifier_allows_only_the_e2c1_corpus_under_canonical():
    names = [
        "StopTheCap/datasets/canonical/e2c1.jsonl",
        "StopTheCap/datasets/canonical/hc3.jsonl",
    ]
    problems = verify_zip_members(
        names,
        required=E2C1_TRAINING_FILES,
        allow_canonical=("datasets/canonical/e2c1.jsonl",),
    )
    assert any("hc3.jsonl" in problem for problem in problems)
    assert not any("e2c1.jsonl" in problem for problem in problems)
    assert any(problem.startswith("missing required member") for problem in problems)


def test_cli_training_flag_requires_all_three_files(tmp_path, capsys):
    root = tmp_path / "repo"
    _seed_training(root)
    (root / "datasets/manifests/e2_c1_corpus.json").unlink()
    out = tmp_path / "train.zip"

    rc = zip_main(["--out", str(out), "--with-e2c1-training"], root=root)

    captured = capsys.readouterr()
    assert rc == 2
    assert "datasets/manifests/e2_c1_corpus.json" in captured.err
    assert not out.exists()


def test_cli_training_flag_writes_and_verifies_the_archive(tmp_path, capsys):
    root = tmp_path / "repo"
    _seed_training(root)
    out = tmp_path / "train.zip"

    rc = zip_main(["--out", str(out), "--with-e2c1-training"], root=root)

    assert rc == 0
    names = _names(out)
    for rel in E2C1_TRAINING_FILES:
        assert f"StopTheCap/{rel}" in names
    assert "E2-C1 training inputs verified" in capsys.readouterr().out


def test_with_data_and_training_together_are_both_allowed(tmp_path, capsys):
    root = tmp_path / "repo"
    _seed_training(root)
    for rel in DATA_FILES:
        _write(root, rel, "{}\n")
    out = tmp_path / "combo.zip"

    rc = zip_main(
        ["--out", str(out), "--with-data", "--with-e2c1-training"], root=root
    )

    assert rc == 0, capsys.readouterr().err
    names = _names(out)
    for rel in E2C1_TRAINING_FILES + DATA_FILES:
        assert f"StopTheCap/{rel}" in names, rel
