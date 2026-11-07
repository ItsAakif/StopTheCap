"""Metric helpers used by the encoder evaluation harness."""

from __future__ import annotations

import numpy as np
import pytest

from ml.train.evaluate_encoder import (
    bootstrap_auc,
    ece,
    evaluate_split,
    reliability_bins,
    wilson,
)


def test_wilson_bounds():
    lo, hi = wilson(5, 100)
    assert 0.0 <= lo <= 0.05 <= hi <= 1.0


def test_wilson_full_and_empty():
    assert wilson(100, 100)[1] == pytest.approx(1.0)
    assert wilson(0, 0) == (0.0, 0.0)


def test_ece_perfect_is_zero():
    y = np.array([0, 0, 1, 1])
    assert ece(y, np.array([0.0, 0.0, 1.0, 1.0])) == pytest.approx(0.0, abs=1e-9)


def test_ece_detects_overconfidence():
    y = np.zeros(10)
    assert ece(y, np.full(10, 0.9)) > 0.8


def test_reliability_bins_cover_all_rows():
    y = np.array([0, 1, 1, 0, 1])
    p = np.array([0.1, 0.3, 0.55, 0.8, 0.95])
    bins = reliability_bins(y, p)
    assert sum(b["n"] for b in bins) == len(y)


def test_evaluate_split_confusion_is_consistent():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    s = np.array([0.1, 0.2, 0.9, 0.95, 0.6, 0.7, 0.8, 0.99])
    r = evaluate_split("t", y, s, 0.5)
    c = r.confusion
    assert c["tn"] + c["fp"] + c["fn"] + c["tp"] == len(y)
    assert r.human_fpr == pytest.approx(c["fp"] / 4)
    assert r.human_fpr_ci is not None


def test_evaluate_split_flags_few_negatives():
    y = np.array([0, 1, 1, 1, 1])
    r = evaluate_split("t", y, np.array([0.1, 0.6, 0.7, 0.8, 0.9]), 0.5)
    assert "indicative" in r.note


def test_evaluate_split_single_class():
    r = evaluate_split("t", np.ones(5), np.linspace(0.1, 0.9, 5), 0.5)
    assert r.roc_auc is None


def test_bootstrap_auc_ci_brackets_point():
    rng = np.random.default_rng(0)
    y = np.array([0] * 60 + [1] * 60)
    s = np.concatenate([rng.normal(0.3, 0.2, 60), rng.normal(0.7, 0.2, 60)])
    out = bootstrap_auc(y, s, n=200)
    assert out["lo"] <= out["point"] <= out["hi"]


def test_bootstrap_auc_insufficient_rows():
    out = bootstrap_auc(np.array([1, 1, 1]), np.array([0.1, 0.2, 0.3]))
    assert out["point"] is None
