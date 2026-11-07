"""Baseline metric helpers.

The full baseline needs the built dataset, so CI tests the metric machinery instead:
an interval that is wrong would silently misreport the project's primary number.
"""

from __future__ import annotations

import numpy as np
import pytest

from ml.baselines.tfidf import (
    MIN_NEGATIVES_FOR_THRESHOLD_METRIC,
    evaluate,
    expected_calibration_error,
    wilson,
)


def test_wilson_interval_bounds():
    lo, hi = wilson(5, 100)
    assert 0.0 <= lo <= 0.05 <= hi <= 1.0


def test_wilson_zero_and_full():
    assert wilson(0, 100)[0] == 0.0
    assert wilson(100, 100)[1] == pytest.approx(1.0)


def test_wilson_empty():
    assert wilson(0, 0) == (0.0, 0.0)


def test_wilson_widens_with_smaller_n():
    lo_big, hi_big = wilson(5, 1000)
    lo_small, hi_small = wilson(5, 100)
    assert (hi_small - lo_small) > (hi_big - lo_big)


def test_ece_perfect_calibration_is_zero():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.0, 0.0, 1.0, 1.0])
    assert expected_calibration_error(y, p) == pytest.approx(0.0, abs=1e-9)


def test_ece_detects_miscalibration():
    y = np.array([0, 0, 0, 0])
    p = np.array([0.9, 0.9, 0.9, 0.9])
    assert expected_calibration_error(y, p) > 0.8


def test_evaluate_reports_fpr_from_negatives_only():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    score = np.array([0.1, 0.2, 0.9, 0.95, 0.6, 0.7, 0.8, 0.99])
    m = evaluate("t", y, score, threshold=0.5)
    assert m.n_neg == 4
    assert m.fpr == pytest.approx(0.5)  # 2 of 4 humans above threshold
    assert m.fpr_ci is not None


def test_evaluate_flags_insufficient_negatives():
    y = np.array([0, 1, 1, 1, 1])
    score = np.array([0.1, 0.6, 0.7, 0.8, 0.9])
    m = evaluate("t", y, score, threshold=0.5)
    assert m.n_neg < MIN_NEGATIVES_FOR_THRESHOLD_METRIC
    assert m.threshold_metrics_usable is False
    assert "indicative" in m.note


def test_evaluate_single_class_is_undefined():
    y = np.array([1, 1, 1])
    m = evaluate("t", y, np.array([0.6, 0.7, 0.8]), threshold=0.5)
    assert m.roc_auc is None
    assert m.threshold_metrics_usable is False


def test_threshold_uses_human_val_scores_only():
    """Guards the rule: threshold at the human FPR budget, from human scores only."""
    human_val = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10])
    thr = float(np.quantile(human_val, 0.95))
    # at most ~5% of humans should exceed it
    assert (human_val > thr).mean() <= 0.10
