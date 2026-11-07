"""Calibration and abstention maths.

The invariant that matters most: temperature scaling must not change ranking. If it did, it would be
a discrimination change, and every claim in the report would be wrong.
"""

from __future__ import annotations

import numpy as np
import pytest

from ml.train.calibrate import (
    apply_temperature,
    brier,
    ece,
    fit_temperature,
    margin_for_coverage,
    reliability,
    risk_coverage,
    threshold_reliability,
    wilson,
)


def test_brier_perfect_and_worst():
    y = np.array([1, 0, 1, 0])
    assert brier(y, y.astype(float)) == pytest.approx(0.0)
    assert brier(y, 1 - y.astype(float)) == pytest.approx(1.0)


def test_ece_zero_when_calibrated():
    y = np.array([0, 0, 1, 1])
    assert ece(y, np.array([0.0, 0.0, 1.0, 1.0])) == pytest.approx(0.0, abs=1e-9)


def test_ece_large_when_overconfident():
    assert ece(np.zeros(20), np.full(20, 0.95)) > 0.9


def test_reliability_counts_all_rows():
    y = np.array([0, 1, 1, 0, 1])
    p = np.array([0.1, 0.3, 0.55, 0.8, 0.95])
    assert sum(b["n"] for b in reliability(y, p)) == len(y)


def test_wilson_bounds():
    lo, hi = wilson(11, 202)
    assert 0 <= lo < 11 / 202 < hi <= 1


def test_temperature_is_positive():
    rng = np.random.default_rng(0)
    y = np.r_[np.zeros(200), np.ones(200)]
    p = np.r_[rng.beta(2, 8, 200), rng.beta(8, 2, 200)]
    fit = fit_temperature(p, y)
    assert fit["temperature"] > 0
    assert fit["n_samples"] == 400


def test_temperature_preserves_ranking():
    """Temperature scaling is monotone: it must never reorder predictions."""
    rng = np.random.default_rng(1)
    p = rng.uniform(0.01, 0.99, 500)
    for T in (0.5, 1.0, 2.0, 5.0):
        q = apply_temperature(p, T)
        assert np.array_equal(np.argsort(p), np.argsort(q))


def test_temperature_of_one_is_identity():
    p = np.array([0.1, 0.4, 0.6, 0.9])
    assert np.allclose(apply_temperature(p, 1.0), p, atol=1e-9)


def test_temperature_reduces_ece_on_overconfident_data():
    rng = np.random.default_rng(2)
    y = np.r_[np.zeros(400), np.ones(400)]
    # deliberately overconfident scores
    p = np.r_[rng.beta(1, 12, 400), rng.beta(12, 1, 400)]
    T = fit_temperature(p, y)["temperature"]
    assert ece(y, apply_temperature(p, T)) <= ece(y, p) + 1e-9


def test_threshold_reliability_reports_counts_and_jackknife():
    rng = np.random.default_rng(3)
    human = rng.beta(2, 20, 202)
    r = threshold_reliability(human, 0.05)
    assert r["n_human_val"] == 202
    assert r["fpr"] == pytest.approx(r["humans_at_or_above_threshold"] / 202)
    assert r["jackknife_max_abs_shift"] >= 0
    assert r["needs_more_humans"] is True


def test_threshold_reliability_flags_enough_data():
    human = np.linspace(0.01, 0.99, 2000)
    assert threshold_reliability(human, 0.05)["needs_more_humans"] is False


def test_margin_increases_as_coverage_falls():
    rng = np.random.default_rng(4)
    y = np.r_[np.zeros(200), np.ones(200)]
    p = np.r_[rng.beta(2, 8, 200), rng.beta(8, 2, 200)]
    m90 = margin_for_coverage(y, p, 0.2, 0.90)
    m50 = margin_for_coverage(y, p, 0.2, 0.50)
    assert m50 >= m90


def test_risk_coverage_achieves_target_coverage():
    rng = np.random.default_rng(5)
    y = np.r_[np.zeros(200), np.ones(200)]
    p = np.r_[rng.beta(2, 8, 200), rng.beta(8, 2, 200)]
    thr = 0.2
    m = margin_for_coverage(y, p, thr, 0.75)
    r = risk_coverage(y, p, thr, [m])[0]
    assert 0.70 <= r["coverage"] <= 0.80


def test_risk_coverage_selective_risk_is_bounded():
    rng = np.random.default_rng(6)
    y = np.r_[np.zeros(100), np.ones(100)]
    p = np.r_[rng.beta(2, 8, 100), rng.beta(8, 2, 100)]
    for row in risk_coverage(y, p, 0.2, [0.0, 0.05, 0.2]):
        assert 0.0 <= row["selective_risk"] <= 1.0


def test_risk_coverage_handles_zero_accepted():
    y = np.array([0, 1, 0, 1])
    p = np.array([0.1, 0.9, 0.2, 0.8])
    row = risk_coverage(y, p, 0.5, [10.0])[0]
    assert row["n_accepted"] == 0
    assert row["coverage"] == 0.0
