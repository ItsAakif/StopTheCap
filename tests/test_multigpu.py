"""Scaling arithmetic for the multi-GPU benchmark (pure logic, no CUDA)."""

from __future__ import annotations

import pytest

from ml.cloud.multigpu_bench import scaling_summary


def test_single_device_is_1x():
    s = scaling_summary([{"device": 0, "examples_per_s": 80.0}], 1)
    assert s["speedup"] == pytest.approx(1.0)
    assert s["efficiency"] == pytest.approx(1.0)


def test_two_devices_linear_scaling():
    s = scaling_summary([
        {"device": 0, "examples_per_s": 80.0},
        {"device": 1, "examples_per_s": 80.0},
    ], 2)
    assert s["aggregate_examples_per_s"] == pytest.approx(160.0)
    assert s["speedup"] == pytest.approx(2.0)
    assert s["efficiency"] == pytest.approx(1.0)


def test_cpu_bound_scaling_is_flagged():
    s = scaling_summary([
        {"device": 0, "examples_per_s": 80.0},
        {"device": 1, "examples_per_s": 10.0},
    ], 2)
    assert s["speedup"] < 1.2
    assert s["efficiency"] < 0.6


def test_missing_results_are_not_usable():
    s = scaling_summary([{"device": 0, "error": "boom"}, {"device": 1}], 2)
    assert s["usable"] is False


def test_partial_results_still_summed():
    s = scaling_summary([
        {"device": 0, "examples_per_s": 80.0},
        {"device": 1, "error": "OOM"},
    ], 2)
    assert s["usable"] is True
    assert s["devices_measured"] == 1
    assert s["devices_requested"] == 2
