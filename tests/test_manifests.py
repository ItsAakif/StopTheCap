"""Manifest discipline: every enabled dataset must have verifiable provenance.

These tests enforce the licensing policy (research/13, ADR-013) mechanically, so a
dataset cannot be added later without its evidence URL or with training silently
enabled on an eval-only source.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

MANIFEST_DIR = Path(__file__).resolve().parents[1] / "datasets" / "manifests"


def _all_manifests():
    out = {}
    for p in sorted(MANIFEST_DIR.glob("*.yaml")):
        if p.name == "COMBINED.yaml":
            continue
        out[p.name] = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return out


def test_manifests_exist():
    assert _all_manifests(), "no dataset manifests found"


def test_every_enabled_manifest_has_licence_evidence():
    for name, m in _all_manifests().items():
        if not m.get("enabled", True):
            continue
        lic = m.get("licence", {})
        assert lic.get("spdx"), f"{name}: missing licence.spdx"
        assert lic.get("evidence_url"), f"{name}: missing licence.evidence_url"
        assert lic.get("verified_at"), f"{name}: missing licence.verified_at"


def test_unknown_licence_is_marked_unresolved():
    for name, m in _all_manifests().items():
        lic = m.get("licence", {})
        if lic.get("spdx") in ("UNKNOWN", None):
            assert lic.get("unresolved") is True, f"{name}: unknown licence must be unresolved"
            assert m.get("enabled", True) is False or not m.get("permitted_uses", {}).get("train"), (
                f"{name}: unknown licence must not be trainable")


def test_eval_only_datasets_cannot_train():
    for name, m in _all_manifests().items():
        if m.get("role") in ("quarantine_eval", "video_inventory"):
            assert m.get("permitted_uses", {}).get("train") is False, (
                f"{name}: role {m.get('role')} must not permit training")


def test_blocked_datasets_are_disabled():
    for name, m in _all_manifests().items():
        if m.get("blocked_reason"):
            assert m.get("enabled") is False, f"{name}: blocked dataset must be disabled"


def test_no_commercial_use_claim_on_noncommercial_licence():
    for name, m in _all_manifests().items():
        spdx = m.get("licence", {}).get("spdx", "")
        if "NC" in str(spdx):
            permits = m.get("permitted_uses", {})
            assert permits.get("commercial") is False, (
                f"{name}: {spdx} must not permit commercial use")


def test_ghostbuster_is_the_only_trainable_text_source():
    trainable = []
    for name, m in _all_manifests().items():
        if not m.get("enabled", True):
            continue
        if not m.get("permitted_uses", {}).get("train"):
            continue
        if "text" not in str(m.get("modality", "")):
            continue
        trainable.append(name)
    assert trainable == ["ghostbuster.yaml"], (
        f"unexpected trainable text sources: {trainable}"
    )
