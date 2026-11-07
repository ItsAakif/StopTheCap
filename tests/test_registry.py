"""Registry/manifest consistency.

The registry is only useful if it tracks the manifests. These tests fail when a dataset is
added to the pipeline without appearing in DATASET_REGISTRY.md, or when the registry claims a
dataset is integrated while no canonical output exists.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "DATASET_REGISTRY.md"
MANIFESTS = ROOT / "datasets" / "manifests"
CANONICAL = ROOT / "datasets" / "canonical"


@pytest.fixture(scope="module")
def registry_text() -> str:
    assert REGISTRY.is_file(), "DATASET_REGISTRY.md is missing"
    return REGISTRY.read_text(encoding="utf-8")


def _manifest_ids() -> list[str]:
    ids = []
    for p in sorted(MANIFESTS.glob("*.yaml")):
        if p.name == "COMBINED.yaml":
            continue
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        if data.get("dataset_id"):
            ids.append(data["dataset_id"])
    return ids


def test_every_manifest_dataset_is_named_in_the_registry(registry_text: str):
    missing = []
    for ds in _manifest_ids():
        # manifest ids use underscores (m4gt_bench); the registry uses prose names
        key = ds.replace("_", "-").lower()
        alt = ds.replace("_", " ").lower()
        if key not in registry_text.lower() and alt not in registry_text.lower():
            missing.append(ds)
    assert not missing, f"datasets missing from DATASET_REGISTRY.md: {missing}"


def test_registry_lists_the_training_source(registry_text: str):
    assert "Ghostbuster" in registry_text
    assert "CC BY 3.0" in registry_text or "CC-BY-3.0" in registry_text


def test_registry_lists_eval_only_sources(registry_text: str):
    for name in ("RAID", "HC3"):
        assert name in registry_text


def test_integrated_claims_have_canonical_output(registry_text: str):
    """A dataset may only be called Integrated if canonical records exist for it."""
    if not CANONICAL.is_dir():
        pytest.skip(
            "datasets/canonical is not distributed with the repository; "
            "run the data build to verify integrated claims locally"
        )
    for ds in ("ghostbuster", "raid", "hc3"):
        if f"`{ds}`" in registry_text or ds in registry_text:
            assert (CANONICAL / f"{ds}.jsonl").is_file(), (
                f"registry mentions {ds} but datasets/canonical/{ds}.jsonl is missing")


def test_registry_training_claims_are_backed_by_artefacts(registry_text: str):
    """A "trained" claim must point at committed evidence, not just prose.

    This replaced an earlier guard that asserted *no* model had been trained. Once E1 ran, that
    guard was wrong - but the underlying rule still holds: never claim training without artefacts.
    """
    lowered = registry_text.lower()
    claims_training = "trained (e1" in lowered or "one model has been trained" in lowered
    if claims_training:
        assert (ROOT / "docs" / "E1-results.md").is_file(), "trained claim needs docs/E1-results.md"
        assert (ROOT / "reports" / "e1" / "E1_eval.json").is_file(), "needs committed eval JSON"


def test_registry_does_not_claim_a_video_detector_exists(registry_text: str):
    lowered = registry_text.lower()
    assert "no architecture implemented" in lowered or "none selected" in lowered


def test_registry_records_the_split_digest(registry_text: str):
    combined = yaml.safe_load((MANIFESTS / "COMBINED.yaml").read_text(encoding="utf-8"))
    digest = combined.get("split_digest")
    assert digest and digest in registry_text, (
        "registry must quote the split digest so a reader can tie it to the data")


def test_registry_marks_unverified_sources(registry_text: str):
    for name in ("SemEval", "HD-VG-130M", "DFDC"):
        assert name in registry_text, f"{name} must appear in the registry"
    assert "UNVERIFIED" in registry_text or "BLOCKED" in registry_text
