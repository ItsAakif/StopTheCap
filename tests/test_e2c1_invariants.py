"""Frozen E1/E2-C1 invariants (§F.3, T13).

These tests are read-only. They hash the pinned artefacts and compare them with the
values recorded in the SPEC. A mismatch is a stop condition: do not update a pin to make
the suite green without a separate decision.

Absent gitignored artefacts are skipped with an explicit reason that names the path; the
committed ``COMBINED.yaml`` and ``reports/e1/E1_eval.json`` are always checked.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]

COMBINED_YAML = "datasets/manifests/COMBINED.yaml"
COMBINED_SHA256 = "7427F17E381BA1164E3CF9B880C6E56DD77DAF27735F823A91D54ECB2095FDD0"
COMBINED_SPLIT_DIGEST = "932d20f1dca64c21537bed89739c633ace1b627c1a25f7d78c45a17915cf27a2"

E1_EVAL_SHA256 = "D0E4079A993628B18D2A9BAFB29381DE5BBD9111A412DBFF00C13D81BDB49B02"

# Only these paths are pinned. Absent gitignored entries are skipped explicitly below.
PINNED_SHA256 = {
    "E1/E1_eval.json": E1_EVAL_SHA256,
    "E1/ckpt/last.pt": "4AC4DE4929D8EE0E9FCBE033E8DA34F6D205829629A4F2951BDDE5F6DF827044",
    "datasets/canonical/hc3.jsonl":
        "24D6C5088FC93332106C2640FEEAB01F2804D0AC2E531D8387A8899637AE124A",
    "datasets/canonical/raid.jsonl":
        "B6D24361FCDBB15D55908C7BD1C2CCBB869942C1118970117CA9CDCE552DCD11",
    "datasets/canonical/ghostbuster.jsonl":
        "54D89A67D1E9D5A6FE21117941EB54563CDCB23CAC42C5F595D205F408ABAF0F",
}

ALL_PINNED = (COMBINED_YAML, "reports/e1/E1_eval.json", *PINNED_SHA256)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _stat_fingerprint(path: Path):
    if not path.is_file():
        return None
    stat = path.stat()
    return (stat.st_size, stat.st_mtime_ns)


@pytest.fixture(scope="session", autouse=True)
def pinned_files_are_never_modified():
    """Fail if any test in this module (or an import) writes to a pinned artefact."""
    before = {rel: _stat_fingerprint(REPO / rel) for rel in ALL_PINNED}
    yield
    after = {rel: _stat_fingerprint(REPO / rel) for rel in ALL_PINNED}
    assert before == after, "a pinned artefact was modified during the invariant tests"


def test_combined_yaml_sha256_is_pinned():
    path = REPO / COMBINED_YAML
    assert path.is_file(), f"committed artefact missing: {COMBINED_YAML}"

    actual = _sha256(path)

    assert actual == COMBINED_SHA256, (
        f"{COMBINED_YAML} sha256 changed:\n  expected {COMBINED_SHA256}\n"
        f"  actual   {actual}\n"
        "STOP: E1 is frozen. Do not update the pin; investigate the change first."
    )


def test_combined_yaml_split_digest_is_pinned():
    path = REPO / COMBINED_YAML
    assert path.is_file(), f"committed artefact missing: {COMBINED_YAML}"

    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    actual = document.get("split_digest")

    assert actual == COMBINED_SPLIT_DIGEST, (
        f"{COMBINED_YAML} split_digest changed:\n  expected {COMBINED_SPLIT_DIGEST}\n"
        f"  actual   {actual}\n"
        "STOP: E1 is frozen. Do not update the pin; investigate the change first."
    )


def test_e1_eval_committed_report_is_pinned():
    rel = "reports/e1/E1_eval.json"
    path = REPO / rel
    assert path.is_file(), f"committed artefact missing: {rel}"

    actual = _sha256(path)

    assert actual == E1_EVAL_SHA256, (
        f"{rel} sha256 changed:\n  expected {E1_EVAL_SHA256}\n  actual   {actual}\n"
        "STOP: do not update the pin."
    )


@pytest.mark.parametrize("rel,expected", sorted(PINNED_SHA256.items()))
def test_pinned_gitignored_artefact(rel: str, expected: str):
    path = REPO / rel
    if not path.is_file():
        pytest.skip(f"pinned artefact absent (gitignored, local-only): {rel}")

    actual = _sha256(path)

    assert actual == expected, (
        f"{rel} sha256 changed:\n  expected {expected}\n  actual   {actual}\n"
        "STOP: do not update the pin."
    )
