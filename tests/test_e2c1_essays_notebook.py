"""Static, offline checks for the essay-matched E2-C1 Colab notebook.

No cell is executed, no GPU/CUDA/network is touched.  The ``--manifest`` tests
monkeypatch ``train`` so the CLI is exercised without importing torch or
loading a model.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import ml.train.train_e2c1 as trainer

REPO = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO / "notebooks" / "colab_train_e2c1_essays.ipynb"
CORPUS = REPO / "datasets" / "canonical" / "e2c1_essays.jsonl"
MANIFEST = REPO / "datasets" / "manifests" / "e2_c1_essays.json"
LEGACY_MANIFEST = REPO / "datasets" / "manifests" / "e2_c1_training.json"
CORPUS_SHA256 = "8b0c0cc3beebe024302c0e87b75995a71378b30ffa3c7605b5a11f8e6a75bf2c"
DRIVE_OUT = "/content/drive/MyDrive/StopTheCap_reports/E2C1_essays"

requires_corpus = pytest.mark.skipif(
    not CORPUS.is_file(),
    reason="essays corpus absent (gitignored, not distributed)",
)


@pytest.fixture(scope="module")
def notebook() -> dict:
    assert NOTEBOOK.is_file(), f"notebook missing: {NOTEBOOK}"
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _source(cell: dict) -> str:
    return "".join(cell["source"])


def _all_source(nb: dict) -> str:
    return "\n".join(_source(cell) for cell in nb["cells"])


# ---------------------------------------------------------------------------
# Notebook: static checks.
# ---------------------------------------------------------------------------


def test_notebook_has_15_to_18_cells(notebook):
    assert notebook["nbformat"] == 4
    assert notebook["nbformat_minor"] == 5
    assert 15 <= len(notebook["cells"]) <= 18


def test_notebook_pins_the_essays_corpus_hash_and_path(notebook):
    text = _all_source(notebook)

    assert CORPUS_SHA256 in text
    assert "datasets/canonical/e2c1_essays.jsonl" in text
    assert "datasets/manifests/e2_c1_essays.json" in text
    assert "2178" in text
    assert "1373" in text
    assert "805" in text


def test_notebook_uses_manifest_flag_and_writes_to_drive(notebook):
    text = _all_source(notebook)

    assert '"--manifest"' in text
    assert '"--corpus"' in text
    assert DRIVE_OUT in text
    assert "drive.mount" in text
    assert "StopTheCap_E2C1.zip" in text
    assert "ml.train.train_e2c1" in text


def test_notebook_checks_gpu_and_prints_the_step_estimate(notebook):
    text = _all_source(notebook)

    assert "torch.cuda.is_available()" in text
    assert "total_memory >= 14.5e9" in text
    assert "BATCH_SIZE = 16" in text
    assert "GRAD_ACCUM = 2" in text
    assert "408" in text
    assert "%pip install" in text
    for package in ("torch>=2.0", "transformers>=4.40", "accelerate>=0.30",
                    "scikit-learn>=1.3"):
        assert package in text, package


def test_notebook_prints_the_gate_failure_warning(notebook):
    text = _all_source(notebook)

    assert "GATE WARNING" in text
    assert "FAIL" in text
    assert "newlines_per_100w" in text
    assert "digits_per_100w" in text


def test_notebook_has_no_public_github_reference(notebook):
    assert "github.com/" not in _all_source(notebook)


# ---------------------------------------------------------------------------
# Trainer: --manifest resolution and validation.
# ---------------------------------------------------------------------------


def test_default_manifest_is_derived_for_each_corpus():
    assert trainer.default_manifest_path(CORPUS) == MANIFEST
    assert trainer.default_manifest_path(
        REPO / "datasets" / "canonical" / "e2c1.jsonl"
    ) == LEGACY_MANIFEST


@requires_corpus
def test_explicit_manifest_validates_the_pinned_sha():
    manifest, digest = trainer.load_and_verify_manifest(
        CORPUS, MANIFEST, required=True
    )

    assert digest == CORPUS_SHA256
    assert manifest["corpus_sha256"] == CORPUS_SHA256


@requires_corpus
def test_manifest_sha_mismatch_is_rejected(tmp_path):
    bad = tmp_path / "bad_manifest.json"
    bad.write_text(json.dumps({"corpus_sha256": "0" * 64}), encoding="utf-8")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        trainer.load_and_verify_manifest(CORPUS, bad, required=True)


def test_missing_explicit_manifest_is_rejected(tmp_path):
    with pytest.raises(FileNotFoundError):
        trainer.load_and_verify_manifest(
            CORPUS, tmp_path / "absent.json", required=True
        )


def test_cli_manifest_overrides_the_default(tmp_path, monkeypatch):
    captured: dict = {}

    def fake_train(cfg, **kwargs):
        captured["kwargs"] = kwargs
        return 0

    monkeypatch.setattr(trainer, "train", fake_train)
    explicit = tmp_path / "explicit_manifest.json"
    explicit.write_text(json.dumps({"corpus_sha256": CORPUS_SHA256}), encoding="utf-8")

    rc = trainer.main(
        [
            "--corpus", str(CORPUS),
            "--manifest", str(explicit),
            "--out", str(tmp_path / "out"),
        ]
    )

    assert rc == 0
    assert captured["kwargs"]["manifest_path"] == explicit
    assert captured["kwargs"]["manifest_required"] is True


def test_cli_without_manifest_derives_it(tmp_path, monkeypatch):
    captured: dict = {}

    def fake_train(cfg, **kwargs):
        captured["kwargs"] = kwargs
        return 0

    monkeypatch.setattr(trainer, "train", fake_train)

    rc = trainer.main(["--corpus", str(CORPUS), "--out", str(tmp_path / "out")])

    assert rc == 0
    assert captured["kwargs"]["manifest_path"] == MANIFEST
    assert captured["kwargs"]["manifest_required"] is False


def test_explicit_manifest_does_not_require_the_legacy_manifest(tmp_path, monkeypatch):
    # Even if the old e2_c1_training.json is gone, an explicit --manifest is enough.
    monkeypatch.setattr(trainer, "MANIFEST", tmp_path / "legacy_absent.json")
    monkeypatch.setattr(trainer, "train", lambda cfg, **kwargs: 0)
    explicit = tmp_path / "explicit_manifest.json"
    explicit.write_text(json.dumps({"corpus_sha256": CORPUS_SHA256}), encoding="utf-8")

    rc = trainer.main(
        [
            "--corpus", str(CORPUS),
            "--manifest", str(explicit),
            "--out", str(tmp_path / "out"),
        ]
    )

    assert rc == 0
