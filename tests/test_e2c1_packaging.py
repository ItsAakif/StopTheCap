"""Colab ZIP packaging: the E2-C1 allowlist and the safety verification (§F)."""

from __future__ import annotations

import zipfile
from pathlib import Path

from scripts.make_colab_zip import (
    DATA_FILES,
    E2C1_FILES,
    REPO,
    build_zip,
    main,
    missing_e2c1_files,
    verify_zip_members,
)

NOTEBOOK = "notebooks/colab_generate_e2c1.ipynb"


def _write(root: Path, rel: str, text: str = "stub\n") -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _seed_e2c1(root: Path) -> None:
    for rel in E2C1_FILES:
        _write(root, rel, "{}\n")
    _write(root, NOTEBOOK, "{}\n")
    _write(root, "ml/__init__.py")
    _write(root, "ml/e2c1/__init__.py")


def _names(path: Path) -> set[str]:
    with zipfile.ZipFile(path) as zf:
        return set(zf.namelist())


def test_default_build_excludes_raw_canonical_binaries_and_evidence(tmp_path):
    root = tmp_path / "repo"
    _seed_e2c1(root)
    _write(root, "datasets/raw/pile.jsonl")
    _write(root, "datasets/canonical/hc3.jsonl")
    _write(root, "E1/ckpt/last.pt")
    _write(root, "research/_evidence/wikipedia.jsonl")
    _write(root, "research/_evidence/summary.csv")

    out, files, size = build_zip(tmp_path / "default.zip", root=root)

    assert out.is_file()
    assert files > 0
    assert size > 0
    names = _names(out)
    assert "StopTheCap/datasets/raw/pile.jsonl" not in names
    assert "StopTheCap/datasets/canonical/hc3.jsonl" not in names
    assert "StopTheCap/E1/ckpt/last.pt" not in names
    assert "StopTheCap/research/_evidence/wikipedia.jsonl" not in names
    assert "StopTheCap/research/_evidence/summary.csv" not in names
    assert "StopTheCap/ml/__init__.py" in names


def test_with_e2c1_includes_five_files_and_notebook(tmp_path, capsys):
    root = tmp_path / "repo"
    _seed_e2c1(root)
    out = tmp_path / "e2c1.zip"

    rc = main(["--out", str(out), "--with-e2c1"], root=root)

    assert rc == 0
    names = _names(out)
    for rel in E2C1_FILES:
        assert f"StopTheCap/{rel}" in names, rel
    assert f"StopTheCap/{NOTEBOOK}" in names
    assert "E2-C1 artefacts verified" in capsys.readouterr().out


def test_with_e2c1_missing_file_exits_two_and_names_it(tmp_path, capsys):
    root = tmp_path / "repo"
    _seed_e2c1(root)
    (root / "datasets/manifests/e2_c1_splits.json").unlink()
    out = tmp_path / "e2c1.zip"

    rc = main(["--out", str(out), "--with-e2c1"], root=root)

    captured = capsys.readouterr()
    assert rc == 2
    assert "datasets/manifests/e2_c1_splits.json" in captured.err
    assert not out.exists(), "must fail before writing a ZIP"


def test_missing_e2c1_files_helper_lists_every_gap(tmp_path):
    root = tmp_path / "repo"
    # Only one of the five required inputs exists.
    _write(root, E2C1_FILES[0], "{}\n")

    missing = missing_e2c1_files(root)

    assert missing == list(E2C1_FILES[1:])


def test_verifier_rejects_forbidden_raw_and_canonical_paths():
    crafted = [
        "StopTheCap/datasets/raw/e2c1_generated/samples.jsonl",
        "StopTheCap/datasets/canonical/hc3.jsonl",
        "StopTheCap/ml/e2c1/generation_config.json",
    ]

    problems = verify_zip_members(crafted, required=[])

    assert any("datasets/raw" in problem for problem in problems), problems
    assert any("datasets/canonical" in problem for problem in problems), problems


def test_verifier_rejects_forbidden_suffixes():
    crafted = [
        "StopTheCap/E1/ckpt/last.pt",
        "StopTheCap/model.safetensors",
        "StopTheCap/checkpoints/epoch.ckpt",
        "StopTheCap/StopTheCap.zip",
    ]

    problems = verify_zip_members(crafted, required=[])

    assert len(problems) == len(crafted), problems


def test_verifier_reports_missing_required_member():
    problems = verify_zip_members(["StopTheCap/ml/__init__.py"], required=E2C1_FILES)

    assert len(problems) == len(E2C1_FILES)
    assert problems[0].startswith("missing required member: StopTheCap/")


def test_e2c1_files_are_not_in_the_data_allowlist():
    assert set(E2C1_FILES).isdisjoint(DATA_FILES)


def test_default_output_is_under_the_given_root_not_the_real_repo(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    _seed_e2c1(root)

    repo_zip = REPO / "StopTheCap.zip"
    before = (repo_zip.exists(), repo_zip.stat().st_mtime_ns if repo_zip.exists() else None)

    rc = main(["--with-e2c1"], root=root)

    assert rc == 0
    assert (root / "StopTheCap.zip").is_file()
    after = (repo_zip.exists(), repo_zip.stat().st_mtime_ns if repo_zip.exists() else None)
    assert before == after, "the real repo's default ZIP must not be written during tests"
