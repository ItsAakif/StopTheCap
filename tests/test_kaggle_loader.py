"""Kaggle loader auto-discovery, tested against a simulated /kaggle/input tree.

The loader lives in the notebook (it must run before the project exists on sys.path), so the
functions are extracted and exercised directly.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "kaggle_measure.ipynb"


def _ns(tmp_path: Path) -> dict:
    d = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cell = next(("".join(c["source"]) for c in d["cells"]
                 if c["cell_type"] == "code" and "def find_source" in "".join(c["source"])), None)
    assert cell, "Kaggle loader cell not found"
    prefix = cell.split("kind, source = find_source")[0]
    ns: dict = {"__name__": "kaggle_loader_test"}
    exec(compile(prefix, "<kaggle_loader>", "exec"), ns)
    ns["EXTRACT_TMP"] = str(tmp_path / "_extract")
    return ns


def _project(root: Path) -> Path:
    for rel in ("ml/__init__.py", "ml/cloud/__init__.py", "ml/cloud/probe.py"):
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("# stub\n")
    return root


@pytest.fixture
def inputdir(tmp_path):
    root = tmp_path / "kaggle_input"
    root.mkdir()
    return root


def test_finds_zip_under_dataset_slug(tmp_path, inputdir):
    ns = _ns(tmp_path)
    ds = inputdir / "stopthecap"
    ds.mkdir()
    (ds / "StopTheCap.zip").write_bytes(b"PK\x05\x06" + b"\x00" * 18)
    kind, src = ns["find_source"](str(inputdir))
    assert kind == "zip" and src.name == "StopTheCap.zip"


def test_finds_already_extracted_project(tmp_path, inputdir):
    ns = _ns(tmp_path)
    ds = inputdir / "stopthecap"
    _project(ds)
    kind, src = ns["find_source"](str(inputdir))
    assert kind == "dir" and (src / "ml/cloud/probe.py").is_file()


def test_finds_nested_project_folder(tmp_path, inputdir):
    ns = _ns(tmp_path)
    ds = inputdir / "stopthecap"
    (ds / "StopTheCap").mkdir(parents=True)
    _project(ds / "StopTheCap")
    kind, src = ns["find_source"](str(inputdir))
    assert kind == "dir" and src.name == "StopTheCap"


def test_ignores_unrelated_datasets(tmp_path, inputdir):
    ns = _ns(tmp_path)
    (inputdir / "other-dataset").mkdir()
    (inputdir / "other-dataset" / "data.csv").write_text("a,b\n")
    (inputdir / "stopthecap").mkdir()
    (inputdir / "stopthecap" / "StopTheCap.zip").write_bytes(b"PK")
    kind, src = ns["find_source"](str(inputdir))
    assert kind == "zip"


def test_missing_project_raises_with_listing(tmp_path, inputdir):
    ns = _ns(tmp_path)
    (inputdir / "other-dataset").mkdir()
    with pytest.raises(RuntimeError) as exc:
        ns["find_source"](str(inputdir))
    assert "other-dataset" in str(exc.value)


def test_missing_input_root_raises(tmp_path):
    ns = _ns(tmp_path)
    with pytest.raises(RuntimeError) as exc:
        ns["find_source"](str(tmp_path / "does_not_exist"))
    assert "Input panel" in str(exc.value)


def test_zip_flatten_and_structure_check(tmp_path, inputdir):
    import zipfile

    ns = _ns(tmp_path)
    ds = inputdir / "stopthecap"
    ds.mkdir()
    z = ds / "StopTheCap.zip"
    with zipfile.ZipFile(z, "w") as zf:
        for rel in ("ml/__init__.py", "ml/cloud/__init__.py", "ml/cloud/probe.py"):
            zf.writestr("StopTheCap/" + rel, "# stub\n")
    dest = ns["extract_zip"](z, tmp_path / "proj")
    assert (Path(dest) / "ml/cloud/probe.py").is_file()
    assert not (Path(dest) / "StopTheCap").exists()


def _all_source() -> str:
    import json
    d = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return "\n".join("".join(c["source"]) for c in d["cells"])


def test_finder_survives_a_skipped_config_cell():
    text = _all_source()
    assert "had not run" in text
    assert 'if "REPO_DIR" not in globals()' in text
