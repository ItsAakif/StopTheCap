"""The notebook's ZIP loader, tested against real ZIPs.

The loader lives inside the notebook because it must run *before* the project exists on
sys.path, so it cannot be imported. This test extracts the function definitions from the
notebook cell and exercises them directly - which is the only way to know the flattening and
validation logic actually works before it runs on a paid GPU session.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "colab_measure.ipynb"


def _loader_namespace(tmp_path: Path) -> dict:
    d = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cell = next(("".join(c["source"]) for c in d["cells"]
                 if c["cell_type"] == "code" and "def load_from_zip" in "".join(c["source"])), None)
    assert cell, "loader cell not found in notebook"
    prefix = cell.split("if LOAD_METHOD ==")[0]
    ns: dict = {"__name__": "loader_test"}
    exec(compile(prefix, "<loader>", "exec"), ns)
    ns["EXTRACT_TMP"] = str(tmp_path / "_extract")
    return ns


def _make_zip(path: Path, prefix: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as zf:
        for rel in ("ml/__init__.py", "ml/cloud/__init__.py", "ml/cloud/probe.py",
                    "datasets/manifests/COMBINED.yaml"):
            zf.writestr(f"{prefix}{rel}", "# stub\n")
    return path


@pytest.fixture(scope="module")
def nb_exists() -> bool:
    return NOTEBOOK.is_file()


def test_loads_nested_zip_and_flattens(tmp_path, nb_exists):
    if not nb_exists:
        pytest.skip("notebook missing")
    ns = _loader_namespace(tmp_path)
    z = _make_zip(tmp_path / "nested.zip", "StopTheCap/")
    dest = ns["load_from_zip"](z, tmp_path / "proj")
    assert (dest / "ml/cloud/probe.py").is_file()
    assert not (dest / "StopTheCap").exists(), "must not create a nested duplicate directory"


def test_loads_flat_zip(tmp_path, nb_exists):
    if not nb_exists:
        pytest.skip("notebook missing")
    ns = _loader_namespace(tmp_path)
    z = _make_zip(tmp_path / "flat.zip", "")
    dest = ns["load_from_zip"](z, tmp_path / "proj")
    assert (dest / "ml/__init__.py").is_file()


def test_missing_zip_raises_clear_error(tmp_path, nb_exists):
    if not nb_exists:
        pytest.skip("notebook missing")
    ns = _loader_namespace(tmp_path)
    with pytest.raises(RuntimeError) as exc:
        ns["load_from_zip"](tmp_path / "nope.zip", tmp_path / "proj")
    assert "ZIP not found" in str(exc.value)


def test_wrong_structure_is_rejected(tmp_path, nb_exists):
    if not nb_exists:
        pytest.skip("notebook missing")
    ns = _loader_namespace(tmp_path)
    z = tmp_path / "wrong.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("readme.txt", "no code here")
    dest = ns["load_from_zip"](z, tmp_path / "proj")
    missing = [f for f in ns["REQUIRED"] if not (Path(dest) / f).is_file()]
    assert missing, "structure check should find missing files"


def test_reload_replaces_existing_directory(tmp_path, nb_exists):
    if not nb_exists:
        pytest.skip("notebook missing")
    ns = _loader_namespace(tmp_path)
    z = _make_zip(tmp_path / "a.zip", "")
    dest = ns["load_from_zip"](z, tmp_path / "proj")
    (Path(dest) / "stale.txt").write_text("old")
    ns["load_from_zip"](z, tmp_path / "proj")
    assert not (Path(dest) / "stale.txt").exists()


def test_no_public_url_required(nb_exists):
    if not nb_exists:
        pytest.skip("notebook missing")
    text = NOTEBOOK.read_text(encoding="utf-8")
    assert "github.com/" not in text


def _all_source() -> str:
    import json
    d = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return "\n".join("".join(c["source"]) for c in d["cells"])


def test_loader_cell_survives_a_skipped_config_cell():
    """Regression: switching runtime type restarts the kernel and wipes LOAD_METHOD/ZIP_PATH.
    The loader must fall back to defaults instead of raising NameError."""
    text = _all_source()
    assert "had not run" in text
    assert 'if "LOAD_METHOD" not in globals()' in text
