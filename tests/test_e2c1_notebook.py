"""Static, offline checks for the E2-C1 Colab notebook (§E, T12).

No cell is executed, no GPU/CUDA/torch/network is touched. The loader cell is the one
exception-by-technique: its function definitions are extracted and exercised against a
synthetic ZIP, exactly as ``tests/test_notebook_loader.py`` does for the E1 notebook.
"""

from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path

import pytest

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "colab_generate_e2c1.ipynb"

E2C1_FILES = (
    "datasets/manifests/e2_c1_prompts.jsonl",
    "datasets/manifests/e2_c1_prompt_audit.json",
    "datasets/manifests/e2_c1_splits.json",
    "datasets/manifests/e2_c1_licences.json",
    "ml/e2c1/generation_config.json",
)


@pytest.fixture(scope="module")
def notebook() -> dict:
    assert NOTEBOOK.is_file(), f"notebook missing: {NOTEBOOK}"
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _source(cell: dict) -> str:
    return "".join(cell["source"])


def _all_source(nb: dict) -> str:
    return "\n".join(_source(cell) for cell in nb["cells"])


def _code_source(nb: dict) -> str:
    return "\n".join(_source(cell) for cell in nb["cells"] if cell["cell_type"] == "code")


def _loader_cell(nb: dict) -> str:
    for cell in nb["cells"]:
        if cell["cell_type"] == "code" and "def load_from_zip" in _source(cell):
            return _source(cell)
    raise AssertionError("loader cell not found in notebook")


def _loader_namespace(tmp_path: Path) -> dict:
    cell = _loader_cell(json.loads(NOTEBOOK.read_text(encoding="utf-8")))
    prefix = cell.split("if LOAD_METHOD ==")[0]
    ns: dict = {"__name__": "loader_test"}
    exec(compile(prefix, "<loader>", "exec"), ns)
    ns["EXTRACT_TMP"] = str(tmp_path / "_extract")
    return ns


def _make_zip(path: Path, prefix: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as zf:
        for rel in ("ml/__init__.py", "ml/e2c1/generation_config.json",
                    "datasets/manifests/e2_c1_prompts.jsonl"):
            zf.writestr(f"{prefix}{rel}", "stub\n")
    return path


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------


def test_valid_json_nbformat_4_and_21_cells(notebook):
    assert notebook["nbformat"] == 4
    assert notebook["nbformat_minor"] == 5
    assert len(notebook["cells"]) == 21


def test_first_cell_is_markdown_and_last_cell_is_code(notebook):
    assert notebook["cells"][0]["cell_type"] == "markdown"
    assert notebook["cells"][-1]["cell_type"] == "code"


def test_required_strings_are_present(notebook):
    text = _all_source(notebook)

    for token in (
        "5000",
        "split_digest",
        "HF_TOKEN",
        "AUTHORISATION REQUIRED",
        "mode",
        "e2_c1_generation.json",
    ):
        assert token in text, token


# ---------------------------------------------------------------------------
# Cell 2: configuration
# ---------------------------------------------------------------------------


def test_config_cell_sets_paths_counter_and_authorisation(notebook):
    source = _source(notebook["cells"][2])

    for assignment in (
        "ZIP_PATH =",
        "REPO_DIR =",
        "RUN_DIR =",
        "SMOKE_OUT =",
        "SMOKE_PER_GENERATOR = 4",
        "AUTHORISED = False",
        'HF_TOKEN = os.environ.get("HF_TOKEN")',
    ):
        assert assignment in source, assignment
    assert "hf_" not in source


# ---------------------------------------------------------------------------
# Cell 6: ZIP loader
# ---------------------------------------------------------------------------


def test_loader_cell_mentions_zipfile_extract_tmp_and_required(notebook):
    cell = _loader_cell(notebook)

    assert "zipfile" in cell
    assert "EXTRACT_TMP" in cell
    assert "REQUIRED" in cell
    assert 'if "LOAD_METHOD" not in globals()' in cell


def test_loader_cell_is_extractable_and_flattens_a_nested_zip(tmp_path):
    ns = _loader_namespace(tmp_path)
    zip_path = _make_zip(tmp_path / "nested.zip", "StopTheCap/")

    dest = ns["load_from_zip"](zip_path, tmp_path / "proj")

    assert (Path(dest) / "ml/e2c1/generation_config.json").is_file()
    assert not (Path(dest) / "StopTheCap").exists()


def test_loader_required_includes_e2c1_files_and_modules(tmp_path):
    ns = _loader_namespace(tmp_path)
    required = set(ns["REQUIRED"])

    assert set(E2C1_FILES) <= required
    for module in (
        "ml/e2c1/__init__.py",
        "ml/e2c1/constants.py",
        "ml/e2c1/prompts.py",
        "ml/e2c1/families.py",
        "ml/e2c1/split.py",
        "ml/e2c1/generation.py",
    ):
        assert module in required, module


# ---------------------------------------------------------------------------
# Cell 8: dependencies
# ---------------------------------------------------------------------------


def test_dependency_cell_installs_and_imports_every_required_package(notebook):
    source = _source(notebook["cells"][8])
    pip_line = next(line for line in source.splitlines() if "%pip install" in line)

    for package in (
        "transformers",
        "accelerate",
        "bitsandbytes",
        "sentencepiece",
        "protobuf",
        "pydantic",
        "PyYAML",
    ):
        assert package in pip_line, package
    for module in (
        "import transformers",
        "import accelerate",
        "import bitsandbytes",
        "import sentencepiece",
        "import google.protobuf",
        "import pydantic",
        "import yaml",
    ):
        assert module in source, module


# ---------------------------------------------------------------------------
# Cell 10: hardware pre-flight
# ---------------------------------------------------------------------------


def test_hardware_cell_asserts_cuda_and_20000_plan(notebook):
    source = _source(notebook["cells"][10])

    assert "torch.cuda.is_available()" in source
    assert "total_memory >= 14.5e9" in source
    assert "config_digest" in source
    assert "5000" in source
    assert "20000" in source
    assert "generator.architecture" in source
    assert "generator.loader" in source


# ---------------------------------------------------------------------------
# Cell 12: licence validation
# ---------------------------------------------------------------------------


def test_licence_cell_is_generic_and_asserts_outputs_licensed(notebook):
    source = _source(notebook["cells"][12])

    assert "validate_config_against_licences" in source
    assert "outputs_licensed is False" in source
    # The table is registry-driven, not a hardcoded family rule.
    assert "llama" not in source.lower()
    assert "not_apache_2_0" not in source
    for field in (
        "generator.generator_id",
        "generator.model_id",
        "generator.model_revision",
        "licence.get('spdx'",
        "entry.get('status'",
        "entry.get('gated'",
        "outputs_licensed",
    ):
        assert field in source, field


# ---------------------------------------------------------------------------
# Cell 14: prompt + split validation
# ---------------------------------------------------------------------------


def test_prompt_split_cell_recomputes_sha256_and_uses_manifest_helpers(notebook):
    source = _source(notebook["cells"][14])

    assert "read_manifest" in source
    assert "validate_manifest" in source
    assert "validate_split_manifest" in source
    assert "read_bytes()" in source
    assert "prompt_manifest_sha256" in source
    assert "by_partition_generator_decoding_topic" in source


# ---------------------------------------------------------------------------
# Cell 16: credentials and gate check
# ---------------------------------------------------------------------------


def test_gate_cell_fails_closed_before_any_model_load(notebook):
    source = _source(notebook["cells"][16])

    assert "GatedRepoError" in source
    assert "401" in source and "403" in source
    assert '"status": "blocked"' in source
    assert "corpus_written" in source
    assert "raise RuntimeError" in source
    assert "model_info" in source
    # The blocked report is written before any model weights are loaded.
    assert source.index('"status": "blocked"') < source.index("raise RuntimeError")


def test_gate_cell_makes_token_optional_for_ungated_models(notebook):
    source = _source(notebook["cells"][16])

    # The token requirement is derived from the registry, not assumed.
    assert "needs_token = any(" in source
    assert 'entry.get("gated") is True' in source
    assert "token required:" in source
    assert "resolved = [" in source

    # Ungated configs proceed without a token and cannot raise.
    assert "if not needs_token:" in source
    assert "No gated models in the generation config; HF token is optional." in source
    ungated = source[source.index("if not needs_token:"):source.index("api = HfApi()")]
    assert "_blocked_report" not in ungated
    assert "raise " not in ungated

    # Gated configs without a token keep the blocked-report hard stop.
    assert "if needs_token and not token:" in source
    gated = source[
        source.index("if needs_token and not token:"):source.index("if not needs_token:")
    ]
    assert "_blocked_report(" in gated
    assert "_atomic_write_report(REPORT_PATH, payload)" in gated
    assert "raise RuntimeError" in gated

    # Model resolution is attempted for every model, with the (possibly absent) token.
    assert "model_info(" in source
    assert "token=token" in source


# ---------------------------------------------------------------------------
# Cell 18: smoke subprocess
# ---------------------------------------------------------------------------


def test_smoke_cell_runs_subprocess_and_asserts_16_attempts(notebook):
    source = _source(notebook["cells"][18])

    assert 'sys.executable, "-m", "ml.e2c1.generate"' in source
    assert '"--config"' in source
    assert '"--prompts"' in source
    assert '"--splits"' in source
    assert '"--out"' in source
    assert '"--smoke"' in source
    assert '"--report"' in source
    assert "attempts == 4 * 2 * 2" in source
    assert "records + failures == attempts" in source


# ---------------------------------------------------------------------------
# Cell 20: report + hard stop
# ---------------------------------------------------------------------------


def test_final_cell_writes_report_atomically_and_raises_system_exit(notebook):
    source = _source(notebook["cells"][-1])

    assert "os.replace" in source
    assert "os.fsync" in source
    assert "SystemExit" in source
    assert "AUTHORISATION REQUIRED - smoke only" in source
    assert "corpus_written" in source


def test_report_payload_contains_every_required_field(notebook):
    source = _source(notebook["cells"][-1])

    for field in (
        '"mode"',
        '"status"',
        '"config_digest"',
        '"prompt_manifest_sha256"',
        '"split_digest"',
        '"gpu"',
        '"versions"',
        '"licences"',
        '"quantisation"',
        '"smoke"',
        '"timings"',
        '"eta_full_hours"',
        '"full_run"',
        '"corpus_written"',
        '"next_action"',
        '"authorised": False',
        '"requested_n": 20000',
    ):
        assert field in source, field


# ---------------------------------------------------------------------------
# Prohibitions
# ---------------------------------------------------------------------------


def test_no_public_url_and_no_token_literal(notebook):
    text = _all_source(notebook)

    assert "github.com/" not in text
    assert "hf_" not in text


def test_no_full_generation_loop(notebook):
    text = _all_source(notebook)

    assert "range(20000)" not in text
    assert "range(20_000)" not in text
    assert not re.search(r"for\s+\w+\s+in\s+range\(5000\)", text)
    assert not re.search(r"for\s+\w+\s+in\s+range\(20_000\)", text)


def test_corpus_written_is_false_and_no_cell_writes_under_datasets(notebook):
    code = _code_source(notebook)

    assert re.search(r'["\']corpus_written["\']\s*:\s*False', code)
    assert 'SMOKE_OUT = "/content/e2c1_smoke/out"' in code
    assert 'REPORT_PATH = "/content/drive/MyDrive/StopTheCap_reports/e2_c1_generation.json"' in code

    write_tokens = ("write_text", "write_bytes", ".mkdir", "os.replace", "open(")
    for line in code.splitlines():
        if "datasets/" in line:
            assert not any(token in line for token in write_tokens), (
                f"cell line appears to write under datasets/: {line}"
            )


def test_authorisation_gate_is_asserted_false_with_a_separate_run_message(notebook):
    code = _code_source(notebook)

    assert "AUTHORISED = False" in code
    assert "assert AUTHORISED is False" in code
    assert "separate, separately authorised notebook/run" in code
