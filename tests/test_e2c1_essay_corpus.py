"""Offline tests for the E2-C1 length-matched essay corpus builder.

Fixture tests use only synthetic data under ``tmp_path``.  One integration test
rebuilds the published corpus from the real (gitignored) inputs when they are
present; it is skipped otherwise.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from ml.e2c1.build_essay_corpus import (
    EXPECTED_SHA256,
    ai_sample_id,
    human_sample_id,
    main,
)

REPO = Path(__file__).resolve().parents[1]

REAL_PROMPTS = REPO / "datasets" / "manifests" / "e2c1_essay_prompts.jsonl"
REAL_HUMAN = REPO / "datasets" / "canonical" / "ghostbuster.jsonl"
REAL_AI_ROOT = REPO / "datasets" / "raw" / "e2c1_essay_ai"

PROMPT_ONE = "essay:aaaa000000000001"
PROMPT_TWO = "essay:bbbb000000000002"
GROUP_ONE = "ghostbuster:student_essay:1"
GROUP_TWO = "ghostbuster:student_essay:2"


# ---------------------------------------------------------------------------
# Fixtures.
# ---------------------------------------------------------------------------


def _write_jsonl(path: Path, rows: list[dict], *, encoding: str = "utf-8") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding=encoding,
    )
    return path


def _prompt_rows() -> list[dict]:
    return [
        {
            "prompt_id": PROMPT_ONE,
            "human_sample_id": "human-1",
            "partition": "train",
            "group_id": GROUP_ONE,
        },
        {
            "prompt_id": PROMPT_TWO,
            "human_sample_id": "human-2",
            "partition": "val",
            "group_id": GROUP_TWO,
        },
    ]


def _ghostbuster_rows() -> list[dict]:
    return [
        {
            "sample_id": "human-1",
            "label": 0,
            "domain": "student_essay",
            "text": "Human essay one.\n\nA second paragraph.",
        },
        {
            "sample_id": "human-2",
            "label": 0,
            "domain": "student_essay",
            "text": "Human essay two.",
        },
        # Wrong domain: must not resolve as a human essay.
        {
            "sample_id": "human-1",
            "label": 0,
            "domain": "other",
            "text": "Not a student essay.",
        },
        # AI-labelled student essay: must not resolve either.
        {
            "sample_id": "human-2",
            "label": 1,
            "domain": "student_essay",
            "text": "AI-labelled essay.",
        },
    ]


def _ai_record(prompt_id: str, generator_id: str, text: str) -> dict:
    return {"prompt_id": prompt_id, "generator_id": generator_id, "text": text}


def _build_fixture(root: Path) -> dict[str, Path]:
    prompts = _write_jsonl(root / "prompts.jsonl", _prompt_rows())
    human = _write_jsonl(root / "ghostbuster.jsonl", _ghostbuster_rows())

    deepseek = root / "ai" / "deepseek_v41_flash" / "shards"
    muse = root / "ai" / "muse_spark_1_3" / "shards"

    # Valid deepseek shard: UTF-8 BOM, a trailing blank line, two records.
    deepseek.mkdir(parents=True, exist_ok=True)
    valid = (
        json.dumps(_ai_record(PROMPT_ONE, "deepseek_v41_flash", "DeepSeek one."))
        + "\n"
        + json.dumps(_ai_record(PROMPT_TWO, "deepseek_v41_flash", "DeepSeek two."))
        + "\n\n"
    )
    (deepseek / "d000.jsonl").write_text(valid, encoding="utf-8-sig")

    # Ignored deepseek files.  Their generator id must never reach the corpus.
    _write_jsonl(
        deepseek / "d001_failures.jsonl",
        [_ai_record(PROMPT_ONE, "ghost_gen", "Failure record.")],
    )
    _write_jsonl(
        deepseek / "_ignored.jsonl",
        [_ai_record(PROMPT_ONE, "ghost_gen", "Ignored record.")],
    )

    muse.mkdir(parents=True, exist_ok=True)
    _write_jsonl(
        muse / "m000.jsonl",
        [_ai_record(PROMPT_ONE, "muse_spark_1_3", "Muse one.")],
    )
    _write_jsonl(
        muse / "tmp_extra.jsonl",
        [_ai_record(PROMPT_ONE, "ghost_gen", "Temporary record.")],
    )

    return {
        "prompts": prompts,
        "human": human,
        "ai_root": root / "ai",
        "out": root / "corpus.jsonl",
        "manifest": root / "manifest.json",
    }


def _run_builder(fixture: dict[str, Path], *, extra: list[str] | None = None) -> int:
    argv = [
        "--prompts",
        str(fixture["prompts"]),
        "--human",
        str(fixture["human"]),
        "--ai-root",
        str(fixture["ai_root"]),
        "--out",
        str(fixture["out"]),
        "--manifest",
        str(fixture["manifest"]),
    ]
    if extra:
        argv.extend(extra)
    else:
        # Tiny fixtures cannot match the published digest; disable the check.
        argv.extend(["--expect-sha256", ""])
    return main(argv)


def _read_corpus(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


# ---------------------------------------------------------------------------
# Fixture-based unit tests.
# ---------------------------------------------------------------------------


def test_builder_emits_expected_rows(tmp_path: Path) -> None:
    fixture = _build_fixture(tmp_path)

    assert _run_builder(fixture) == 0
    rows = _read_corpus(fixture["out"])

    assert len(rows) == 5
    assert [row["sample_id"] for row in rows] == sorted(
        row["sample_id"] for row in rows
    )

    human_rows = [row for row in rows if row["label"] == 0]
    ai_rows = [row for row in rows if row["label"] == 1]
    assert len(human_rows) == 2
    assert len(ai_rows) == 3

    assert {row["sample_id"] for row in human_rows} == {
        human_sample_id(PROMPT_ONE),
        human_sample_id(PROMPT_TWO),
    }
    assert all(row["sample_id"].startswith("essayh:") for row in human_rows)
    assert all(row["sample_id"].startswith("essaya:") for row in ai_rows)
    assert all(row["generator"] is None for row in human_rows)
    assert {row["generator"] for row in ai_rows} == {
        "deepseek_v41_flash",
        "muse_spark_1_3",
    }
    assert "ghost_gen" not in {row["generator"] for row in rows}

    assert {row["sample_id"] for row in ai_rows} == {
        ai_sample_id(PROMPT_ONE, "deepseek_v41_flash"),
        ai_sample_id(PROMPT_TWO, "deepseek_v41_flash"),
        ai_sample_id(PROMPT_ONE, "muse_spark_1_3"),
    }

    human_one = next(
        row for row in human_rows if row["sample_id"] == human_sample_id(PROMPT_ONE)
    )
    assert human_one["text"] == "Human essay one.\n\nA second paragraph."
    assert human_one["partition"] == "train"
    assert human_one["group_id"] == GROUP_ONE
    assert human_one["source_dataset"] == "ghostbuster_student_essay"
    assert human_one["topic_category"] == "student_essay"


def test_builder_manifest_counts(tmp_path: Path) -> None:
    fixture = _build_fixture(tmp_path)

    assert _run_builder(fixture) == 0
    manifest = json.loads(fixture["manifest"].read_text(encoding="utf-8"))

    assert manifest["schema_version"] == "1.0"
    assert manifest["corpus"] == "e2c1_essays"
    assert manifest["created_at"] == "2026-11-12"
    assert manifest["gate"] == {
        "status": "FAIL",
        "reason": "newlines_per_100w AUC 0.901; digits_per_100w AUC 0.097",
        "word_count_auc": 0.486,
    }
    assert manifest["notes"] == (
        "AI essays generated to match human essay lengths; formatting/digit "
        "tells accepted per project decision."
    )

    counts = manifest["counts"]
    assert counts["total"] == 5
    assert counts["by_label"] == {"1": 3, "0": 2}
    assert counts["by_partition"] == {"train": 3, "val": 2}
    assert counts["by_partition_label"] == {
        "train:1": 2,
        "val:1": 1,
        "train:0": 1,
        "val:0": 1,
    }
    assert counts["by_generator"] == {
        "deepseek_v41_flash": 2,
        "muse_spark_1_3": 1,
    }

    corpus_bytes = fixture["out"].read_bytes()
    assert manifest["corpus_sha256"] == hashlib.sha256(corpus_bytes).hexdigest()


def test_expect_sha256_mismatch_exits_nonzero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fixture = _build_fixture(tmp_path)

    result = _run_builder(fixture, extra=["--expect-sha256", "0" * 64])

    assert result != 0
    assert "sha256 mismatch" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Integration test against the real (gitignored) inputs.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not (REAL_PROMPTS.is_file() and REAL_HUMAN.is_file() and REAL_AI_ROOT.is_dir()),
    reason="real E2-C1 essay inputs are not present",
)
def test_real_inputs_reproduce_published_corpus(tmp_path: Path) -> None:
    out = tmp_path / "corpus.jsonl"
    manifest = tmp_path / "manifest.json"

    result = main(
        [
            "--prompts",
            str(REAL_PROMPTS),
            "--human",
            str(REAL_HUMAN),
            "--ai-root",
            str(REAL_AI_ROOT),
            "--out",
            str(out),
            "--manifest",
            str(manifest),
        ]
    )

    assert result == 0
    assert hashlib.sha256(out.read_bytes()).hexdigest() == (
        "8b0c0cc3beebe024302c0e87b75995a71378b30ffa3c7605b5a11f8e6a75bf2c"
    )
    assert len(_read_corpus(out)) == 2178

    written_manifest = json.loads(manifest.read_text(encoding="utf-8"))
    assert written_manifest["corpus_sha256"] == EXPECTED_SHA256
    assert written_manifest["counts"]["total"] == 2178
