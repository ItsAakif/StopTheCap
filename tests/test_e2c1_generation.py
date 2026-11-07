"""E2-C1 generation contract: torch-free import, schema, rejection, resume, CLI.

Every test is offline: no GPU, no network, no real prompt/split manifest and no
``datasets/`` write. Fixtures are synthetic and live in ``tmp_path``.
"""

from __future__ import annotations

import copy
import hashlib
import json
import random
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from ml.e2c1 import generate
from ml.e2c1.constants import SPLIT_SALT
from ml.e2c1.generation import (
    GenerationConfig,
    GenerationFailure,
    GenerationRecord,
    LicenceValidationError,
    PromptRejection,
    ResumeCorruptionError,
    ResumeMismatch,
    append_record,
    config_digest,
    load_config,
    load_state,
    load_stream,
    make_sample_id,
    postprocess,
    render_prompt,
    run_generation,
    save_state,
    select_smoke_prompts,
    validate_config_against_licences,
)
from ml.e2c1.ids import make_sample_id as ids_make_sample_id
from ml.e2c1.split import e2c1_split_digest

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "ml" / "e2c1" / "generation_config.json"
REGISTRY_PATH = REPO_ROOT / "datasets" / "manifests" / "e2_c1_licences.json"


# ---------------------------------------------------------------------------
# Fixtures and builders
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def config() -> GenerationConfig:
    return load_config(CONFIG_PATH)


@pytest.fixture(scope="module")
def registry() -> dict:
    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def passage(n: int = 60, prefix: str = "passageword") -> str:
    return " ".join(f"{prefix}{index}" for index in range(n))


def source_text(n: int = 200, prefix: str = "sourceword") -> str:
    return " ".join(f"{prefix}{index}" for index in range(n))


def make_prompt(index: int, *, partition: str = "train") -> dict:
    text = passage(60, prefix=f"prompt{index}word")
    return {
        "prompt_id": f"pmt:{index:024x}",
        "prompt_text": text,
        "prompt_hash": sha(text),
        "topic_category": "other",
        "partition": partition,
    }


def make_prompt_rows(n: int) -> list[dict]:
    return [make_prompt(index) for index in range(n)]


def write_prompt_manifest(path: Path, rows: list[dict]) -> str:
    payload = "".join(
        json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows
    )
    data = payload.encode("utf-8")
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def write_split_manifest(
    path: Path,
    rows: list[dict],
    assignments: dict[str, str],
    prompt_sha: str,
) -> dict:
    split = {
        "schema_version": "1.0",
        "salt": SPLIT_SALT,
        "prompt_manifest_sha256": prompt_sha,
        "assignments": {key: assignments[key] for key in sorted(assignments)},
    }
    split["split_digest"] = e2c1_split_digest(split["assignments"], SPLIT_SALT)
    path.write_text(json.dumps(split), encoding="utf-8")
    return split


def make_record(
    config: GenerationConfig,
    prompt: dict,
    *,
    text: str | None = None,
    finish_reason: str = "eos",
    decoding_index: int = 0,
    generator_index: int = 0,
) -> GenerationRecord:
    generator = config.generators[generator_index]
    decoding = config.decodings[decoding_index]
    body = text if text is not None else passage(60)
    return GenerationRecord(
        sample_id=make_sample_id(
            prompt["prompt_id"],
            generator.generator_id,
            generator.model_revision,
            decoding.decoding_config_id,
        ),
        prompt_id=prompt["prompt_id"],
        generator_id=generator.generator_id,
        model_id=generator.model_id,
        model_revision=generator.model_revision,
        licence_ref=generator.generator_id,
        decoding_config_id=decoding.decoding_config_id,
        temperature=decoding.temperature,
        top_p=decoding.top_p,
        max_new_tokens=decoding.max_new_tokens,
        seed=12345,
        batch_index=0,
        do_sample=decoding.do_sample,
        generated_text=body,
        generated_text_hash=sha(body),
        generation_timestamp="2026-11-06T00:00:00Z",
        partition="train",
        topic_category=prompt["topic_category"],
        source_prompt_hash=prompt["prompt_hash"],
        quantisation=generator.quantisation.model_dump(mode="json"),
        prompt_template_id=config.prompt_template_id,
        chat_template_sha256=sha("template"),
        config_digest=config_digest(config),
        finish_reason=finish_reason,
        quality_flags=[],
    )


def _environment() -> dict[str, str]:
    import os

    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    return env


# ---------------------------------------------------------------------------
# Import safety / CLI help
# ---------------------------------------------------------------------------


def test_import_works_with_torch_blocked():
    code = (
        "import sys; sys.modules['torch']=None; "
        "import ml.e2c1.generation; print('ok')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        env=_environment(),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def test_help_works_without_torch():
    result = subprocess.run(
        [sys.executable, "-m", "ml.e2c1.generate", "--help"],
        cwd=REPO_ROOT,
        env=_environment(),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert "usage" in result.stdout.lower()
    assert "--authorise-full-run" in result.stdout
    assert "torch" not in result.stderr.lower()


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def test_config_digest_is_stable_and_canonical(config):
    first = config_digest(config)
    second = config_digest(json.loads(CONFIG_PATH.read_text(encoding="utf-8")))
    assert first == second
    assert len(first) == 64
    assert first == config_digest(load_config(CONFIG_PATH))
    # key order must not matter
    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    reordered = dict(reversed(list(payload.items())))
    assert config_digest(reordered) == first


def test_config_unknown_field_rejected(tmp_path):
    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    payload["unexpected_field"] = 1
    path = tmp_path / "bad_config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValidationError):
        load_config(path)
    with pytest.raises(ValidationError):
        GenerationConfig.model_validate(payload)


# ---------------------------------------------------------------------------
# Licence validation
# ---------------------------------------------------------------------------


def test_licence_registry_accepts_the_real_config(config, registry):
    validate_config_against_licences(config, registry)


def test_licence_validation_missing_generator(config, registry):
    bad_registry = copy.deepcopy(registry)
    bad_registry["models"] = [
        entry
        for entry in bad_registry["models"]
        if entry["generator_id"] != "mistral_7b_instruct_v02"
    ]
    with pytest.raises(LicenceValidationError, match="missing generator"):
        validate_config_against_licences(config, bad_registry)


def test_licence_validation_revision_mismatch(config, registry):
    bad_config = config.model_copy(deep=True)
    bad_config.generators[0].model_revision = "f" * 40
    with pytest.raises(LicenceValidationError, match="model_revision"):
        validate_config_against_licences(bad_config, registry)


def test_licence_validation_blocked_status(config, registry):
    bad_registry = copy.deepcopy(registry)
    bad_registry["models"][0]["status"] = "blocked"
    with pytest.raises(LicenceValidationError, match="status"):
        validate_config_against_licences(config, bad_registry)


def test_licence_validation_apache_mislabel(config, registry):
    bad_registry = copy.deepcopy(registry)
    bad_registry["models"][0]["licence"]["spdx"] = "MIT"
    with pytest.raises(LicenceValidationError, match="Apache-2.0"):
        validate_config_against_licences(config, bad_registry)


def test_licence_validation_outputs_licensed(config, registry):
    bad_registry = copy.deepcopy(registry)
    bad_registry["models"][0]["output_terms"]["outputs_licensed"] = True
    with pytest.raises(LicenceValidationError, match="outputs_licensed"):
        validate_config_against_licences(config, bad_registry)


def test_licence_validation_llama_not_apache_flag(config, registry):
    bad_registry = copy.deepcopy(registry)
    for entry in bad_registry["models"]:
        if "llama" in entry["model_id"].lower():
            entry["licence"]["not_apache_2_0"] = False
    with pytest.raises(LicenceValidationError, match="not_apache_2_0"):
        validate_config_against_licences(config, bad_registry)


# ---------------------------------------------------------------------------
# Prompt rendering
# ---------------------------------------------------------------------------


class FakeTokenizer:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        self.calls.append(
            {
                "messages": messages,
                "tokenize": tokenize,
                "add_generation_prompt": add_generation_prompt,
            }
        )
        return "RENDERED"


def test_render_prompt_messages_and_word_range(config):
    tokenizer = FakeTokenizer()
    result = render_prompt("The source passage.", tokenizer, config)
    assert result == "RENDERED"
    assert len(tokenizer.calls) == 1
    call = tokenizer.calls[0]
    assert call["tokenize"] is False
    assert call["add_generation_prompt"] is True
    assert [message["role"] for message in call["messages"]] == ["system", "user"]

    system = call["messages"][0]["content"]
    user = call["messages"][1]["content"]
    assert system == (
        "You are a careful encyclopedic writer. You never copy source sentences."
    )
    assert "<<<SOURCE" in user and "SOURCE>>>" in user
    assert "The source passage." in user
    assert "100-150 words" in user
    assert "120-180 words" not in user


def test_render_prompt_rejects_delimiter_collision(config):
    tokenizer = FakeTokenizer()
    for bad in ("before <<<SOURCE after", "before SOURCE>>> after"):
        with pytest.raises(PromptRejection, match="prompt_delimiter_collision"):
            render_prompt(bad, tokenizer, config)
    assert tokenizer.calls == []


# ---------------------------------------------------------------------------
# Post-processing / rejection
# ---------------------------------------------------------------------------


def test_postprocess_empty(config):
    text, _flags, reason = postprocess("", source_text(), "eos", config)
    assert text == ""
    assert reason == "empty"


def test_postprocess_short(config):
    _text, _flags, reason = postprocess(passage(10), source_text(), "eos", config)
    assert reason == "length_short"


def test_postprocess_truncated(config):
    _text, _flags, reason = postprocess(passage(60), source_text(), "length", config)
    assert reason == "truncated"


def test_postprocess_refusal(config):
    body = passage(20, "neutral") + " I cannot provide this passage. " + passage(
        30, "moreneutral"
    )
    _text, _flags, reason = postprocess(body, source_text(), "eos", config)
    assert reason == "refusal"


def test_postprocess_echo_by_jaccard(config):
    source = source_text(200)
    generated = " ".join(source.split()[:70])
    _text, _flags, reason = postprocess(generated, source, "eos", config)
    assert reason == "echo"


def test_postprocess_echo_by_containment(config):
    source = source_text(300)
    generated = " ".join(source.split()[:50] + passage(10).split())
    _text, _flags, reason = postprocess(generated, source, "eos", config)
    assert reason == "echo"


def test_postprocess_echo_by_12gram(config):
    source = source_text(300)
    shared_span = " ".join(source.split()[100:112])  # exactly 12 words
    generated = shared_span + " " + passage(60)
    _text, _flags, reason = postprocess(generated, source, "eos", config)
    assert reason == "echo"


def test_postprocess_repetition_by_5gram_ratio(config):
    generated = "alpha beta gamma delta epsilon " * 20
    _text, _flags, reason = postprocess(generated, source_text(), "eos", config)
    assert reason == "repetition"


def test_postprocess_repetition_by_8gram_count(config):
    block = " ".join(f"repeatword{index}" for index in range(8))
    generated = " ".join([block] * 4) + " " + passage(30, "tailword")
    _text, _flags, reason = postprocess(generated, source_text(), "eos", config)
    assert reason == "repetition"


def test_postprocess_non_latin_output(config):
    generated = " ".join(chr(0x4E00 + index) for index in range(60))
    _text, _flags, reason = postprocess(generated, source_text(), "eos", config)
    assert reason == "non_latin_output"


def test_postprocess_strips_wrapped_quotes(config):
    body = '"' + passage(60) + '"'
    text, flags, reason = postprocess(body, source_text(), "eos", config)
    assert reason is None
    assert text == passage(60)
    assert "wrapped_quotes_removed" in flags


def test_postprocess_retains_preamble_flag(config):
    body = "Sure!\n" + passage(60)
    text, flags, reason = postprocess(body, source_text(), "eos", config)
    assert reason is None
    assert "assistant_preamble_removed" in flags
    assert text == passage(60)


# ---------------------------------------------------------------------------
# JSONL append / load / resume
# ---------------------------------------------------------------------------


def test_append_and_load_stream_roundtrip(config, tmp_path):
    prompt = make_prompt(0)
    record = make_record(config, prompt)
    path = tmp_path / "nested" / "samples-a.jsonl"
    append_record(path, record)
    append_record(path, make_record(config, make_prompt(1)))

    records, stats = load_stream(path)
    assert len(records) == 2
    assert stats["samples"] == 2
    assert stats["truncated_partial_lines"] == 0
    assert records[0]["generated_text"] == record.generated_text
    assert records[0]["sample_id"] == record.sample_id
    # one canonical line per record, sorted keys, trailing newline
    raw = path.read_bytes()
    assert raw.endswith(b"\n")
    assert raw.count(b"\n") == 2
    parsed = json.loads(raw.split(b"\n")[0])
    assert list(parsed) == sorted(parsed)


def test_load_stream_truncates_torn_final_line(config, tmp_path):
    path = tmp_path / "samples-torn.jsonl"
    append_record(path, make_record(config, make_prompt(0)))
    with path.open("ab") as handle:
        handle.write(b'{"partial":')
    records, stats = load_stream(path)
    assert len(records) == 1
    assert stats["truncated_partial_lines"] == 1


def test_load_stream_corrupted_hash_raises(config, tmp_path):
    prompt = make_prompt(0)
    record = make_record(config, prompt)
    payload = record.model_dump(mode="json")
    payload["generated_text"] = passage(60, "tampered")
    path = tmp_path / "samples-bad.jsonl"
    path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ResumeCorruptionError, match="generated_text_hash"):
        load_stream(path)


def test_load_stream_duplicate_sample_id_raises(config, tmp_path):
    record = make_record(config, make_prompt(0))
    path = tmp_path / "samples-dupe.jsonl"
    payload = json.dumps(
        record.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
    )
    path.write_text(payload + "\n" + payload + "\n", encoding="utf-8")
    with pytest.raises(ResumeCorruptionError, match="duplicate sample_id"):
        load_stream(path)


def test_failure_records_can_repeat_attempts(config, tmp_path):
    prompt = make_prompt(0)
    common = make_record(config, prompt).model_dump(mode="json")
    failure = {
        key: common[key]
        for key in (
            "sample_id",
            "prompt_id",
            "generator_id",
            "model_id",
            "model_revision",
            "licence_ref",
            "decoding_config_id",
            "temperature",
            "top_p",
            "max_new_tokens",
            "seed",
            "batch_index",
            "do_sample",
            "generation_timestamp",
            "partition",
            "topic_category",
            "source_prompt_hash",
            "quantisation",
            "prompt_template_id",
            "chat_template_sha256",
            "config_digest",
        )
    }
    failure.update(
        status="rejected",
        rejection_reason="length_short",
        attempt=1,
        raw_text_hash=sha("raw"),
        finish_reason="eos",
    )
    path = tmp_path / "failures-a.jsonl"
    append_record(path, GenerationFailure.model_validate(failure))
    failure["attempt"] = 2
    append_record(path, GenerationFailure.model_validate(failure))
    records, stats = load_stream(path)
    assert len(records) == 2
    assert stats["failures"] == 2
    assert [record["attempt"] for record in records] == [1, 2]


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


def _expected_state() -> dict:
    return {
        "config_digest": sha("config"),
        "prompt_manifest_sha256": sha("prompts"),
        "split_digest": sha("split"),
    }


def test_state_save_and_load_roundtrip(tmp_path):
    path = tmp_path / "state.json"
    expected = _expected_state()
    state = {
        **expected,
        "schema_version": "1.0",
        "streams": {
            "g__greedy": {"ok": [], "failed": {}, "last_offset": 0}
        },
    }
    save_state(path, state)
    loaded = load_state(path, expected)
    assert loaded["config_digest"] == expected["config_digest"]
    assert loaded["prompt_manifest_sha256"] == expected["prompt_manifest_sha256"]
    assert loaded["split_digest"] == expected["split_digest"]


def test_state_jsonl_wins_over_stale_state(config, tmp_path):
    expected = _expected_state()
    real_record = make_record(config, make_prompt(0))
    stale_id = make_sample_id(
        "pmt:" + "f" * 24,
        config.generators[0].generator_id,
        config.generators[0].model_revision,
        "greedy",
    )
    stale = {
        **expected,
        "schema_version": "1.0",
        "streams": {
            "mistral_7b_instruct_v02__greedy": {
                "ok": [stale_id],
                "failed": {},
                "last_offset": 0,
            }
        },
    }
    path = tmp_path / "state.json"
    save_state(path, stale)

    actual = {
        "mistral_7b_instruct_v02__greedy": {
            "ok": [real_record.sample_id],
            "failed": {},
            "last_offset": 42,
        }
    }
    loaded = load_state(path, {**expected, "streams": actual})
    stream = loaded["streams"]["mistral_7b_instruct_v02__greedy"]
    assert stream["ok"] == [real_record.sample_id]
    assert stale_id not in stream["ok"]
    assert loaded["state_repaired"] == 1


def test_state_refuses_mismatched_config_digest(tmp_path):
    path = tmp_path / "state.json"
    expected = _expected_state()
    save_state(path, {**expected, "schema_version": "1.0", "streams": {}})
    other = dict(expected)
    other["config_digest"] = sha("different config")
    with pytest.raises(ResumeMismatch, match="config_digest"):
        load_state(path, other)


# ---------------------------------------------------------------------------
# Planning: sample ids, smoke selection, dry run
# ---------------------------------------------------------------------------


def test_sample_id_golden():
    assert make_sample_id(
        "pmt:0123456789abcdef01234567",
        "mistral_7b_instruct_v02",
        "63a8b081895390a26e140280378bc85ec8bce07a",
        "greedy",
    ) == "gen:02c4c41a3165d8d3cab8522a4493eada"
    assert make_sample_id is ids_make_sample_id


def test_smoke_selection_deterministic_and_partitioned():
    rows = make_prompt_rows(100)
    assignments = {}
    for index, row in enumerate(rows):
        if index < 80:
            assignments[row["prompt_id"]] = "train"
        elif index < 90:
            assignments[row["prompt_id"]] = "val"
        else:
            assignments[row["prompt_id"]] = "test_holdout"

    chosen = select_smoke_prompts(rows, 4, assignments)
    chosen_ids = [row["prompt_id"] for row in chosen]
    assert len(chosen_ids) == 4
    counts = {"train": 0, "val": 0, "test_holdout": 0}
    for prompt_id in chosen_ids:
        counts[assignments[prompt_id]] += 1
    assert counts == {"train": 2, "val": 1, "test_holdout": 1}

    shuffled = list(rows)
    random.Random(11).shuffle(shuffled)
    shuffled_assignments = dict(random.Random(12).sample(list(assignments.items()), len(assignments)))
    again = select_smoke_prompts(shuffled, 4, shuffled_assignments)
    assert [row["prompt_id"] for row in again] == chosen_ids

    five = select_smoke_prompts(rows, 5, assignments)
    five_counts = {"train": 0, "val": 0, "test_holdout": 0}
    for row in five:
        five_counts[assignments[row["prompt_id"]]] += 1
    assert five_counts == {"train": 3, "val": 1, "test_holdout": 1}


def test_dry_run_plans_20000_unique_ids_and_8_stream_files(config, tmp_path):
    rows = make_prompt_rows(5000)
    assignments = {
        row["prompt_id"]: (
            "train" if index < 4000 else ("val" if index < 4500 else "test_holdout")
        )
        for index, row in enumerate(rows)
    }
    prompt_path = tmp_path / "prompts.jsonl"
    prompt_sha = write_prompt_manifest(prompt_path, rows)
    split_path = tmp_path / "splits.json"
    write_split_manifest(split_path, rows, assignments, prompt_sha)

    report = run_generation(
        config,
        prompts_path=prompt_path,
        splits_path=split_path,
        out_dir=tmp_path / "run",
        dry_run=True,
    )

    assert report["mode"] == "dry-run"
    assert report["n_prompts"] == 5000
    assert report["sample_id_count"] == 20000
    assert report["unique_sample_ids"] == 20000
    assert len(set(report["expected_sample_ids"])) == 20000
    assert report["stream_count"] == 8
    assert len(report["streams"]) == 8
    kinds = [entry["kind"] for entry in report["streams"]]
    assert kinds.count("samples") == 4
    assert kinds.count("failures") == 4
    for entry in report["streams"]:
        name = Path(entry["path"]).name
        assert name.startswith(f"{entry['kind']}-")
        assert name.endswith(".jsonl")
    assert report["corpus_written"] is False
    assert not (tmp_path / "run").exists()


def test_plan_only_does_not_import_torch(config, tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "transformers", None)
    rows = make_prompt_rows(4)
    assignments = {row["prompt_id"]: "train" for row in rows}
    prompt_path = tmp_path / "prompts.jsonl"
    prompt_sha = write_prompt_manifest(prompt_path, rows)
    split_path = tmp_path / "splits.json"
    write_split_manifest(split_path, rows, assignments, prompt_sha)

    report = run_generation(
        config,
        prompts_path=prompt_path,
        splits_path=split_path,
        out_dir=tmp_path / "run",
        plan_only=True,
    )
    assert report["mode"] == "plan-only"
    assert report["sample_id_count"] == 16


# ---------------------------------------------------------------------------
# CLI safety gates
# ---------------------------------------------------------------------------


def test_cli_smoke_refuses_datasets_output():
    code = generate.main(
        [
            "--config",
            str(CONFIG_PATH),
            "--smoke",
            "4",
            "--out",
            str(REPO_ROOT / "datasets" / "should_not_exist"),
            "--prompts",
            "does-not-exist.jsonl",
            "--splits",
            "does-not-exist.json",
        ]
    )
    assert code == 2


def test_cli_full_run_requires_authorisation(capsys):
    code = generate.main(
        [
            "--config",
            str(CONFIG_PATH),
            "--out",
            "somewhere",
            "--prompts",
            "does-not-exist.jsonl",
            "--splits",
            "does-not-exist.json",
        ]
    )
    captured = capsys.readouterr()
    assert code == 2
    assert "--authorise-full-run" in captured.err
