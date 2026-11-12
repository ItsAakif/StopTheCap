"""Offline contract tests for the frozen E2-C1 API shard merge.

Every fixture is synthetic and lives in ``tmp_path``.  The real shards, the real
manifests and the network are never touched.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from ml.e2c1 import merge_api_shards as merge
from ml.e2c1.constants import SPLIT_SALT
from ml.e2c1.generation import GenerationRecord
from ml.e2c1.split import e2c1_split_digest


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def prompt_id(index: int) -> str:
    return f"pmt:{index:024x}"


def make_prompt(index: int, topic: str = "other") -> dict:
    text = f"prompt {index} source passage"
    return {
        "prompt_id": prompt_id(index),
        "prompt_text": text,
        "prompt_hash": sha(text),
        "topic_category": topic,
    }


def write_prompts(path: Path, prompts: list[dict]) -> str:
    data = b"".join(
        json.dumps(row, sort_keys=True, ensure_ascii=False).encode("utf-8") + b"\n"
        for row in prompts
    )
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def write_splits(
    path: Path,
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


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        b"".join(
            json.dumps(row, sort_keys=True, ensure_ascii=False).encode("utf-8")
            + b"\n"
            for row in rows
        )
    )


def shard_row(prompt: str, decoding: str, text: str) -> dict:
    return {
        "prompt_id": prompt,
        "decoding_config_id": decoding,
        "text": text,
    }


@pytest.fixture()
def corpus(tmp_path: Path) -> dict:
    prompts = [
        make_prompt(0, "other"),
        make_prompt(1, "stem"),
        make_prompt(2, "history"),
        make_prompt(3, "other"),
    ]
    assignments = {
        prompts[0]["prompt_id"]: "train",
        prompts[1]["prompt_id"]: "val",
        prompts[2]["prompt_id"]: "test_holdout",
        prompts[3]["prompt_id"]: "train",
    }
    prompts_path = tmp_path / "prompts.jsonl"
    prompt_sha = write_prompts(prompts_path, prompts)
    splits_path = tmp_path / "splits.json"
    write_splits(splits_path, assignments, prompt_sha)

    root = tmp_path / "shards"
    deepseek = root / "deepseek_v41_flash" / "shards"
    muse = root / "muse_spark_1_3" / "shards"
    # Directory name is a leftover from the abandoned GLM plan.
    longcat = root / "glm_53_flash" / "shards"

    deepseek_rows = []
    for prompt in prompts:
        for decoding in ("greedy", "sampling"):
            deepseek_rows.append(
                shard_row(prompt["prompt_id"], decoding, f"deepseek {prompt['prompt_id']} {decoding}")
            )
    # One row whose prompt is not in the frozen manifest: must be dropped.
    deepseek_rows.append(
        shard_row("pmt:" + "f" * 24, "greedy", "orphan prompt row")
    )
    # One exact duplicate key: must be rejected, not merged.
    deepseek_rows.append(
        shard_row(prompts[0]["prompt_id"], "greedy", "duplicate deepseek row")
    )
    write_jsonl(deepseek / "d000.jsonl", deepseek_rows)
    write_jsonl(
        deepseek / "d000_failures.jsonl",
        [
            {"prompt_id": prompts[0]["prompt_id"], "rejection_reason": "length_short"},
            {"prompt_id": prompts[1]["prompt_id"], "rejection_reason": "echo"},
        ],
    )

    muse_rows = []
    for prompt in prompts:
        for decoding in ("greedy", "sampling"):
            muse_rows.append(
                shard_row(prompt["prompt_id"], decoding, f"muse {prompt['prompt_id']} {decoding}")
            )
    write_jsonl(muse / "m000.jsonl", muse_rows)
    # Intermediate part: must be excluded.
    write_jsonl(
        muse / "m000_p1.jsonl",
        [shard_row(prompts[3]["prompt_id"], "greedy", "muse intermediate")],
    )
    write_jsonl(
        muse / "m000_failures.jsonl",
        [{"prompt_id": prompts[2]["prompt_id"], "reason": "repetition"}],
    )

    glm_rows = []
    for prompt in prompts[:2]:
        for decoding in ("greedy", "sampling"):
            glm_rows.append(
                shard_row(prompt["prompt_id"], decoding, f"longcat {prompt['prompt_id']} {decoding}")
            )
    write_jsonl(longcat / "g000.jsonl", glm_rows)
    # Cancelled shard: must be ignored.
    write_jsonl(
        longcat / "g004.jsonl",
        [shard_row(prompts[2]["prompt_id"], "greedy", "cancelled longcat shard")],
    )

    result = merge.merge_corpus(
        shards_root=root,
        prompts_path=prompts_path,
        splits_path=splits_path,
    )
    return {
        "result": result,
        "prompts": prompts,
        "assignments": assignments,
        "root": root,
        "prompts_path": prompts_path,
        "splits_path": splits_path,
    }


def test_generator_mapping_is_frozen():
    assert merge.GENERATOR_IDS == (
        "deepseek_v41_flash",
        "muse_spark_1_3",
        "longcat_25",
    )
    expected = {
        "deepseek_v41_flash": ("api:deepseek-v4.1-flash", "deepseek-v4.1-flash"),
        "muse_spark_1_3": ("api:muse-spark-1.3", "muse-spark-1.3"),
        "longcat_25": ("api:longcat-2.5", "longcat-2.5"),
    }
    for generator_id, (model_id, served) in expected.items():
        spec = merge.GENERATOR_BY_ID[generator_id]
        assert spec.model_id == model_id
        assert spec.model_as_served == served
        assert spec.model_revision == hashlib.sha256(
            f"{served}|2026-11-10".encode("utf-8")
        ).hexdigest()[:40]

    payload = {
        "config_id": "e2c1-api-v1",
        "generators": list(merge.GENERATOR_IDS),
        "max_new_tokens": 192,
        "prompt_template_id": "e2c1-wiki-passage-v2",
    }
    assert merge.DEFAULT_CONFIG_DIGEST == hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
    ).hexdigest()
    assert merge.DEFAULT_CHAT_TEMPLATE_SHA256 == sha(
        merge.SYSTEM + "\n" + merge.USER_TEMPLATE
    )


def test_invalid_prompt_ids_are_dropped(corpus):
    result: merge.MergeResult = corpus["result"]
    report = result.report
    assert report["input"]["successes"] == 22
    assert report["dropped_invalid_rows"]["count"] == 1
    assert report["dropped_invalid_rows"]["distinct_prompt_ids"] == 1
    dropped = report["dropped_invalid_rows"]["rows"][0]
    assert dropped["prompt_id"] == "pmt:" + "f" * 24
    assert dropped["prompt_id"] not in {r["prompt_id"] for r in result.records}


def test_duplicate_keys_are_rejected(corpus):
    result: merge.MergeResult = corpus["result"]
    report = result.report
    assert report["duplicates"]["count"] == 1
    duplicate = report["duplicates"]["rows"][0]
    assert duplicate["generator_id"] == "deepseek_v41_flash"
    assert duplicate["prompt_id"] == corpus["prompts"][0]["prompt_id"]
    assert duplicate["decoding_config_id"] == "greedy"
    keys = [
        (r["generator_id"], r["prompt_id"], r["decoding_config_id"])
        for r in result.records
    ]
    assert len(keys) == len(set(keys))


def test_partition_comes_from_split_assignments(corpus):
    result: merge.MergeResult = corpus["result"]
    assignments = corpus["assignments"]
    for record in result.records:
        assert record["partition"] == assignments[record["prompt_id"]]


def test_sample_id_is_deterministic_and_recomputes(corpus):
    result: merge.MergeResult = corpus["result"]
    for record in result.records:
        expected = "gen:" + hashlib.sha256(
            (
                f"{record['prompt_id']}|{record['generator_id']}|"
                f"{record['model_revision']}|{record['decoding_config_id']}"
            ).encode("utf-8")
        ).hexdigest()[:32]
        assert record["sample_id"] == expected
        assert merge.sample_id_for(
            record["prompt_id"],
            record["generator_id"],
            record["model_revision"],
            record["decoding_config_id"],
        ) == expected
    assert [r["sample_id"] for r in result.records] == sorted(
        r["sample_id"] for r in result.records
    )


def test_merged_records_match_generation_record_schema(corpus):
    result: merge.MergeResult = corpus["result"]
    prompts = {row["prompt_id"]: row for row in corpus["prompts"]}
    for payload in result.records:
        record = GenerationRecord.model_validate(payload)
        assert record.finish_reason == "unknown"
        assert record.quality_flags == ["agent_generated"]
        assert record.max_new_tokens == 192
        assert record.seed == -1
        assert record.batch_index == 0
        assert record.generation_timestamp == "2026-11-10T00:00:00Z"
        assert record.prompt_template_id == "e2c1-wiki-passage-v2"
        assert record.licence_ref == record.generator_id
        assert record.quantisation == {
            "method": "provider-default",
            "deviation_from_published_checkpoint": False,
        }
        assert record.chat_template_sha256 == merge.DEFAULT_CHAT_TEMPLATE_SHA256
        assert record.config_digest == merge.DEFAULT_CONFIG_DIGEST
        assert record.model_revision == merge.GENERATOR_BY_ID[
            record.generator_id
        ].model_revision
        assert record.source_prompt_hash == prompts[record.prompt_id]["prompt_hash"]
        assert record.topic_category == prompts[record.prompt_id]["topic_category"]
        if record.decoding_config_id == "greedy":
            assert record.temperature == 0.0
            assert record.top_p == 1.0
            assert record.do_sample is False
        else:
            assert record.temperature == 0.8
            assert record.top_p == 0.95
            assert record.do_sample is True


def test_counts_reconcile(corpus):
    result: merge.MergeResult = corpus["result"]
    report = result.report
    reconciliation = report["reconciliation"]
    assert reconciliation["reconciled"] is True
    assert reconciliation["successes"] == (
        reconciliation["merged"]
        + reconciliation["dropped_invalid"]
        + reconciliation["duplicates"]
    )
    counts = report["counts"]
    assert sum(counts["by_generator"].values()) == len(result.records)
    assert sum(counts["by_partition"].values()) == len(result.records)
    assert sum(counts["by_decoding"].values()) == len(result.records)
    assert sum(counts["by_topic"].values()) == len(result.records)
    assert counts["by_generator"] == {
        "deepseek_v41_flash": 8,
        "muse_spark_1_3": 8,
        "longcat_25": 4,
    }
    assert counts["by_partition"] == {
        "train": 10,
        "val": 6,
        "test_holdout": 4,
    }
    assert counts["by_decoding"] == {"greedy": 10, "sampling": 10}

    matrix = counts["by_generator_partition_decoding_topic"]
    total = 0
    for generator in merge.GENERATOR_IDS:
        for partition in ("train", "val", "test_holdout"):
            for decoding in merge.DECODING_IDS:
                total += sum(matrix[generator][partition][decoding].values())
    assert total == len(result.records)


def test_shard_discovery_excludes_intermediates_and_cancelled_shards(corpus):
    root = corpus["root"]
    muse_files = merge.discover_shard_files(
        root, merge.GENERATOR_BY_ID["muse_spark_1_3"]
    )
    assert [path.name for path in muse_files] == ["m000.jsonl"]
    glm_files = merge.discover_shard_files(
        root, merge.GENERATOR_BY_ID["longcat_25"]
    )
    assert [path.name for path in glm_files] == ["g000.jsonl"]
    deepseek_files = merge.discover_shard_files(
        root, merge.GENERATOR_BY_ID["deepseek_v41_flash"]
    )
    assert [path.name for path in deepseek_files] == ["d000.jsonl"]

    result: merge.MergeResult = corpus["result"]
    assert result.report["failures_found"]["count"] == 3
    assert result.report["failures_found"]["by_reason"] == {
        "echo": 1,
        "length_short": 1,
        "repetition": 1,
    }


def test_split_counts_update_preserves_digest(corpus):
    result: merge.MergeResult = corpus["result"]
    original = result.split
    updated = merge.build_split_counts(original, result.records)

    assert updated["assignments"] == original["assignments"]
    assert updated["salt"] == original["salt"]
    assert updated["split_digest"] == original["split_digest"]
    assert e2c1_split_digest(updated["assignments"], updated["salt"]) == (
        original["split_digest"]
    )

    matrix = updated["counts"]["by_partition_generator_decoding"]
    assert set(matrix) == {"train", "val", "test_holdout"}
    for partition in matrix:
        assert set(matrix[partition]) == set(merge.GENERATOR_IDS)
        for generator in matrix[partition]:
            assert set(matrix[partition][generator]) == set(merge.DECODING_IDS)
    total = sum(
        cell
        for partition in matrix.values()
        for generator in partition.values()
        for cell in generator.values()
    )
    assert total == len(result.records)


def test_corpus_manifest_reports_coverage_and_sha(corpus):
    result: merge.MergeResult = corpus["result"]
    manifest = merge.build_corpus_manifest(
        result,
        merged_path=Path("merged.jsonl"),
        prompt_manifest_path=Path("prompts.jsonl"),
        split_manifest_path=Path("splits.json"),
    )
    assert manifest["merged"]["sha256"] == result.merged_sha256
    assert manifest["merged"]["rows"] == len(result.records)
    assert manifest["split_digest"] == result.split_digest
    assert manifest["prompt_manifest_sha256"] == result.prompt_manifest_sha256
    coverage = manifest["coverage"]
    assert (
        coverage["prompts_with_1_generator"]
        + coverage["prompts_with_2_generators"]
        + coverage["prompts_with_3_generators"]
        == coverage["prompts_with_samples"]
    )
    assert coverage["histogram"] == {
        "1": coverage["prompts_with_1_generator"],
        "2": coverage["prompts_with_2_generators"],
        "3": coverage["prompts_with_3_generators"],
    }
    assert manifest["composition"]["n_rows"] == len(result.records)


def test_licence_registry_shape():
    existing = {
        "models": [
            {
                "generator_id": "mistral_7b_instruct_v02",
                "model_id": "mistralai/Mistral-7B-Instruct-v0.2",
                "licence": {"spdx": "Apache-2.0", "name": "Apache License 2.0"},
                "output_terms": {"outputs_licensed": False},
                "status": "approved_private_research",
            }
        ],
        "withdrawn_models": [
            {
                "generator_id": "llama_32_3b_instruct",
                "model_id": "meta-llama/Llama-3.2-3B-Instruct",
                "licence": {"spdx": "NOASSERTION"},
                "output_terms": {"outputs_licensed": False},
                "status": "not_admitted_access_pending",
            }
        ],
    }
    registry = merge.build_licence_registry(existing)
    assert [entry["generator_id"] for entry in registry["models"]] == list(
        merge.GENERATOR_IDS
    )
    for entry in registry["models"]:
        assert entry["licence"]["spdx"] == "NOASSERTION"
        assert entry["licence"]["name"] == "Provider API terms"
        assert entry["licence"]["licence_file_present"] is False
        assert entry["output_terms"]["outputs_licensed"] is False
        assert entry["status"] == "approved_private_research"
        assert entry["confidence"] == "low"
        assert "not independently audited" in entry["note"].lower()
        assert "authorised" in entry["note"].lower()
    withdrawn = {
        entry["generator_id"]: entry for entry in registry["withdrawn_models"]
    }
    assert set(withdrawn) == {
        "llama_32_3b_instruct",
        "mistral_7b_instruct_v02",
        "glm_53_flash",
    }
    assert withdrawn["llama_32_3b_instruct"]["reason"] == "access pending"
    assert withdrawn["mistral_7b_instruct_v02"]["reason"] == (
        "no samples generated; unused"
    )
    assert withdrawn["glm_53_flash"]["reason"] == (
        "no samples generated; directory name is a leftover"
    )
