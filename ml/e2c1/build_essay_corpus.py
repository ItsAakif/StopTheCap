"""Build the E2-C1 length-matched essay corpus.

The published E2-C1 essay result (``datasets/canonical/e2c1_essays.jsonl``) was
produced by an ad-hoc script that was never committed.  This module is that
builder, formalised.  It pairs every Ghostbuster student essay referenced by the
prompt manifest with the API-generated essays produced for the same prompt.

Human rows come from the canonical Ghostbuster corpus (``label == 0`` and
``domain == "student_essay"``), keyed by ``sample_id``.  AI rows come from the
per-generator shard directories under ``datasets/raw/e2c1_essay_ai``.  Rows are
sorted by ``sample_id`` and serialised with sorted keys so the output is
deterministic and can be checked against the published SHA-256.

The canonical artefact uses CRLF line endings; they are written explicitly so
the digest is identical on every platform.  Everything here is deterministic,
offline and CPU only -- no torch import.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

PROMPTS_PATH = "datasets/manifests/e2c1_essay_prompts.jsonl"
HUMAN_PATH = "datasets/canonical/ghostbuster.jsonl"
AI_ROOT = "datasets/raw/e2c1_essay_ai"
OUT_CORPUS = "datasets/canonical/e2c1_essays.jsonl"
OUT_MANIFEST = "datasets/manifests/e2_c1_essays.json"

EXPECTED_SHA256 = "8b0c0cc3beebe024302c0e87b75995a71378b30ffa3c7605b5a11f8e6a75bf2c"

#: Frozen manifest metadata; copied from the published manifest so a rebuild is
#: byte-identical rather than merely row-identical.
CREATED_AT = "2026-11-12"
GATE: dict[str, Any] = {
    "status": "FAIL",
    "reason": "newlines_per_100w AUC 0.901; digits_per_100w AUC 0.097",
    "word_count_auc": 0.486,
}
NOTES = (
    "AI essays generated to match human essay lengths; formatting/digit tells "
    "accepted per project decision."
)

#: The published corpus uses CRLF; keep it explicit for cross-platform digests.
LINE_ENDING = "\r\n"

HUMAN_SOURCE_DATASET = "ghostbuster_student_essay"
AI_SOURCE_DATASET = "e2c1_essay_ai"
TOPIC_CATEGORY = "student_essay"

_WHITESPACE = re.compile(r"\s+")


# ---------------------------------------------------------------------------
# Hashing / IO helpers.
# ---------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def content_hash(text: str) -> str:
    """Hash whitespace-normalised text (matches the published corpus rows)."""
    return sha256_text(_WHITESPACE.sub(" ", text).strip())


def human_sample_id(prompt_id: str) -> str:
    return "essayh:" + sha256_text(prompt_id)[:24]


def ai_sample_id(prompt_id: str, generator_id: str) -> str:
    return "essaya:" + sha256_text(f"{prompt_id}|{generator_id}")[:24]


def read_jsonl(path: Path, *, encoding: str = "utf-8") -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding=encoding) as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: record is not an object")
            rows.append(row)
    return rows


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise


def serialize_rows(rows: Sequence[Mapping[str, Any]]) -> bytes:
    """Canonical JSONL encoding; the corpus SHA-256 is taken over these bytes."""
    return "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + LINE_ENDING
        for row in rows
    ).encode("utf-8")


def serialize_manifest(manifest: Mapping[str, Any]) -> bytes:
    """Indented JSON with the same line endings as the published manifest."""
    text = json.dumps(manifest, indent=2, ensure_ascii=False)
    return text.replace("\n", LINE_ENDING).encode("utf-8")


# ---------------------------------------------------------------------------
# Input loading.
# ---------------------------------------------------------------------------


def is_usable_shard(path: Path) -> bool:
    """Skip failure logs, private helpers and temporary shard files."""
    name = Path(path).name
    return not (
        "_failures" in name or name.startswith("_") or name.startswith("tmp_")
    )


def iter_shard_paths(ai_root: Path) -> Iterable[Path]:
    """Yield usable shard files, generator dirs and shard names sorted."""
    ai_root = Path(ai_root)
    for generator_dir in sorted(path for path in ai_root.iterdir() if path.is_dir()):
        shards_dir = generator_dir / "shards"
        if not shards_dir.is_dir():
            continue
        for shard in sorted(shards_dir.glob("*.jsonl")):
            if is_usable_shard(shard):
                yield shard


def load_prompts(path: Path) -> dict[str, dict[str, Any]]:
    """Prompt manifest keyed by ``prompt_id`` (first record wins)."""
    prompts: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        prompt_id = str(row["prompt_id"])
        prompts.setdefault(prompt_id, row)
    return prompts


def load_human_essays(path: Path) -> dict[str, str]:
    """Ghostbuster student essays keyed by ``sample_id`` -> text."""
    essays: dict[str, str] = {}
    for row in read_jsonl(path):
        if row.get("domain") != TOPIC_CATEGORY:
            continue
        try:
            label = int(row.get("label"))
        except (TypeError, ValueError):
            continue
        if label != 0:
            continue
        essays[str(row["sample_id"])] = str(row["text"])
    return essays


# ---------------------------------------------------------------------------
# Row construction.
# ---------------------------------------------------------------------------


def build_human_rows(
    prompts: Mapping[str, Mapping[str, Any]],
    human_essays: Mapping[str, str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for prompt_id, prompt in prompts.items():
        text = human_essays.get(str(prompt.get("human_sample_id")))
        if text is None:
            continue
        rows.append(
            {
                "sample_id": human_sample_id(prompt_id),
                "text": text,
                "label": 0,
                "partition": str(prompt["partition"]),
                "source_dataset": HUMAN_SOURCE_DATASET,
                "group_id": str(prompt["group_id"]),
                "generator": None,
                "topic_category": TOPIC_CATEGORY,
                "content_hash": content_hash(text),
            }
        )
    return rows


def build_ai_rows(
    ai_root: Path,
    prompts: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for shard in iter_shard_paths(ai_root):
        for record in read_jsonl(shard, encoding="utf-8-sig"):
            prompt = prompts.get(str(record.get("prompt_id")))
            if prompt is None:
                continue
            prompt_id = str(record["prompt_id"])
            generator_id = str(record["generator_id"])
            text = str(record["text"])
            rows.append(
                {
                    "sample_id": ai_sample_id(prompt_id, generator_id),
                    "text": text,
                    "label": 1,
                    "partition": str(prompt["partition"]),
                    "source_dataset": AI_SOURCE_DATASET,
                    "group_id": str(prompt["group_id"]),
                    "generator": generator_id,
                    "topic_category": TOPIC_CATEGORY,
                    "content_hash": content_hash(text),
                }
            )
    return rows


def build_rows(
    prompts: Mapping[str, Mapping[str, Any]],
    human_essays: Mapping[str, str],
    ai_root: Path,
) -> list[dict[str, Any]]:
    rows = build_human_rows(prompts, human_essays) + build_ai_rows(ai_root, prompts)
    rows.sort(key=lambda row: str(row["sample_id"]))
    return rows


# ---------------------------------------------------------------------------
# Counts / manifest.
# ---------------------------------------------------------------------------


def build_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_label: Counter[str] = Counter()
    by_partition: Counter[str] = Counter()
    by_partition_label: Counter[str] = Counter()
    by_generator: Counter[str] = Counter()

    for row in rows:
        label = str(int(row["label"]))
        partition = str(row["partition"])
        by_label[label] += 1
        by_partition[partition] += 1
        by_partition_label[f"{partition}:{label}"] += 1
        if row.get("generator") is not None:
            by_generator[str(row["generator"])] += 1

    return {
        "total": len(rows),
        "by_label": dict(by_label),
        "by_partition": dict(by_partition),
        "by_partition_label": dict(by_partition_label),
        "by_generator": dict(by_generator),
    }


def build_manifest(
    rows: Sequence[Mapping[str, Any]], corpus_sha256: str
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "corpus": "e2c1_essays",
        "created_at": CREATED_AT,
        "counts": build_counts(rows),
        "corpus_sha256": corpus_sha256,
        "gate": dict(GATE),
        "notes": NOTES,
    }


# ---------------------------------------------------------------------------
# Full build.
# ---------------------------------------------------------------------------


def plan_corpus(
    prompts: Mapping[str, Mapping[str, Any]],
    human_essays: Mapping[str, str],
    ai_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], str]:
    """Build rows, manifest and digest without writing anything."""
    rows = build_rows(prompts, human_essays, ai_root)
    sample_ids = [str(row["sample_id"]) for row in rows]
    duplicates = sorted(
        sample_id
        for sample_id, count in Counter(sample_ids).items()
        if count > 1
    )
    if duplicates:
        raise ValueError(f"duplicate sample_id(s): {duplicates[:5]}")
    payload = serialize_rows(rows)
    digest = hashlib.sha256(payload).hexdigest()
    return rows, build_manifest(rows, digest), digest


def write_corpus(
    rows: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    *,
    out_path: Path,
    manifest_path: Path,
) -> str:
    """Atomically write the corpus and manifest; returns the corpus SHA-256."""
    payload = serialize_rows(rows)
    digest = hashlib.sha256(payload).hexdigest()
    if digest != manifest["corpus_sha256"]:
        raise ValueError("corpus changed between planning and writing")
    _write_bytes_atomic(Path(out_path), payload)
    _write_bytes_atomic(Path(manifest_path), serialize_manifest(manifest))
    return digest


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build the E2-C1 length-matched essay corpus (offline, CPU only)."
        )
    )
    parser.add_argument("--prompts", default=PROMPTS_PATH, help="prompt manifest JSONL")
    parser.add_argument("--human", default=HUMAN_PATH, help="canonical Ghostbuster JSONL")
    parser.add_argument("--ai-root", default=AI_ROOT, help="generated essay shard root")
    parser.add_argument("--out", default=OUT_CORPUS, help="output corpus JSONL")
    parser.add_argument("--manifest", default=OUT_MANIFEST, help="output manifest JSON")
    parser.add_argument(
        "--expect-sha256",
        default=EXPECTED_SHA256,
        help="expected corpus SHA-256; the build fails if the digest differs",
    )
    return parser


def _print_report(manifest: Mapping[str, Any], out_path: Path, manifest_path: Path) -> None:
    counts = manifest["counts"]
    print(f"rows={counts['total']}")
    print(f"by_label={counts['by_label']}")
    print(f"by_partition={counts['by_partition']}")
    print(f"by_partition_label={counts['by_partition_label']}")
    print(f"by_generator={counts['by_generator']}")
    print(f"wrote {out_path}")
    print(f"wrote {manifest_path}")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)

    try:
        prompts = load_prompts(Path(args.prompts))
        human_essays = load_human_essays(Path(args.human))
        rows, manifest, digest = plan_corpus(
            prompts, human_essays, Path(args.ai_root)
        )
    except FileNotFoundError as exc:
        print(f"input not found: {exc}", file=sys.stderr)
        return 2
    except (OSError, ValueError, KeyError) as exc:
        print(f"invalid input: {exc}", file=sys.stderr)
        return 2

    try:
        written_digest = write_corpus(
            rows,
            manifest,
            out_path=Path(args.out),
            manifest_path=Path(args.manifest),
        )
    except OSError as exc:
        print(f"write failed: {exc}", file=sys.stderr)
        return 2

    actual = sha256_file(Path(args.out))
    if args.expect_sha256 and actual != args.expect_sha256:
        print(
            f"corpus sha256 mismatch: expected {args.expect_sha256}, got {actual}",
            file=sys.stderr,
        )
        return 1

    _print_report(manifest, Path(args.out), Path(args.manifest))
    print(f"corpus_sha256={written_digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
