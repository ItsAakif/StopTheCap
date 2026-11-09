"""E2-C1 generation contract (§D).

This module is deliberately import-safe: **it never imports torch, transformers,
accelerate or bitsandbytes at module import time**. The real model imports happen
inside :func:`run_generation`, after all planning, resume and authorisation checks
have passed. ``import ml.e2c1.generation`` therefore works on a CPU-only machine
with those packages absent or blocked.

The pieces here are:

* pydantic v2 schemas for the config and the two JSONL record kinds;
* config loading, canonical digesting and licence-registry validation;
* prompt rendering, post-processing and §D.4 rejection;
* append/read/stats helpers for the JSONL streams;
* checkpoint state helpers;
* :func:`run_generation`, the orchestrator used by the CLI and the Colab notebook.
"""

from __future__ import annotations

import collections
import gc
import hashlib
import json
import os
import re
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ml.ingest.normalise import (
    jaccard,
    normalise_and_clean,
    normalise_text,
    shingles,
    word_count,
)

from .constants import GENERATION_SEED, SPLIT_SALT
from .ids import make_sample_id
from .prompts import latin_ratio
from .split import e2c1_split_digest

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "ml" / "e2c1" / "generation_config.json"
DEFAULT_OUT_DIR = Path("datasets/raw/e2c1_generated")
SMOKE_SEED = "e2c1-smoke-v1"
STATE_SCHEMA_VERSION = "1.0"
STATE_FLUSH_EVERY = 25

REVISION_RE = r"^[0-9a-f]{40,64}$"
SAMPLE_ID_RE = r"^gen:[0-9a-f]{32}$"
PROMPT_ID_RE = r"^pmt:[0-9a-f]{24}$"
SHA256_RE = r"^[0-9a-f]{64}$"
TIMESTAMP_RE = r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$"

DEFAULT_ALLOWED_STATUSES = (
    "approved_private_research",
    "approved_private_research_conditional",
)

REFUSAL_RE = re.compile(r"\b(I cannot|I can't|I'm sorry|I am unable)\b", re.IGNORECASE)


class GenerationError(RuntimeError):
    """Base class for generation-contract refusals."""


class LicenceValidationError(GenerationError):
    """One or more config/licence-registry violations.

    ``problems`` is machine readable; ``str(exc)`` joins every violation so a
    caller never has to fix them one at a time.
    """

    def __init__(self, problems: Sequence[str]) -> None:
        self.problems = list(problems)
        super().__init__("; ".join(self.problems))


class PromptRejection(GenerationError):
    """Raised when a prompt cannot be built (delimiter collision, too long)."""

    def __init__(self, reason: str, detail: str = "") -> None:
        self.reason = reason
        message = reason if not detail else f"{reason}: {detail}"
        super().__init__(message)


class ResumeCorruptionError(GenerationError):
    """Raised when a JSONL stream is internally inconsistent; never guessed at."""


class ResumeMismatch(GenerationError):
    """Raised when a resume would merge different configs/prompts/splits."""


class AuthorisationError(GenerationError):
    """Raised when a full run is attempted without explicit authorisation."""


class SmokeOutputError(GenerationError):
    """Raised when smoke output would be written under ``datasets/``."""


# ---------------------------------------------------------------------------
# Schemas (§D.2)
# ---------------------------------------------------------------------------


class QuantisationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    method: str
    load_in_4bit: bool | None = None
    quant_type: str | None = None
    double_quant: bool | None = None
    compute_dtype: str | None = None
    dtype: str | None = None
    deviation_from_published_checkpoint: bool


class GeneratorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    generator_id: str
    model_id: str
    model_revision: str = Field(pattern=REVISION_RE)
    architecture: str
    loader: Literal["causal", "multimodal"]
    dtype: str
    quantisation: QuantisationConfig
    gated: bool = False


class DecodingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decoding_config_id: str
    do_sample: bool
    temperature: float
    top_p: float
    max_new_tokens: int
    seed_policy: str


class RejectionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min_words: int = 40
    echo_jaccard: float = 0.30
    echo_containment: float = 0.20
    max_shared_12grams: int = 0
    repeat_ratio: float = 0.50
    max_8gram_repeat: int = 3
    latin_ratio: float = 0.80


class GenerationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str
    config_id: str
    seed: str
    prompt_manifest: str
    split_manifest: str
    licence_registry: str
    expected_prompts: int
    partitions: list[str]
    batch_size: int
    prompt_template_id: str
    max_input_tokens: int
    generators: list[GeneratorConfig] = Field(min_length=1)
    decodings: list[DecodingConfig] = Field(min_length=1)
    rejection: RejectionConfig


class GenerationRecord(BaseModel):
    """A successful sample (§D.2). ``extra="forbid"`` is part of the contract."""

    model_config = ConfigDict(extra="forbid")

    sample_id: str = Field(pattern=SAMPLE_ID_RE)
    prompt_id: str = Field(pattern=PROMPT_ID_RE)
    generator_id: str
    model_id: str
    model_revision: str = Field(pattern=REVISION_RE)
    licence_ref: str
    decoding_config_id: str
    temperature: float
    top_p: float
    max_new_tokens: int
    seed: int
    batch_index: int
    do_sample: bool
    generated_text: str = Field(min_length=1)
    generated_text_hash: str = Field(pattern=SHA256_RE)
    generation_timestamp: str = Field(pattern=TIMESTAMP_RE)
    partition: str
    topic_category: str
    source_prompt_hash: str = Field(pattern=SHA256_RE)
    quantisation: dict[str, Any]
    prompt_template_id: str
    chat_template_sha256: str = Field(pattern=SHA256_RE)
    config_digest: str = Field(pattern=SHA256_RE)
    finish_reason: Literal["eos", "length"]
    quality_flags: list[str] = Field(default_factory=list)


class GenerationFailure(BaseModel):
    """A rejected attempt. Several attempts may exist for one ``sample_id``."""

    model_config = ConfigDict(extra="forbid")

    sample_id: str = Field(pattern=SAMPLE_ID_RE)
    prompt_id: str = Field(pattern=PROMPT_ID_RE)
    generator_id: str
    model_id: str
    model_revision: str = Field(pattern=REVISION_RE)
    licence_ref: str
    decoding_config_id: str
    temperature: float
    top_p: float
    max_new_tokens: int
    seed: int
    batch_index: int
    do_sample: bool
    generation_timestamp: str = Field(pattern=TIMESTAMP_RE)
    partition: str
    topic_category: str
    source_prompt_hash: str = Field(pattern=SHA256_RE)
    quantisation: dict[str, Any]
    prompt_template_id: str
    chat_template_sha256: str = Field(pattern=SHA256_RE)
    config_digest: str = Field(pattern=SHA256_RE)
    status: Literal["rejected"] = "rejected"
    rejection_reason: str
    attempt: int = Field(ge=1)
    raw_text_hash: str = Field(pattern=SHA256_RE)
    finish_reason: str = "length"
    quality_flags: list[str] = Field(default_factory=list)
    generated_text: str | None = None
    generated_text_hash: str | None = Field(default=None, pattern=SHA256_RE)


# ---------------------------------------------------------------------------
# Config / digest / licences
# ---------------------------------------------------------------------------


def _as_config(config: GenerationConfig | Mapping[str, Any] | str | Path) -> GenerationConfig:
    if isinstance(config, GenerationConfig):
        return config
    if isinstance(config, (str, Path)):
        return load_config(Path(config))
    return GenerationConfig.model_validate(dict(config))


def _model_json(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    return value


def load_config(path: Path) -> GenerationConfig:
    """Load and validate the §D.1 config. Unknown fields are rejected."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return GenerationConfig.model_validate(payload)


def canonical_json(value: Any) -> str:
    """Canonical JSON used for every digest in this module."""
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )


def config_digest(config: GenerationConfig | Mapping[str, Any] | str | Path) -> str:
    """SHA-256 of the canonical config JSON (§D.1). Stable across runs."""
    if isinstance(config, (str, Path)):
        config = load_config(Path(config))
    if isinstance(config, GenerationConfig):
        payload = config.model_dump(mode="json")
    else:
        # Validate mappings too, so omitted optional fields are canonicalised to
        # their defaults exactly as they are for a loaded config.
        payload = GenerationConfig.model_validate(dict(config)).model_dump(mode="json")
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def load_registry(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _registry_document(registry: Mapping[str, Any] | str | Path) -> Mapping[str, Any]:
    if isinstance(registry, (str, Path)):
        return load_registry(Path(registry))
    return registry


def _allowed_statuses(registry: Mapping[str, Any]) -> tuple[str, ...]:
    validation = registry.get("validation")
    if isinstance(validation, Mapping):
        values = validation.get("allowed_statuses")
        if values:
            return tuple(str(value) for value in values)
    values = registry.get("allowed_statuses")
    if values:
        return tuple(str(value) for value in values)
    return DEFAULT_ALLOWED_STATUSES


def _describes_apache(entry: Mapping[str, Any], licence: Mapping[str, Any]) -> bool:
    blobs = [
        entry.get("generator_id"),
        entry.get("model_id"),
        entry.get("display_name"),
        licence.get("name"),
        licence.get("spdx"),
    ]
    return any("apache" in str(value).lower() for value in blobs if value is not None)


def validate_config_against_licences(
    config: GenerationConfig | Mapping[str, Any] | str | Path,
    registry: Mapping[str, Any] | str | Path,
) -> None:
    """Validate every config generator against the §D licence registry.

    Raises one :class:`LicenceValidationError` listing *every* violation. A
    missing entry, model/revision mismatch, disallowed status, bad revision format,
    ``outputs_licensed`` not false, and an Apache-2.0 mislabel are all checked.
    Only ``models`` is validated; ``withdrawn_models`` is historical provenance.
    """
    cfg = _as_config(config)
    document = _registry_document(registry)
    models = document.get("models") or []
    by_generator: dict[str, Mapping[str, Any]] = {}
    for entry in models:
        if isinstance(entry, Mapping) and entry.get("generator_id"):
            by_generator[str(entry["generator_id"])] = entry

    allowed = _allowed_statuses(document)
    problems: list[str] = []

    for generator in cfg.generators:
        gid = generator.generator_id
        entry = by_generator.get(gid)
        if entry is None:
            problems.append(
                f"generator {gid!r}: missing generator entry in licence registry"
            )
            continue

        licence = entry.get("licence")
        if not isinstance(licence, Mapping):
            licence = {}

        entry_model_id = entry.get("model_id")
        if entry_model_id != generator.model_id:
            problems.append(
                f"generator {gid!r}: model_id {generator.model_id!r} does not match "
                f"registry model_id {entry_model_id!r}"
            )

        entry_revision = entry.get("revision", entry.get("model_revision"))
        if entry_revision != generator.model_revision:
            problems.append(
                f"generator {gid!r}: model_revision {generator.model_revision!r} "
                f"does not match registry revision {entry_revision!r}"
            )
        if not re.match(REVISION_RE, str(generator.model_revision)):
            problems.append(
                f"generator {gid!r}: model_revision {generator.model_revision!r} "
                f"does not match {REVISION_RE}"
            )
        if not re.match(REVISION_RE, str(entry_revision or "")):
            problems.append(
                f"generator {gid!r}: registry revision {entry_revision!r} "
                f"does not match {REVISION_RE}"
            )

        status = entry.get("status")
        if status not in allowed:
            problems.append(
                f"generator {gid!r}: status {status!r} is not in allowed_statuses "
                f"{list(allowed)!r}"
            )

        output_terms = entry.get("output_terms")
        if not isinstance(output_terms, Mapping) or output_terms.get("outputs_licensed") is not False:
            problems.append(
                f"generator {gid!r}: outputs_licensed must be false "
                f"(got {None if not isinstance(output_terms, Mapping) else output_terms.get('outputs_licensed')!r})"
            )

        spdx = licence.get("spdx")
        if _describes_apache(entry, licence) and spdx != "Apache-2.0":
            problems.append(
                f"generator {gid!r}: model described as Apache-2.0 but licence.spdx "
                f"is {spdx!r} (must be 'Apache-2.0')"
            )

    if problems:
        raise LicenceValidationError(problems)


# ---------------------------------------------------------------------------
# Prompt template (§D.3)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a careful encyclopedic writer. You never copy source sentences."
)

USER_TEMPLATE = (
    "Source material (do not copy sentences from it):\n"
    "<<<SOURCE\n"
    "{source_text}\n"
    "SOURCE>>>\n\n"
    "Write a new, self-contained encyclopedic passage of 90-120 words about the "
    "same subject. Use your own wording. Do not quote or paraphrase "
    "sentence-by-sentence. Do not add a title, headings, notes, or commentary. "
    "Begin directly with the passage."
)

THINKING_DISABLED_KEY = "thinking_disabled"
_THINKING_KWARG_ERRORS = (TypeError, ValueError)


def _apply_chat_template(
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    thinking_state: dict[str, bool] | None = None,
) -> str:
    """Render messages, asking the template to suppress reasoning when supported.

    ``enable_thinking=False`` is accepted by Qwen3.5-style templates. Templates
    that do not understand the keyword (e.g. Mistral) raise ``TypeError`` or
    ``ValueError``; they are retried without it. ``thinking_state`` records
    whether the keyword was accepted so :func:`run_generation` can report it.
    """
    try:
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except _THINKING_KWARG_ERRORS:
        if thinking_state is not None:
            thinking_state[THINKING_DISABLED_KEY] = False
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    if thinking_state is not None:
        thinking_state[THINKING_DISABLED_KEY] = True
    return rendered


def render_prompt(
    source_text: str,
    tokenizer: Any,
    config: Any,
    flags: list[str] | None = None,
    thinking_state: dict[str, bool] | None = None,
) -> str:
    """Render the §D.3 system+user messages with the tokenizer's chat template.

    The instruction is **90-120 words** so that 160 new tokens rarely truncates
    the passage. Thinking is disabled when the template supports it. A source text
    containing a delimiter is rejected as ``prompt_delimiter_collision`` rather
    than silently corrupting the prompt.

    Some chat templates (notably multimodal processor templates) reject a
    ``system`` role. In that case the system text is prepended to the user message
    and ``system_role_merged`` is appended to ``flags`` when a list is supplied.
    """
    del config  # the template id is recorded on the record; rendering is fixed
    if "<<<SOURCE" in source_text or "SOURCE>>>" in source_text:
        raise PromptRejection(
            "prompt_delimiter_collision",
            "source_text contains an E2-C1 prompt delimiter",
        )

    user_text = USER_TEMPLATE.format(source_text=source_text)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ]
    try:
        return _apply_chat_template(
            tokenizer, messages, thinking_state=thinking_state
        )
    except Exception:
        merged = [
            {
                "role": "user",
                "content": f"{SYSTEM_PROMPT}\n\n{user_text}",
            }
        ]
        rendered = _apply_chat_template(
            tokenizer, merged, thinking_state=thinking_state
        )
        if flags is not None:
            flags.append("system_role_merged")
        return rendered


# ---------------------------------------------------------------------------
# Post-processing and rejection (§D.4)
# ---------------------------------------------------------------------------


def _rejection_config(config: Any) -> RejectionConfig:
    if isinstance(config, GenerationConfig):
        return config.rejection
    if isinstance(config, RejectionConfig):
        return config
    if isinstance(config, Mapping):
        block = config.get("rejection", config)
        return RejectionConfig.model_validate(dict(block))
    return RejectionConfig()


def _tokens(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def _ngrams(tokens: Sequence[str], n: int) -> list[tuple[str, ...]]:
    if len(tokens) < n:
        return []
    return [tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1)]


def _echo_reason(text: str, source_text: str, rejection: RejectionConfig) -> str | None:
    text_shingles = shingles(text, 5)
    source_shingles = shingles(source_text, 5)

    if jaccard(text_shingles, source_shingles) >= rejection.echo_jaccard:
        return "echo"

    if len(text_shingles):
        containment = len(text_shingles & source_shingles) / len(text_shingles)
        if containment >= rejection.echo_containment:
            return "echo"

    shared_12 = set(_ngrams(_tokens(text), 12)) & set(_ngrams(_tokens(source_text), 12))
    if len(shared_12) > rejection.max_shared_12grams:
        return "echo"

    return None


def _repetition_reason(text: str, rejection: RejectionConfig) -> str | None:
    tokens = _tokens(text)
    five = _ngrams(tokens, 5)
    if five:
        ratio = 1.0 - len(set(five)) / len(five)
        if ratio > rejection.repeat_ratio:
            return "repetition"

    eight = collections.Counter(_ngrams(tokens, 8))
    if eight and max(eight.values()) > rejection.max_8gram_repeat:
        return "repetition"

    return None


def postprocess(
    raw: str,
    source_text: str,
    finish_reason: str,
    config: Any,
) -> tuple[str, list[str], str | None]:
    """Normalise, strip wrapping quotes, then apply §D.4 in order (first hit wins).

    Returns ``(text, flags, rejection_reason)``; ``rejection_reason`` is ``None``
    on success. ``flags`` is retained on the record even when the sample passes.
    A complete ``</think>`` block is dropped before normalisation; an unclosed
    ``<think>`` block is left for the ``truncated`` rule to reject.
    """
    rejection = _rejection_config(config)
    thinking_stripped = "</think>" in raw
    if thinking_stripped:
        raw = raw.rsplit("</think>", 1)[1].lstrip()
    text, flags = normalise_and_clean(raw)
    if thinking_stripped:
        flags.append("thinking_stripped")

    if text[:1] == '"' and text[-1:] == '"' and text.count('"') == 2:
        text = text[1:-1].strip()
        flags.append("wrapped_quotes_removed")

    if text == "":
        return text, flags, "empty"
    if word_count(text) < rejection.min_words:
        return text, flags, "length_short"
    if finish_reason == "length":
        return text, flags, "truncated"
    if REFUSAL_RE.search(text[:200]):
        return text, flags, "refusal"

    echo = _echo_reason(text, source_text, rejection)
    if echo is not None:
        return text, flags, echo

    repetition = _repetition_reason(text, rejection)
    if repetition is not None:
        return text, flags, repetition

    if latin_ratio(text) < rejection.latin_ratio:
        return text, flags, "non_latin_output"

    return text, flags, None


# ---------------------------------------------------------------------------
# JSONL streams (§D.6)
# ---------------------------------------------------------------------------


def _record_mapping(record: Mapping[str, Any] | BaseModel) -> dict[str, Any]:
    if isinstance(record, BaseModel):
        return record.model_dump(mode="json")
    return dict(record)


def _json_line(record: Mapping[str, Any] | BaseModel) -> bytes:
    payload = json.dumps(
        _record_mapping(record),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return (payload + "\n").encode("utf-8")


def append_record(path: Path, record: Mapping[str, Any] | BaseModel) -> None:
    """Append one canonical JSON line and fsync it (§D.6).

    Parent directories are created. The line is emitted with a single
    ``O_APPEND`` write so a crash can only leave a trailing partial line, which
    :func:`load_stream` detects and truncates.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _json_line(record)
    with path.open("ab") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _empty_stream_stats(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "lines": 0,
        "records": 0,
        "samples": 0,
        "failures": 0,
        "truncated_partial_lines": 0,
        "sample_ids": 0,
    }


def _validate_stream_record(obj: Mapping[str, Any], path: Path, line_number: int) -> dict[str, Any]:
    status = obj.get("status")
    model = GenerationFailure if status == "rejected" else GenerationRecord
    try:
        record = model.model_validate(obj)
    except ValidationError as exc:
        raise ResumeCorruptionError(
            f"{path}:{line_number}: invalid record: {exc}"
        ) from exc

    data = record.model_dump(mode="json")
    expected_id = make_sample_id(
        str(data["prompt_id"]),
        str(data["generator_id"]),
        str(data["model_revision"]),
        str(data["decoding_config_id"]),
    )
    if data["sample_id"] != expected_id:
        raise ResumeCorruptionError(
            f"{path}:{line_number}: sample_id {data['sample_id']!r} does not "
            f"recompute (expected {expected_id!r})"
        )

    generated_text = data.get("generated_text")
    if generated_text is not None:
        expected_hash = hashlib.sha256(generated_text.encode("utf-8")).hexdigest()
        stored_hash = data.get("generated_text_hash")
        if stored_hash != expected_hash:
            raise ResumeCorruptionError(
                f"{path}:{line_number}: generated_text_hash {stored_hash!r} does "
                f"not recompute (expected {expected_hash!r})"
            )

    return data


def load_stream(path: Path) -> tuple[list[dict], dict]:
    """Read one JSONL stream, truncating a torn final line and validating all.

    Every line is parsed and validated. A trailing partial line (no final newline)
    is truncated to the last newline and counted in ``truncated_partial_lines``.
    A duplicate ``sample_id`` among successful records, or any recomputed-hash
    mismatch, raises :class:`ResumeCorruptionError`. No bad line is ever dropped
    silently.
    """
    path = Path(path)
    if not path.exists():
        return [], _empty_stream_stats(path)

    data = path.read_bytes()
    truncated = 0
    if data and not data.endswith(b"\n"):
        last_newline = data.rfind(b"\n")
        data = data[: last_newline + 1] if last_newline >= 0 else b""
        truncated = 1

    records: list[dict] = []
    seen_sample_ids: set[str] = set()
    samples = 0
    failures = 0
    lines = data.split(b"\n")
    if lines and lines[-1] == b"":
        lines.pop()

    for line_number, line in enumerate(lines, start=1):
        try:
            obj = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ResumeCorruptionError(
                f"{path}:{line_number}: invalid JSON: {exc}"
            ) from exc
        if not isinstance(obj, Mapping):
            raise ResumeCorruptionError(
                f"{path}:{line_number}: record is not a JSON object"
            )

        record = _validate_stream_record(obj, path, line_number)
        if record.get("status") == "rejected":
            failures += 1
        else:
            samples += 1
            sample_id = record["sample_id"]
            if sample_id in seen_sample_ids:
                raise ResumeCorruptionError(
                    f"{path}:{line_number}: duplicate sample_id {sample_id!r} "
                    f"in successful samples stream"
                )
            seen_sample_ids.add(sample_id)
        records.append(record)

    return records, {
        "path": str(path),
        "lines": len(lines),
        "records": len(records),
        "samples": samples,
        "failures": failures,
        "truncated_partial_lines": truncated,
        "sample_ids": len(seen_sample_ids),
    }


# ---------------------------------------------------------------------------
# State checkpoint (§D.6)
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _base_state(expected: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": STATE_SCHEMA_VERSION,
        "config_digest": expected.get("config_digest"),
        "prompt_manifest_sha256": expected.get("prompt_manifest_sha256"),
        "split_digest": expected.get("split_digest"),
        "streams": {},
        "state_repaired": 0,
        "updated_at": _now_iso(),
    }


def _reconcile_state(state: dict[str, Any], actual_streams: Mapping[str, Any]) -> dict[str, Any]:
    """Make ``state`` agree with the JSONL streams (the JSONL always wins)."""
    repaired = 0
    streams = state.setdefault("streams", {})
    for stream_id, actual in actual_streams.items():
        entry = streams.setdefault(
            stream_id, {"ok": [], "failed": {}, "last_offset": 0}
        )
        actual_ok = sorted(str(value) for value in actual.get("ok", []))
        actual_failed = {
            str(key): int(value) for key, value in dict(actual.get("failed", {})).items()
        }
        stale = set(entry.get("ok", [])) - set(actual_ok)
        repaired += len(stale)
        entry["ok"] = actual_ok
        entry["failed"] = dict(sorted(actual_failed.items()))
        entry["ok_count"] = len(actual_ok)
        entry["failed_count"] = len(actual_failed)
        entry["last_offset"] = int(actual.get("last_offset", entry.get("last_offset", 0)))

    if repaired:
        state["state_repaired"] = int(state.get("state_repaired", 0)) + repaired
    state["updated_at"] = _now_iso()
    return state


def load_state(path: Path, expected: Mapping[str, Any]) -> dict:
    """Load ``state.json``, refusing digest mismatches and reconciling with JSONL.

    ``expected`` carries the three run digests and may additionally carry a
    ``streams`` mapping of the actual JSONL contents. When present, state ids
    absent from JSONL are removed and counted in ``state_repaired``, while ids in
    JSONL but missing from state are added.
    """
    path = Path(path)
    if not path.exists():
        state = _base_state(expected)
        if "streams" in expected:
            _reconcile_state(state, expected["streams"])
        return state

    state = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(state, dict):
        raise ResumeCorruptionError(f"{path}: state is not a JSON object")

    for field in ("config_digest", "prompt_manifest_sha256", "split_digest"):
        if state.get(field) != expected.get(field):
            raise ResumeMismatch(
                f"resume refused: {field} changed "
                f"({state.get(field)!r} != {expected.get(field)!r})"
            )

    state.setdefault("streams", {})
    state.setdefault("state_repaired", 0)
    if "streams" in expected:
        _reconcile_state(state, expected["streams"])
    else:
        state["updated_at"] = _now_iso()
    return state


def save_state(path: Path, state: Mapping[str, Any]) -> None:
    """Atomically write ``state.json``: temp file + fsync + ``os.replace``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(state)
    payload["updated_at"] = _now_iso()
    data = (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")

    descriptor, temp_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise


def _atomic_write_json(path: Path, obj: Mapping[str, Any]) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (
        json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    descriptor, temp_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# Planning helpers (§D.5, §D.6)
# ---------------------------------------------------------------------------


def batch_seed(config_seed: str, stream_id: str, first_sample_id: str) -> int:
    """Per-batch seed (§D.5): recorded, not promised to reproduce sampling."""
    digest = hashlib.sha256(
        f"{config_seed}\x1f{stream_id}\x1f{first_sample_id}".encode("utf-8")
    ).hexdigest()
    return int(digest[:8], 16) & 0x7FFFFFFF


def stream_id_for(generator_id: str, decoding_config_id: str) -> str:
    return f"{generator_id}__{decoding_config_id}"


def stream_paths(
    run_dir: Path, generator_id: str, decoding_config_id: str
) -> dict[str, Path]:
    stream_id = stream_id_for(generator_id, decoding_config_id)
    return {
        "stream_id": stream_id,
        "samples": Path(run_dir) / f"samples-{stream_id}.jsonl",
        "failures": Path(run_dir) / f"failures-{stream_id}.jsonl",
    }


def planned_streams(
    run_dir: Path,
    generator_ids: Sequence[str],
    decoding_ids: Sequence[str],
) -> list[dict[str, str]]:
    """The 8 file entries: 2 generators x 2 decodings x {samples, failures}."""
    out: list[dict[str, str]] = []
    for generator_id in generator_ids:
        for decoding_id in decoding_ids:
            paths = stream_paths(run_dir, generator_id, decoding_id)
            out.append(
                {
                    "stream_id": paths["stream_id"],
                    "kind": "samples",
                    "path": str(paths["samples"]),
                }
            )
            out.append(
                {
                    "stream_id": paths["stream_id"],
                    "kind": "failures",
                    "path": str(paths["failures"]),
                }
            )
    return out


def expected_sample_ids_for(
    prompts: Sequence[Mapping[str, Any]],
    config: GenerationConfig,
    generator_ids: Sequence[str] | None = None,
    decoding_ids: Sequence[str] | None = None,
) -> list[str]:
    generators = [
        generator
        for generator in config.generators
        if generator_ids is None or generator.generator_id in set(generator_ids)
    ]
    decodings = [
        decoding
        for decoding in config.decodings
        if decoding_ids is None or decoding.decoding_config_id in set(decoding_ids)
    ]
    ids: list[str] = []
    for prompt in prompts:
        prompt_id = str(prompt["prompt_id"])
        for generator in generators:
            for decoding in decodings:
                ids.append(
                    make_sample_id(
                        prompt_id,
                        generator.generator_id,
                        generator.model_revision,
                        decoding.decoding_config_id,
                    )
                )
    return ids


def smoke_rank(prompt_id: str) -> str:
    return hashlib.sha256(f"{SMOKE_SEED}\x1f{prompt_id}".encode("utf-8")).hexdigest()


def _partition_quotas(n: int) -> dict[str, int]:
    if n >= 3:
        return {"train": n - 2, "val": 1, "test_holdout": 1}
    if n == 2:
        return {"train": 1, "val": 1, "test_holdout": 0}
    return {"train": max(n, 0), "val": 0, "test_holdout": 0}


def select_smoke_prompts(
    prompts: Sequence[Mapping[str, Any]],
    n: int,
    assignments: Mapping[str, str],
) -> list[Mapping[str, Any]]:
    """Deterministically pick ``n`` prompts with the §E partition coverage.

    Ranking is ``sha256("e2c1-smoke-v1\\x1f"+prompt_id)`` so the same prompts are
    chosen regardless of manifest order. For ``n == 4`` the quota is 2 train / 1
    val / 1 test_holdout; for ``n >= 3`` val and test_holdout each get at least
    one and the remainder goes to train.
    """
    if n <= 0:
        return []
    ranked = sorted(prompts, key=lambda row: (smoke_rank(str(row["prompt_id"])), str(row["prompt_id"])))

    buckets: dict[str, list[Mapping[str, Any]]] = {
        "train": [],
        "val": [],
        "test_holdout": [],
    }
    for row in ranked:
        partition = assignments.get(str(row["prompt_id"]))
        if partition in buckets:
            buckets[partition].append(row)

    quota = _partition_quotas(n)
    selected: list[Mapping[str, Any]] = []
    for partition in ("train", "val", "test_holdout"):
        selected.extend(buckets[partition][: quota[partition]])

    # Fall back deterministically if a small partition cannot fill its quota.
    selected_ids = {str(row["prompt_id"]) for row in selected}
    for row in ranked:
        if len(selected) >= n:
            break
        prompt_id = str(row["prompt_id"])
        if prompt_id not in selected_ids:
            selected.append(row)
            selected_ids.add(prompt_id)

    selected.sort(key=lambda row: (smoke_rank(str(row["prompt_id"])), str(row["prompt_id"])))
    return selected[:n]


def _is_under(path: Path, parent: Path) -> bool:
    try:
        Path(path).resolve().relative_to(Path(parent).resolve())
        return True
    except ValueError:
        return False


def _refuse_smoke_under_datasets(out_dir: str | Path) -> None:
    if _is_under(Path(out_dir), REPO_ROOT / "datasets"):
        raise SmokeOutputError(
            f"refusing --smoke output under datasets/: {Path(out_dir)!s} "
            "(smoke output is ephemeral and must never enter the corpus)"
        )


def _read_prompt_rows(path: Path) -> tuple[list[dict[str, Any]], str]:
    data = Path(path).read_bytes()
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(data.split(b"\n"), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise GenerationError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
        if not isinstance(row, dict):
            raise GenerationError(f"{path}:{line_number}: prompt row is not an object")
        rows.append(row)
    return rows, hashlib.sha256(data).hexdigest()


def _read_split(path: Path) -> dict[str, Any]:
    split = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(split, dict):
        raise GenerationError(f"{path}: split manifest is not an object")
    return split


def _select_generators_and_decodings(
    config: GenerationConfig,
    generator_ids: Sequence[str] | None,
    decoding_ids: Sequence[str] | None,
) -> tuple[list[GeneratorConfig], list[DecodingConfig]]:
    generators = list(config.generators)
    if generator_ids:
        wanted = set(generator_ids)
        known = {generator.generator_id for generator in generators}
        unknown = sorted(wanted - known)
        if unknown:
            raise GenerationError(f"unknown generator(s): {unknown}; known {sorted(known)}")
        generators = [
            generator for generator in generators if generator.generator_id in wanted
        ]

    decodings = list(config.decodings)
    if decoding_ids:
        wanted = set(decoding_ids)
        known = {decoding.decoding_config_id for decoding in decodings}
        unknown = sorted(wanted - known)
        if unknown:
            raise GenerationError(f"unknown decoding(s): {unknown}; known {sorted(known)}")
        decodings = [
            decoding for decoding in decodings if decoding.decoding_config_id in wanted
        ]
    return generators, decodings


def _run_identifier(config: GenerationConfig) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{config_digest(config)[:8]}"


def _build_plan(
    *,
    mode: str,
    config: GenerationConfig,
    prompts: Sequence[Mapping[str, Any]],
    assignments: Mapping[str, str],
    prompt_sha: str,
    split_digest: str,
    run_dir: Path,
    generators: Sequence[GeneratorConfig],
    decodings: Sequence[DecodingConfig],
    selected_prompts: Sequence[Mapping[str, Any]],
    total_expected_ids: int,
) -> dict[str, Any]:
    generator_ids = [generator.generator_id for generator in generators]
    decoding_ids = [decoding.decoding_config_id for decoding in decodings]
    sample_ids = expected_sample_ids_for(
        selected_prompts, config, generator_ids, decoding_ids
    )
    streams = planned_streams(run_dir, generator_ids, decoding_ids)
    return {
        "schema_version": "1.0",
        "mode": mode,
        "status": "planned",
        "config_id": config.config_id,
        "config_digest": config_digest(config),
        "prompt_manifest_sha256": prompt_sha,
        "split_digest": split_digest,
        "run_dir": str(run_dir),
        "n_prompts": len(selected_prompts),
        "n_total_prompts": len(prompts),
        "generators": generator_ids,
        "generator_specs": {
            generator.generator_id: {
                "model_id": generator.model_id,
                "model_revision": generator.model_revision,
                "architecture": generator.architecture,
                "loader": generator.loader,
            }
            for generator in generators
        },
        "decodings": decoding_ids,
        "streams": streams,
        "stream_count": len(streams),
        "expected_sample_ids": sample_ids,
        "sample_id_count": len(sample_ids),
        "unique_sample_ids": len(set(sample_ids)),
        "total_expected_sample_ids": total_expected_ids,
        "assignments_covered": len(assignments),
        "corpus_written": False,
    }


def _read_actual_streams(
    run_dir: Path,
    generators: Sequence[GeneratorConfig],
    decodings: Sequence[DecodingConfig],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, int]]]:
    """Read all streams once, returning (state-shaped actuals, per-stream counts)."""
    actual: dict[str, dict[str, Any]] = {}
    counts: dict[str, dict[str, int]] = {}
    for generator in generators:
        for decoding in decodings:
            paths = stream_paths(run_dir, generator.generator_id, decoding.decoding_config_id)
            samples, sample_stats = load_stream(paths["samples"])
            failures, failure_stats = load_stream(paths["failures"])
            ok_ids = {row["sample_id"] for row in samples}
            latest_attempt: dict[str, int] = {}
            for row in failures:
                sample_id = row["sample_id"]
                latest_attempt[sample_id] = max(
                    latest_attempt.get(sample_id, 0), int(row["attempt"])
                )
            stream_id = paths["stream_id"]
            actual[stream_id] = {
                "ok": sorted(ok_ids),
                "failed": dict(sorted(latest_attempt.items())),
                "last_offset": paths["samples"].stat().st_size
                if paths["samples"].exists()
                else 0,
            }
            counts[stream_id] = {
                "ok": len(ok_ids),
                "failed": len(latest_attempt),
                "truncated_partial_lines": sample_stats["truncated_partial_lines"]
                + failure_stats["truncated_partial_lines"],
            }
    return actual, counts


def _now_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Model loading / encoding helpers
# ---------------------------------------------------------------------------


def _from_pretrained_with_dtype(
    model_cls: Any,
    model_id: str,
    *,
    dtype: Any,
    **kwargs: Any,
) -> Any:
    """Load a model with the transformers-5 ``dtype`` kwarg, falling back once.

    Transformers 5 renamed the ``from_pretrained`` dtype kwarg from ``torch_dtype``
    to ``dtype``. Older releases reject the unknown keyword with ``TypeError``, so
    retry with the legacy name only for that specific failure.
    """
    try:
        return model_cls.from_pretrained(model_id, dtype=dtype, **kwargs)
    except TypeError as exc:
        message = str(exc).lower()
        # Only retry when ``dtype`` itself is the offending keyword. Other
        # unsupported keywords (e.g. ``attn_implementation``) must reach the
        # attention fallback, not trigger a pointless torch_dtype retry.
        if "dtype" not in message:
            raise
    return model_cls.from_pretrained(model_id, torch_dtype=dtype, **kwargs)


_ATTN_IMPLEMENTATION = "sdpa"
_ATTN_FALLBACK_ERRORS = (TypeError, ValueError, NotImplementedError)


def _load_with_attn(
    loader: Any,
    model_id: str,
    *,
    attn_state: dict[str, Any] | None = None,
    **kwargs: Any,
) -> Any:
    """Load a model requesting SDPA attention, falling back once if unsupported.

    ``attn_implementation="sdpa"`` is requested unconditionally. A loader that
    raises ``TypeError``/``ValueError``/``NotImplementedError`` is retried without
    the keyword; ``attn_state`` records ``"sdpa"`` or ``"default"`` for the report.
    """
    try:
        model = loader(
            model_id, attn_implementation=_ATTN_IMPLEMENTATION, **kwargs
        )
    except _ATTN_FALLBACK_ERRORS:
        model = loader(model_id, **kwargs)
        if attn_state is not None:
            attn_state["attn_implementation"] = "default"
    else:
        if attn_state is not None:
            attn_state["attn_implementation"] = _ATTN_IMPLEMENTATION
    return model


def _load_generator(
    generator: GeneratorConfig,
    *,
    torch: Any,
    transformers: Any,
    auto_tokenizer: Any,
    auto_model_causal: Any,
    auto_processor: Any,
    auto_model_image_text: Any,
    attn_state: dict[str, Any] | None = None,
) -> tuple[Any | None, Any, Any, bool]:
    """Load ``(processor, tokenizer, model, multimodal)`` for one generator.

    The model/tokenizer classes are injected from :func:`run_generation` (where the
    only ``transformers`` imports live), so this helper stays torch-free at module
    import time and is directly testable with fakes. A multimodal processor also
    supplies the chat template and decoder through its ``tokenizer``.

    ``attn_state`` (when supplied) receives ``attn_implementation`` so the caller
    can report which attention implementation was actually used.
    """

    def load_with_dtype(model_cls: Any, model_dtype: Any, **kwargs: Any) -> Any:
        return _load_with_attn(
            lambda model_id, **extra: _from_pretrained_with_dtype(
                model_cls, model_id, dtype=model_dtype, **extra
            ),
            generator.model_id,
            attn_state=attn_state,
            **kwargs,
        )

    if generator.loader == "multimodal":
        processor = auto_processor.from_pretrained(
            generator.model_id, revision=generator.model_revision
        )
        tokenizer = getattr(processor, "tokenizer", processor)
        model = load_with_dtype(
            auto_model_image_text,
            torch.float16,
            revision=generator.model_revision,
            device_map="cuda",
        )
        multimodal = True
    else:
        tokenizer = auto_tokenizer.from_pretrained(
            generator.model_id, revision=generator.model_revision
        )
        load_kwargs: dict[str, Any] = {"revision": generator.model_revision}
        quantisation = generator.quantisation.model_dump(mode="json")
        method = str(quantisation.get("method", "none"))
        dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }.get(str(generator.dtype), torch.float16)
        if method.startswith("bitsandbytes"):
            compute_dtype = {
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
                "float32": torch.float32,
            }.get(str(quantisation.get("compute_dtype", generator.dtype)), torch.float16)
            load_kwargs["quantization_config"] = transformers.BitsAndBytesConfig(
                load_in_4bit=bool(quantisation.get("load_in_4bit", True)),
                bnb_4bit_quant_type=str(quantisation.get("quant_type", "nf4")),
                bnb_4bit_use_double_quant=bool(
                    quantisation.get("double_quant", True)
                ),
                bnb_4bit_compute_dtype=compute_dtype,
            )
            model = _load_with_attn(
                auto_model_causal.from_pretrained,
                generator.model_id,
                attn_state=attn_state,
                **load_kwargs,
            )
        else:
            model = load_with_dtype(auto_model_causal, dtype, **load_kwargs)
        processor = None
        multimodal = False

    if getattr(tokenizer, "pad_token_id", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
    return processor, tokenizer, model, multimodal


def _encode_rendered(
    rendered: str,
    *,
    processor: Any | None,
    tokenizer: Any,
    multimodal: bool,
) -> Any:
    """Encode one rendered prompt with the multimodal processor or tokenizer."""
    if multimodal:
        return processor(text=[rendered], return_tensors="pt")
    return tokenizer(rendered, return_tensors="pt")


def _decode_new_tokens(
    output_ids: Any,
    prompt_length: int,
    tokenizer: Any,
    row: int = 0,
) -> tuple[list[int], str]:
    """Decode only the continuation, never the prompt tokens."""
    token_ids = output_ids[row, prompt_length:].tolist()
    return token_ids, tokenizer.decode(token_ids, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Generation runner
# ---------------------------------------------------------------------------


def run_generation(
    config: GenerationConfig | Mapping[str, Any] | str | Path,
    *,
    prompts_path: str | Path | None = None,
    splits_path: str | Path | None = None,
    out_dir: str | Path | None = None,
    generator_ids: Sequence[str] | None = None,
    decoding_ids: Sequence[str] | None = None,
    smoke: int = 0,
    dry_run: bool = False,
    plan_only: bool = False,
    retry_failed: bool = False,
    max_prompts: int | None = None,
    authorise_full_run: bool = False,
    report_path: str | Path | None = None,
    registry: Mapping[str, Any] | str | Path | None = None,
) -> dict:
    """Plan or execute the §D generation matrix.

    ``dry_run``/``plan_only`` return the plan *without importing torch*. A full
    run (no ``smoke`` and neither plan mode) raises :class:`AuthorisationError`
    unless ``authorise_full_run`` is true. ``smoke`` output under ``datasets/`` is
    refused. Real torch/transformers imports happen only after all of these checks.
    """
    cfg = _as_config(config)
    if prompts_path is None:
        prompts_path = REPO_ROOT / cfg.prompt_manifest
    if splits_path is None:
        splits_path = REPO_ROOT / cfg.split_manifest

    if smoke and out_dir is not None:
        _refuse_smoke_under_datasets(out_dir)
    if smoke and out_dir is None:
        raise SmokeOutputError(
            "refusing --smoke without --out: the default full-run directory is "
            "under datasets/, and smoke output must never enter the corpus"
        )
    if not smoke and not dry_run and not plan_only and not authorise_full_run:
        raise AuthorisationError(
            "refusing a full 20,000-sample run without --authorise-full-run; "
            "the flag exists so the notebook cannot start a corpus by accident"
        )

    prompts, prompt_sha = _read_prompt_rows(Path(prompts_path))
    split = _read_split(Path(splits_path))
    assignments_raw = split.get("assignments") or {}
    if not isinstance(assignments_raw, Mapping):
        raise GenerationError("split manifest has no assignments object")
    assignments = {str(key): str(value) for key, value in assignments_raw.items()}
    split_salt = str(split.get("salt", SPLIT_SALT))
    computed_split_digest = e2c1_split_digest(assignments, split_salt)
    declared_split_digest = split.get("split_digest")
    if declared_split_digest is not None and declared_split_digest != computed_split_digest:
        raise ResumeMismatch(
            f"split_digest mismatch: {declared_split_digest!r} != "
            f"{computed_split_digest!r}"
        )
    declared_prompt_sha = split.get("prompt_manifest_sha256")
    if declared_prompt_sha is not None and declared_prompt_sha != prompt_sha:
        raise ResumeMismatch(
            f"prompt_manifest_sha256 mismatch: {declared_prompt_sha!r} != "
            f"{prompt_sha!r}"
        )

    generators, decodings = _select_generators_and_decodings(
        cfg, generator_ids, decoding_ids
    )

    ordered_prompts = sorted(prompts, key=lambda row: str(row["prompt_id"]))
    if max_prompts is not None:
        ordered_prompts = ordered_prompts[: max(0, int(max_prompts))]

    for prompt in ordered_prompts:
        prompt_id = str(prompt["prompt_id"])
        if prompt_id not in assignments:
            raise GenerationError(f"prompt {prompt_id} has no partition assignment")

    total_expected_ids = len(
        expected_sample_ids_for(
            prompts,
            cfg,
            [generator.generator_id for generator in generators],
            [decoding.decoding_config_id for decoding in decodings],
        )
    )

    if smoke:
        selected_prompts = select_smoke_prompts(ordered_prompts, int(smoke), assignments)
        mode = "smoke"
    else:
        selected_prompts = ordered_prompts
        mode = "dry-run" if dry_run else ("plan-only" if plan_only else "full")

    base_dir = Path(out_dir) if out_dir is not None else DEFAULT_OUT_DIR
    run_dir = base_dir
    if mode == "full":
        # One directory per full run (§D.6) unless the caller named a run dir.
        if not re.search(r"\d{8}T\d{6}Z-[0-9a-f]{8}$", str(run_dir)):
            run_dir = base_dir / _run_identifier(cfg)

    plan = _build_plan(
        mode=mode,
        config=cfg,
        prompts=prompts,
        assignments=assignments,
        prompt_sha=prompt_sha,
        split_digest=computed_split_digest,
        run_dir=run_dir,
        generators=generators,
        decodings=decodings,
        selected_prompts=selected_prompts,
        total_expected_ids=total_expected_ids,
    )

    if dry_run or plan_only:
        if report_path is not None:
            _atomic_write_json(Path(report_path), plan)
        return plan

    if registry is not None:
        validate_config_against_licences(cfg, registry)

    # ------------------------------------------------------------------
    # Everything below may touch a model; torch is imported only here.
    # ------------------------------------------------------------------
    import torch  # noqa: PLC0415 -- deliberate: import-safe module
    import transformers  # noqa: PLC0415
    from transformers import (  # noqa: PLC0415
        AutoModelForCausalLM,
        AutoModelForImageTextToText,
        AutoProcessor,
        AutoTokenizer,
    )

    run_dir.mkdir(parents=True, exist_ok=True)
    state_path = run_dir / "state.json"
    expected_digests = {
        "config_digest": config_digest(cfg),
        "prompt_manifest_sha256": prompt_sha,
        "split_digest": computed_split_digest,
    }

    actual_streams, stream_counts = _read_actual_streams(run_dir, generators, decodings)
    if state_path.exists():
        state = load_state(
            state_path, {**expected_digests, "streams": actual_streams}
        )
    else:
        state = _base_state(expected_digests)
        _reconcile_state(state, actual_streams)

    completed: dict[str, set[str]] = {
        stream_id: set(actual.get("ok", []))
        for stream_id, actual in actual_streams.items()
    }
    failed_attempts: dict[str, dict[str, int]] = {
        stream_id: dict(actual.get("failed", {}))
        for stream_id, actual in actual_streams.items()
    }

    records_since_state = 0
    n_records = 0
    n_failures = 0
    failure_report: list[dict[str, Any]] = []
    per_sample_seconds: list[float] = []
    attn_implementations: dict[str, str] = {}
    thinking_disabled_by_generator: dict[str, bool] = {}
    successes_by_generator: dict[str, int] = {}
    failures_by_generator: dict[str, int] = {}
    rejections: collections.Counter[str] = collections.Counter()
    rejected_examples: list[dict[str, Any]] = []

    def flush_state() -> None:
        nonlocal records_since_state
        save_state(state_path, state)
        records_since_state = 0

    try:
        for generator in generators:
            attn_state: dict[str, Any] = {}
            processor, tokenizer, model, multimodal = _load_generator(
                generator,
                torch=torch,
                transformers=transformers,
                auto_tokenizer=AutoTokenizer,
                auto_model_causal=AutoModelForCausalLM,
                auto_processor=AutoProcessor,
                auto_model_image_text=AutoModelForImageTextToText,
                attn_state=attn_state,
            )
            attn_implementations[generator.generator_id] = str(
                attn_state.get("attn_implementation", "default")
            )
            successes_by_generator.setdefault(generator.generator_id, 0)
            failures_by_generator.setdefault(generator.generator_id, 0)
            thinking_state: dict[str, bool] = {}
            quantisation = generator.quantisation.model_dump(mode="json")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            if not multimodal:
                model.to(device)
            model.eval()

            chat_template = getattr(tokenizer, "chat_template", None) or ""
            chat_sha = _sha256_text(str(chat_template))
            config_digest_value = config_digest(cfg)

            for decoding in decodings:
                paths = stream_paths(
                    run_dir, generator.generator_id, decoding.decoding_config_id
                )
                stream_id = paths["stream_id"]
                ok_ids = completed.setdefault(stream_id, set())
                failed = failed_attempts.setdefault(stream_id, {})

                pending = []
                for prompt in selected_prompts:
                    prompt_id = str(prompt["prompt_id"])
                    sample_id = make_sample_id(
                        prompt_id,
                        generator.generator_id,
                        generator.model_revision,
                        decoding.decoding_config_id,
                    )
                    if sample_id in ok_ids:
                        continue
                    if sample_id in failed and not retry_failed:
                        continue
                    pending.append((prompt, sample_id))

                for start in range(0, len(pending), cfg.batch_size):
                    batch = pending[start : start + cfg.batch_size]
                    if not batch:
                        continue
                    first_sample_id = batch[0][1]
                    seed = batch_seed(cfg.seed, stream_id, first_sample_id)
                    try:
                        transformers.set_seed(seed)
                    except AttributeError:  # older transformers
                        import random  # noqa: PLC0415

                        random.seed(seed)
                        torch.manual_seed(seed)

                    prepared: list[dict[str, Any]] = []
                    for prompt, sample_id in batch:
                        prompt_id = str(prompt["prompt_id"])
                        source_text = normalise_text(str(prompt.get("prompt_text", "")))
                        built: dict[str, Any] = {
                            "prompt": prompt,
                            "sample_id": sample_id,
                            "render_error": None,
                            "render_flags": [],
                            "rendered": None,
                            "input_ids": None,
                        }
                        try:
                            rendered = render_prompt(
                                source_text,
                                tokenizer,
                                cfg,
                                built["render_flags"],
                                thinking_state=thinking_state,
                            )
                            built["rendered"] = rendered
                            encoded = _encode_rendered(
                                rendered,
                                processor=processor,
                                tokenizer=tokenizer,
                                multimodal=multimodal,
                            )
                            input_ids = encoded["input_ids"][0]
                            if int(input_ids.shape[-1]) > cfg.max_input_tokens:
                                raise PromptRejection(
                                    "prompt_too_long",
                                    f"{int(input_ids.shape[-1])} > {cfg.max_input_tokens}",
                                )
                            built["input_ids"] = input_ids
                        except PromptRejection as exc:
                            built["render_error"] = exc
                        prepared.append(built)

                    runnable = [item for item in prepared if item["render_error"] is None]
                    generated: dict[int, tuple[str, str]] = {}
                    if runnable:
                        pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
                        max_len = max(int(item["input_ids"].shape[-1]) for item in runnable)
                        input_ids = torch.full(
                            (len(runnable), max_len), pad_id, dtype=torch.long
                        )
                        attention = torch.zeros_like(input_ids)
                        for index, item in enumerate(runnable):
                            ids = item["input_ids"]
                            length = int(ids.shape[-1])
                            input_ids[index, max_len - length :] = ids
                            attention[index, max_len - length :] = 1
                        input_ids = input_ids.to(device)
                        attention = attention.to(device)

                        sample_kwargs: dict[str, Any] = {
                            "max_new_tokens": decoding.max_new_tokens,
                            "do_sample": decoding.do_sample,
                            "pad_token_id": pad_id,
                            "use_cache": True,
                        }
                        if decoding.do_sample:
                            sample_kwargs["temperature"] = decoding.temperature
                            sample_kwargs["top_p"] = decoding.top_p

                        began = time.perf_counter()
                        with torch.inference_mode():
                            output = model.generate(
                                input_ids=input_ids,
                                attention_mask=attention,
                                **sample_kwargs,
                            )
                        elapsed = time.perf_counter() - began
                        per_sample_seconds.append(elapsed / max(1, len(runnable)))
                        for index, item in enumerate(runnable):
                            token_ids, raw = _decode_new_tokens(
                                output, max_len, tokenizer, row=index
                            )
                            eos_id = tokenizer.eos_token_id
                            finish_reason = "eos" if eos_id in token_ids else "length"
                            generated[id(item)] = (raw, finish_reason)

                    for batch_index, item in enumerate(prepared):
                        prompt = item["prompt"]
                        prompt_id = str(prompt["prompt_id"])
                        sample_id = item["sample_id"]
                        partition = assignments[prompt_id]
                        topic = str(prompt.get("topic_category", "other"))
                        source_prompt_hash = str(
                            prompt.get(
                                "prompt_hash", _sha256_text(str(prompt.get("prompt_text", "")))
                            )
                        )
                        attempt = int(failed.get(sample_id, 0)) + 1
                        common = {
                            "sample_id": sample_id,
                            "prompt_id": prompt_id,
                            "generator_id": generator.generator_id,
                            "model_id": generator.model_id,
                            "model_revision": generator.model_revision,
                            "licence_ref": generator.generator_id,
                            "decoding_config_id": decoding.decoding_config_id,
                            "temperature": decoding.temperature,
                            "top_p": decoding.top_p,
                            "max_new_tokens": decoding.max_new_tokens,
                            "seed": seed,
                            "batch_index": batch_index,
                            "do_sample": decoding.do_sample,
                            "generation_timestamp": _now_timestamp(),
                            "partition": partition,
                            "topic_category": topic,
                            "source_prompt_hash": source_prompt_hash,
                            "quantisation": quantisation,
                            "prompt_template_id": cfg.prompt_template_id,
                            "chat_template_sha256": chat_sha,
                            "config_digest": config_digest_value,
                        }

                        error = item["render_error"]
                        if error is not None:
                            reason = error.reason
                            flags: list[str] = []
                            raw = ""
                            finish_reason = "length"
                            generated_text = None
                        else:
                            raw, finish_reason = generated[id(item)]
                            generated_text, flags, reason = postprocess(
                                raw, normalise_text(str(prompt.get("prompt_text", ""))),
                                finish_reason, cfg,
                            )

                        quality_flags = list(item["render_flags"]) + list(flags)

                        if reason is not None:
                            failure = GenerationFailure(
                                **common,
                                status="rejected",
                                rejection_reason=reason,
                                attempt=attempt,
                                raw_text_hash=_sha256_text(raw),
                                finish_reason=finish_reason,
                                quality_flags=quality_flags,
                                generated_text=generated_text or None,
                                generated_text_hash=(
                                    _sha256_text(generated_text)
                                    if generated_text
                                    else None
                                ),
                            )
                            append_record(paths["failures"], failure)
                            failed[sample_id] = attempt
                            n_failures += 1
                            rejections[reason] += 1
                            failures_by_generator[generator.generator_id] += 1
                            if len(rejected_examples) < 3:
                                rejected_text = generated_text or raw or ""
                                rejected_examples.append(
                                    {
                                        "generator_id": generator.generator_id,
                                        "decoding_config_id": decoding.decoding_config_id,
                                        "rejection_reason": reason,
                                        "finish_reason": finish_reason,
                                        "text_prefix": rejected_text[:120],
                                    }
                                )
                            failure_report.append(
                                {
                                    "sample_id": sample_id,
                                    "rejection_reason": reason,
                                    "attempt": attempt,
                                }
                            )
                            state["streams"].setdefault(stream_id, {})[
                                "failed"
                            ] = dict(sorted(failed.items()))
                            records_since_state += 1
                        else:
                            record = GenerationRecord(
                                **common,
                                generated_text=generated_text,
                                generated_text_hash=_sha256_text(generated_text),
                                finish_reason=finish_reason,
                                quality_flags=quality_flags,
                            )
                            append_record(paths["samples"], record)
                            ok_ids.add(sample_id)
                            n_records += 1
                            successes_by_generator[generator.generator_id] += 1
                            entry = state["streams"].setdefault(
                                stream_id, {"ok": [], "failed": {}, "last_offset": 0}
                            )
                            entry["ok"] = sorted(ok_ids)
                            entry["failed"] = dict(sorted(failed.items()))
                            entry["ok_count"] = len(ok_ids)
                            entry["failed_count"] = len(failed)
                            entry["last_offset"] = paths["samples"].stat().st_size
                            records_since_state += 1

                        if records_since_state >= STATE_FLUSH_EVERY:
                            flush_state()

            thinking_disabled_by_generator[generator.generator_id] = bool(
                thinking_state.get(THINKING_DISABLED_KEY, False)
            )
            flush_state()
            del model
            del tokenizer
            if processor is not None:
                del processor
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        if run_dir.exists():
            flush_state()

    elapsed_values = sorted(per_sample_seconds)
    median = (
        elapsed_values[len(elapsed_values) // 2] if elapsed_values else None
    )
    total_attempts = len(selected_prompts) * len(generators) * len(decodings)
    eta_hours = None
    if median is not None and median > 0:
        eta_hours = (median * total_expected_ids) / 3600.0

    # Reflect the reconciled state, not the pre-run snapshot.
    for stream_id, entry in state.get("streams", {}).items():
        stream_counts.setdefault(stream_id, {})
        stream_counts[stream_id]["ok"] = len(entry.get("ok", []))
        stream_counts[stream_id]["failed"] = len(entry.get("failed", {}))

    unique_attn = set(attn_implementations.values())
    attn_summary: Any = (
        next(iter(unique_attn))
        if len(unique_attn) == 1
        else dict(sorted(attn_implementations.items()))
    )
    diagnostics = {
        "rejections": dict(sorted(rejections.items())),
        "rejected_examples": rejected_examples,
        "successes_by_generator": dict(sorted(successes_by_generator.items())),
        "failures_by_generator": dict(sorted(failures_by_generator.items())),
    }

    report = {
        **plan,
        "status": "ok",
        "n_records": n_records,
        "n_failures": n_failures,
        "attempts": total_attempts,
        "records_plus_failures": n_records + n_failures,
        "failures": failure_report,
        "stream_counts": stream_counts,
        "state_repaired": state.get("state_repaired", 0),
        "corpus_written": mode == "full",
        "thinking_disabled": any(thinking_disabled_by_generator.values()),
        "thinking_disabled_by_generator": dict(
            sorted(thinking_disabled_by_generator.items())
        ),
        "attn_implementation": attn_summary,
        "attn_implementation_by_generator": dict(
            sorted(attn_implementations.items())
        ),
        **diagnostics,
        "full_run": {
            "authorised": bool(authorise_full_run),
            "requested_n": total_expected_ids,
        },
        "quantisation": {
            generator.generator_id: generator.quantisation.model_dump(mode="json")
            for generator in generators
        },
        "timings": {
            "per_sample_s": list(per_sample_seconds),
            "median_s": median,
            "eta_full_hours": eta_hours,
        },
    }
    if mode == "smoke":
        report["smoke"] = {
            "n_prompts_per_generator": len(selected_prompts),
            "n_records": n_records,
            "prompt_ids": [str(prompt["prompt_id"]) for prompt in selected_prompts],
            "failures": failure_report,
            **diagnostics,
        }

    if report_path is not None:
        _atomic_write_json(Path(report_path), report)
    return report
