"""Shared reader contract, manifest loading and raw->canonical mapping.

Separation of concerns:
  * a *reader* only parses a source layout and yields ``RawSample`` objects;
  * this module attaches provenance from the committed manifest, normalises text,
    computes stable ids and emits ``CanonicalRecord``.

That split means a reader never hard-codes a licence or a version -- provenance
always comes from the reviewed manifest, so a licence change is a one-line diff in
a YAML file rather than a hunt through Python.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Protocol

import yaml

from ..normalise import normalise_and_clean, word_count
from ..schema import (
    CanonicalRecord,
    CollectionMethod,
    Modality,
    Split,
    Task,
    TRANSFORMATION_VERSION,
    make_content_hash,
    make_sample_id,
)


@dataclass
class RawSample:
    """A parsed record before provenance and normalisation are attached."""

    source_sample_id: str
    original_label: str
    label: int
    task: Task
    group_id: str
    text: str | None = None
    domain: str | None = None
    generator: str | None = None
    language: str = "en"
    source_uri: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    modality: Modality = Modality.TEXT
    media_hash: str | None = None
    width: int | None = None
    height: int | None = None
    duration_s: float | None = None
    fps: float | None = None
    identity_group: str | None = None
    frame_sampling: dict[str, Any] | None = None


class Reader(Protocol):
    name: str

    def read(self, root: Path) -> Iterator[RawSample]: ...


@dataclass
class DatasetManifest:
    """Parsed provenance YAML. Only fields the pipeline actually needs."""

    dataset_id: str
    version: str
    licence_spdx: str
    licence_evidence_url: str
    collection_method: CollectionMethod
    permitted_train: bool
    permitted_eval_only: bool
    redistribute_original: bool
    commercial_use: bool
    attribution_required: bool
    unresolved: bool = False
    notes: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> "DatasetManifest":
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        lic = data.get("licence", {})
        permits = data.get("permitted_uses", {})
        return cls(
            dataset_id=data["dataset_id"],
            version=str(data.get("version", {}).get("value", "unpinned")),
            licence_spdx=str(lic.get("spdx", "UNKNOWN")),
            licence_evidence_url=str(lic.get("evidence_url", "")),
            collection_method=CollectionMethod(data["access_method"]),
            permitted_train=bool(permits.get("train", False)),
            permitted_eval_only=bool(permits.get("internal_eval", False)),
            redistribute_original=bool(lic.get("redistribution") == "permitted"),
            commercial_use=bool(lic.get("commercial_use") in ("permitted", True)),
            attribution_required=bool(lic.get("attribution_required", True)),
            unresolved=bool(lic.get("unresolved", False)),
            notes=str(data.get("notes", "")),
            raw=data,
        )

    def licence_ref(self) -> str:
        return f"{self.licence_spdx}::{self.licence_evidence_url}" if self.licence_evidence_url else self.licence_spdx


# Records shorter than this are excluded: short text is where detectors fail and
# where a confident-looking score is most misleading (see research doc 15 s15.4).
MIN_WORDS = 40


def build_record(
    raw: RawSample,
    manifest: DatasetManifest,
    split: Split = Split.EXCLUDED,
) -> CanonicalRecord:
    """Attach provenance, normalise, and compute stable ids."""
    if raw.text is not None:
        text, clean_flags = normalise_and_clean(raw.text)
    else:
        text, clean_flags = None, []

    if raw.modality is Modality.TEXT:
        content_hash = make_content_hash(text or "")
    else:
        content_hash = raw.media_hash or make_content_hash(raw.source_sample_id)

    meta = dict(raw.metadata)
    meta["word_count"] = word_count(text or "") if text is not None else 0
    if clean_flags:
        meta["contamination_flags"] = clean_flags

    return CanonicalRecord(
        sample_id=make_sample_id(manifest.dataset_id, manifest.version, raw.source_sample_id),
        modality=raw.modality,
        task=raw.task,
        label=raw.label,
        original_label=raw.original_label,
        source_dataset=manifest.dataset_id,
        source_version=manifest.version,
        source_sample_id=raw.source_sample_id,
        source_uri=raw.source_uri,
        content_hash=content_hash,
        group_id=raw.group_id,
        domain=raw.domain,
        language=raw.language,
        generator=raw.generator,
        collection_method=manifest.collection_method,
        licence_ref=manifest.licence_ref(),
        split=split,
        metadata=meta,
        transformation_version=TRANSFORMATION_VERSION,
        text=text,
        media_hash=raw.media_hash,
        width=raw.width,
        height=raw.height,
        duration_s=raw.duration_s,
        fps=raw.fps,
        identity_group=raw.identity_group,
        frame_sampling=raw.frame_sampling,
    )


def quality_flags(rec: CanonicalRecord) -> list[str]:
    """Flags a record as unusable without deleting the evidence it existed."""
    flags: list[str] = []
    wc = int(rec.metadata.get("word_count", 0))
    if rec.modality is Modality.TEXT and wc < MIN_WORDS:
        flags.append("too_short")
    if rec.modality is Modality.TEXT and not rec.content_hash:
        flags.append("empty_text")
    if rec.label not in (0, 1, 2):
        flags.append("invalid_label")
    if rec.label == 0 and rec.generator:
        # a "human" record carrying a generator id is a mapping bug
        flags.append("label_generator_conflict")
    if rec.label == 1 and not rec.generator:
        flags.append("missing_generator")
    return flags
