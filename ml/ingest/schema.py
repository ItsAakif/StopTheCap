"""Canonical schema shared by every dataset reader.

Design rules (see research/14-canonical-schema-and-splits.md):

1. ``original_label`` is never overwritten -- ``label`` is additive.
2. ``split`` lives on the record so leakage is auditable by query.
3. Grouping keys are mandatory where the source provides them, and explicitly
   ``None`` otherwise (a null key is a documented leakage risk, not an oversight).
4. Unknown metadata is never invented; absent values stay ``None``.
"""

from __future__ import annotations

import hashlib
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

TRANSFORMATION_VERSION = "nfkc-quotes-v1"


class Modality(str, Enum):
    TEXT = "text"
    VIDEO = "video"


class Task(str, Enum):
    """A task is a label space. Never mix two tasks in one model."""

    TEXT_BINARY = "text_binary"                # human vs machine text
    TEXT_GENERATOR_ATTRIBUTION = "text_generator_attribution"  # which model wrote it
    TEXT_MIXED_BOUNDARY = "text_mixed_boundary"  # human<->machine boundary in one doc
    VIDEO_GENERATED = "video_generated"        # fully AI-generated video
    VIDEO_MANIPULATED = "video_manipulated"    # real footage, manipulated (deepfake)


class Split(str, Enum):
    TRAIN = "train"
    VAL = "val"
    TEST_ID = "test_id"
    TEST_OOD_GENERATOR = "test_ood_generator"
    TEST_OOD_DOMAIN = "test_ood_domain"
    TEST_ADVERSARIAL = "test_adversarial"
    TEST_HOLDOUT = "test_holdout"   # E2-C1 prompt holdout (additive; not trainable)
    QUARANTINE = "quarantine"   # evaluation-only; must never enter a training path
    EXCLUDED = "excluded"

    @property
    def is_trainable(self) -> bool:
        return self in (Split.TRAIN, Split.VAL)


class CollectionMethod(str, Enum):
    GIT_CLONE = "git_clone"
    HTTP_DOWNLOAD = "http_download"
    HUB_API_SAMPLE = "hub_api_sample"
    OFFICIAL_SITE = "official_site"
    DERIVED = "derived"


class CanonicalRecord(BaseModel):
    """One analysable unit. Text or video."""

    model_config = ConfigDict(extra="forbid", use_enum_values=False)

    # ---- identity & provenance ----
    sample_id: str = Field(min_length=8)
    modality: Modality
    task: Task
    label: int
    original_label: str
    source_dataset: str
    source_version: str
    source_sample_id: str
    source_uri: str | None = None
    content_hash: str = Field(min_length=16)
    group_id: str
    domain: str | None = None
    language: str = "en"
    generator: str | None = None
    collection_method: CollectionMethod
    licence_ref: str
    split: Split = Split.EXCLUDED
    metadata: dict[str, Any] = Field(default_factory=dict)
    transformation_version: str = TRANSFORMATION_VERSION

    # ---- payload (text) ----
    #: Normalised text. Present for text records only. Video records carry
    #: ``source_uri`` to the media instead of embedding bytes.
    text: str | None = None

    # ---- video-only fields (None for text) ----
    media_hash: str | None = None
    width: int | None = None
    height: int | None = None
    duration_s: float | None = None
    fps: float | None = None
    identity_group: str | None = None
    frame_sampling: dict[str, Any] | None = None

    @field_validator("label")
    @classmethod
    def _label_non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError("label must be >= 0")
        return v

    @model_validator(mode="after")
    def _task_label_consistency(self) -> "CanonicalRecord":
        if self.task in (Task.TEXT_BINARY, Task.TEXT_MIXED_BOUNDARY):
            if self.label not in (0, 1, 2):
                raise ValueError(f"{self.task} label must be 0/1/2, got {self.label}")
        if self.task == Task.VIDEO_GENERATED and self.label not in (0, 1):
            raise ValueError("video_generated label must be 0 (real) or 1 (generated)")
        if self.task == Task.VIDEO_MANIPULATED and self.label not in (0, 2):
            raise ValueError("video_manipulated label must be 0 (real) or 2 (manipulated)")
        if self.modality is Modality.TEXT and self.media_hash is not None:
            raise ValueError("text records must not carry media_hash")
        if self.modality is Modality.VIDEO and self.content_hash == "":
            raise ValueError("video records require content_hash")
        return self


def make_sample_id(source_dataset: str, source_version: str, source_sample_id: str) -> str:
    """Stable identity that does not depend on content (content can be re-normalised)."""
    key = f"{source_dataset}|{source_version}|{source_sample_id}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]


def make_content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
