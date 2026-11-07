"""Shared test fixtures and record builders."""

from __future__ import annotations

import pytest

from ml.ingest.schema import CanonicalRecord, CollectionMethod, Modality, Split, Task


def text_record(
    sample_id: str,
    text: str,
    label: int,
    *,
    group_id: str | None = None,
    dataset: str = "ghostbuster",
    split: Split = Split.EXCLUDED,
    generator: str | None = None,
    domain: str = "student_essay",
    word_count: int | None = None,
    eval_only: bool = False,
) -> CanonicalRecord:
    import hashlib

    # schema requires >= 8 chars; pad short ids deterministically
    sample_id = f"{sample_id}".ljust(12, "x")
    meta = {
        "word_count": word_count if word_count is not None else len(text.split()),
        "quality_flags": [],
        "_eval_only": eval_only,
    }
    return CanonicalRecord(
        sample_id=sample_id,
        modality=Modality.TEXT,
        task=Task.TEXT_BINARY,
        label=label,
        original_label="human" if label == 0 else (generator or "machine"),
        source_dataset=dataset,
        source_version="test",
        source_sample_id=sample_id,
        content_hash=hashlib.sha256(text.encode()).hexdigest(),
        group_id=group_id or sample_id,
        domain=domain,
        generator=generator,
        collection_method=CollectionMethod.GIT_CLONE,
        licence_ref="CC-BY-3.0::https://example.invalid/LICENSE",
        split=split,
        metadata=meta,
        text=text,
    )


def video_record(
    sample_id: str,
    label: int,
    *,
    group_id: str | None = None,
    identity_group: str | None = None,
    split: Split = Split.EXCLUDED,
) -> CanonicalRecord:
    sample_id = f"{sample_id}".ljust(12, "v")
    return CanonicalRecord(
        sample_id=sample_id,
        modality=Modality.VIDEO,
        task=Task.VIDEO_GENERATED,
        label=label,
        original_label="real" if label == 0 else "generated",
        source_dataset="genvidbench",
        source_version="test",
        source_sample_id=sample_id,
        content_hash=f"manifest:{sample_id}",
        group_id=group_id or sample_id,
        domain="genvidbench_pair1",
        generator=None if label == 0 else "sora",
        collection_method=CollectionMethod.HTTP_DOWNLOAD,
        licence_ref="CC-BY-NC-4.0::https://example.invalid",
        split=split,
        metadata={"media_present": False, "word_count": 0, "_eval_only": False},
        media_hash=f"manifest:{sample_id}",
        identity_group=identity_group,
        frame_sampling={"n_frames": None},
    )


@pytest.fixture
def clean_corpus() -> list[CanonicalRecord]:
    """Small, genuinely non-artifactual synthetic corpus."""
    recs = []
    for i in range(40):
        recs.append(text_record(f"h{i}", f"human sentence number {i} about ordinary things", 0))
        recs.append(text_record(
            f"m{i}", f"a different generated phrase number {i} concerning assorted matters", 1,
            generator="gpt"))
    return recs
