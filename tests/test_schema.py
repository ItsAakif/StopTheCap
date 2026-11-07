"""Schema invariants: the rules that must never be silently violated."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ml.ingest.schema import (
    CanonicalRecord,
    CollectionMethod,
    Modality,
    Split,
    Task,
    make_content_hash,
    make_sample_id,
)


def _base(**over):
    kw = dict(
        sample_id="a" * 32,
        modality=Modality.TEXT,
        task=Task.TEXT_BINARY,
        label=0,
        original_label="human",
        source_dataset="ghostbuster",
        source_version="v1",
        source_sample_id="x",
        content_hash=make_content_hash("hello"),
        group_id="g",
        collection_method=CollectionMethod.GIT_CLONE,
        licence_ref="CC-BY-3.0",
        split=Split.EXCLUDED,
    )
    kw.update(over)
    return CanonicalRecord(**kw)


def test_sample_id_is_stable_and_content_independent():
    a = make_sample_id("ghostbuster", "v1", "essay/human/1.txt")
    b = make_sample_id("ghostbuster", "v1", "essay/human/1.txt")
    c = make_sample_id("ghostbuster", "v2", "essay/human/1.txt")
    assert a == b
    assert a != c  # version change must produce a new id


def test_content_hash_detects_change():
    assert make_content_hash("a") != make_content_hash("b")


def test_text_record_rejects_media_hash():
    with pytest.raises(ValidationError):
        _base(media_hash="deadbeef")


def test_video_generated_labels_are_restricted():
    with pytest.raises(ValidationError):
        _base(modality=Modality.VIDEO, task=Task.VIDEO_GENERATED, label=2)


def test_video_manipulated_accepts_only_0_or_2():
    r = _base(modality=Modality.VIDEO, task=Task.VIDEO_MANIPULATED, label=2,
              content_hash="manifest:deadbeefdeadbeef")
    assert r.label == 2
    with pytest.raises(ValidationError):
        _base(modality=Modality.VIDEO, task=Task.VIDEO_MANIPULATED, label=1,
              content_hash="manifest:deadbeefdeadbeef")


def test_negative_label_rejected():
    with pytest.raises(ValidationError):
        _base(label=-1)


def test_trainable_split_property():
    assert Split.TRAIN.is_trainable
    assert Split.VAL.is_trainable
    assert not Split.QUARANTINE.is_trainable
    assert not Split.TEST_ID.is_trainable
    assert not Split.EXCLUDED.is_trainable


def test_extra_fields_are_forbidden():
    with pytest.raises(ValidationError):
        _base(unexpected_field=1)
