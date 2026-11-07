"""The additive ``Split.TEST_HOLDOUT`` enum value (§C.5)."""

from __future__ import annotations

from ml.ingest.schema import (
    CanonicalRecord,
    CollectionMethod,
    Modality,
    Split,
    Task,
    make_content_hash,
)

#: The exact Split members that existed before E2-C1 T8. This list is the pin: it
#: must never change as a side effect of adding a value.
PRE_EXISTING_VALUES = {
    "train",
    "val",
    "test_id",
    "test_ood_generator",
    "test_ood_domain",
    "test_adversarial",
    "quarantine",
    "excluded",
}

PRE_EXISTING_NAMES = {
    "TRAIN",
    "VAL",
    "TEST_ID",
    "TEST_OOD_GENERATOR",
    "TEST_OOD_DOMAIN",
    "TEST_ADVERSARIAL",
    "QUARANTINE",
    "EXCLUDED",
}


def _record(**over) -> CanonicalRecord:
    kwargs = dict(
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
    kwargs.update(over)
    return CanonicalRecord(**kwargs)


def test_new_value_parses_from_a_string():
    assert Split("test_holdout") is Split.TEST_HOLDOUT
    assert Split.TEST_HOLDOUT.value == "test_holdout"


def test_new_value_is_not_trainable():
    assert Split.TEST_HOLDOUT.is_trainable is False


def test_pre_existing_values_still_round_trip():
    for value in PRE_EXISTING_VALUES:
        assert Split(value).value == value
        assert Split(value).is_trainable is (value in {"train", "val"})


def test_pre_existing_value_and_name_sets_are_unchanged():
    assert {member.value for member in Split} == PRE_EXISTING_VALUES | {"test_holdout"}
    assert {member.name for member in Split} == PRE_EXISTING_NAMES | {"TEST_HOLDOUT"}
    assert {
        member.value for member in Split if member is not Split.TEST_HOLDOUT
    } == PRE_EXISTING_VALUES
    assert {
        member.name for member in Split if member is not Split.TEST_HOLDOUT
    } == PRE_EXISTING_NAMES


def test_trainability_is_unchanged_for_every_member():
    trainable = {member.value for member in Split if member.is_trainable}
    assert trainable == {"train", "val"}


def test_canonical_record_accepts_the_new_split_from_a_string():
    record = _record(split="test_holdout")
    assert record.split is Split.TEST_HOLDOUT
    assert record.split.is_trainable is False
