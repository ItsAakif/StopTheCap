"""Cross-dataset diagnosis: the metric plumbing, and the no-tuning rule."""

from __future__ import annotations

import numpy as np
import pytest

from ml.baselines.diagnose import DomainResult, diagnose
from ml.ingest.schema import Split
from tests.conftest import text_record


def _corpus(n=120):
    """Two separable classes plus a held-out domain, all in one licensed corpus."""
    recs = []
    for i in range(n):
        split = (Split.TRAIN if i < 50 else
                 Split.VAL if i < 80 else Split.TEST_ID)
        recs.append(text_record(f"h{i}", f"human essay {i} about rivers " * 6, 0,
                                group_id=f"g{i}", split=split, domain="student_essay"))
        recs.append(text_record(f"m{i}", f"machine essay {i} about mountains " * 6, 1,
                                group_id=f"g{i}", split=split, generator="gpt",
                                domain="student_essay"))
    # an out-of-domain block in the same corpus
    for i in range(40):
        split = Split.TRAIN if i < 20 else Split.TEST_OOD_DOMAIN
        recs.append(text_record(f"oh{i}", f"human reuters {i} market report " * 6, 0,
                                group_id=f"od{i}", split=split, domain="news_reuters"))
        recs.append(text_record(f"om{i}", f"machine reuters {i} market report " * 6, 1,
                                group_id=f"od{i}", split=split, generator="gpt",
                                domain="news_reuters"))
    return recs


def test_diagnose_returns_threshold_and_domains():
    d = diagnose(_corpus())
    assert 0.0 <= d.threshold <= 1.0
    names = {r.name for r in d.results}
    assert any("test_id" in n for n in names)
    assert any("ood_domain" in n for n in names)


def test_oov_rate_is_bounded():
    d = diagnose(_corpus())
    for r in d.results:
        assert 0.0 <= r.oov_rate <= 1.0


def test_diagnose_notes_domain_drop():
    d = diagnose(_corpus())
    assert d.notes, "expected at least one interpretation note"


def test_domain_result_serialises():
    r = DomainResult("x", 10, 5, 5, 0.5, 0.1, 0.5)
    assert r.__dict__["name"] == "x"


def test_diagnose_requires_train_and_val():
    recs = [text_record(f"h{i}", "text " * 10, 0, group_id=f"g{i}",
                        split=Split.TEST_ID) for i in range(5)]
    with pytest.raises(RuntimeError):
        diagnose(recs)
