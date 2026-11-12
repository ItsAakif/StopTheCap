"""Offline tests for the exported model-card template.

``ml.train.export_checkpoint`` imports torch/transformers lazily inside
``export``/``verify``, so importing the module and rendering the card must work
without either dependency installed.
"""

from __future__ import annotations

import importlib

from ml.train.export_checkpoint import FROZEN_THRESHOLD, render_model_card


def test_card_uses_the_essay_corpus_builder() -> None:
    card = render_model_card("distilbert-base-cased", 2)

    assert "python -m ml.e2c1.build_essay_corpus" in card
    assert "build_training_corpus" not in card


def test_card_keeps_threshold_results_and_citation() -> None:
    card = render_model_card("distilbert-base-cased", 2)

    assert "0.0007257819990627467" in card
    assert repr(FROZEN_THRESHOLD) == "0.0007257819990627467"
    assert "| HC3 | 0.447 | 88.3% |" in card
    assert "| RAID | 0.377 | 88.2% |" in card
    assert "## Citation" in card


def test_card_has_markdown_structure() -> None:
    card = render_model_card("distilbert-base-cased", 2)

    assert card.startswith("# ")
    assert "## Reproduce" in card


def test_module_imports_without_torch() -> None:
    module = importlib.import_module("ml.train.export_checkpoint")

    assert callable(module.render_model_card)
