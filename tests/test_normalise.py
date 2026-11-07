"""Normalisation and contamination stripping.

These tests encode the numeric findings in research/A1-seed-repo-inspection.md sA.9
(curly apostrophe => 95.2% precision rule) and the prompt-echo contamination found
during milestone 1. They exist so those artefacts cannot silently return.
"""

from __future__ import annotations

from ml.ingest.normalise import (
    jaccard,
    normalise_and_clean,
    normalise_text,
    shingles,
    strip_contamination,
    word_count,
)


def test_curly_apostrophe_is_canonicalised():
    # The single character that separated 20.2% of AI from 1.0% of human text.
    assert normalise_text("it\u2019s") == "it's"


def test_typographic_quotes_and_dashes_canonicalised():
    assert normalise_text("\u201cquoted\u201d") == '"quoted"'
    assert normalise_text("a\u2014b") == "a-b"
    assert normalise_text("a\u2013b") == "a-b"


def test_non_breaking_hyphen_canonicalised():
    assert normalise_text("co\u2011operate") == "co-operate"


def test_zero_width_and_invisible_removed():
    assert normalise_text("a\u200bb\ufeffc\u00ad") == "abc"


def test_nfkc_composes_compatibility_characters():
    assert normalise_text("\uff21\uff22") == "AB"  # fullwidth A B


def test_whitespace_collapsed():
    assert normalise_text("a   b\t\tc\n\n\n\nd") == "a b c\n\nd"


def test_idempotent():
    once = normalise_text("\u201cHello\u201d  \u2014 world\u2019s")
    assert normalise_text(once) == once


def test_prompt_echo_header_stripped():
    text = "Introduction (approximately 500 words)\nSocial justice matters because..."
    cleaned, flags = strip_contamination(text)
    assert "prompt_echo_removed" in flags
    assert not cleaned.startswith("Introduction")
    assert "Social justice matters" in cleaned


def test_assistant_preamble_stripped():
    text = "Sure! Here's a draft for you.\nOnce upon a time there was a forest."
    cleaned, flags = strip_contamination(text)
    assert "assistant_preamble_removed" in flags
    assert cleaned.startswith("Once upon a time")


def test_word_count_echo_removed():
    text = "The policy (approximately 500 words) was introduced in 1990."
    cleaned, flags = strip_contamination(text)
    assert "word_count_echo_removed" in flags
    assert "500 words" not in cleaned


def test_clean_text_unchanged():
    text = "A perfectly ordinary sentence with no artifacts."
    cleaned, flags = strip_contamination(text)
    assert cleaned == text
    assert flags == []


def test_normalise_and_clean_returns_flags():
    cleaned, flags = normalise_and_clean("Title: My Essay\nBody text follows here.")
    assert "prompt_echo_removed" in flags
    assert cleaned.startswith("Body text")


def test_shingles_and_jaccard():
    assert jaccard(shingles("a b c d e"), shingles("a b c d e")) == 1.0
    assert jaccard(shingles("a b c d e"), shingles("v w x y z")) == 0.0
    assert 0.0 < jaccard(shingles("a b c d e f"), shingles("a b c d e g")) < 1.0


def test_word_count():
    assert word_count("one two three") == 3
    assert word_count("") == 0
