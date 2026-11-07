"""Deterministic text normalisation.

Why this module exists and why it is versioned: research/A1-seed-repo-inspection.md
found that a *single character* (U+2019, the curly apostrophe) separated 20.2% of
AI texts from 1.0% of human texts in the seed corpus -- enough for a 95%-precision
rule with no learning at all. Normalisation removes that shortcut deliberately, and
TRANSFORMATION_VERSION records which ruleset produced a given record so scores stay
comparable over time.

Every rule here is intentional and auditable. Do not add a rule without bumping the
version.
"""

from __future__ import annotations

import re
import unicodedata

# Typographic variants that encode *generator identity* rather than authorship style.
# GPT-family detokenisers emit curly quotes; Wikipedia/raw web text usually does not.
_QUOTE_MAP = str.maketrans(
    {
        "\u2018": "'",  # left single quotation mark
        "\u2019": "'",  # right single quotation mark  <-- the shortcut feature
        "\u201a": "'",
        "\u201b": "'",
        "\u201c": '"',  # left double quotation mark
        "\u201d": '"',  # right double quotation mark
        "\u201e": '"',
        "\u201f": '"',
        "\u2032": "'",  # prime
        "\u2033": '"',  # double prime
        "\u00b4": "'",  # acute accent used as apostrophe
        "\u0060": "'",  # grave accent used as apostrophe
    }
)

_DASH_MAP = str.maketrans(
    {
        "\u2010": "-",  # hyphen
        "\u2011": "-",  # non-breaking hyphen  <-- also a generator fingerprint
        "\u2012": "-",
        "\u2013": "-",  # en dash
        "\u2014": "-",  # em dash
        "\u2015": "-",
        "\u2212": "-",  # minus sign
    }
)

# Zero-width and other invisible characters used in adversarial attacks.
_INVISIBLE = dict.fromkeys(
    map(ord, "\u200b\u200c\u200d\u2060\ufeff\u00ad"), None
)

_WS = re.compile(r"[ \t\f\v]+")
_NEWLINES = re.compile(r"\n{3,}")
_MULTISPACE = re.compile(r" {2,}")

# ---------------------------------------------------------------------------
# Contamination stripping
#
# Found by the character-only audit probe (AUC 0.996 before this filter): several
# Ghostbuster machine variants *echo the instruction* into the text, e.g.
# "Introduction (approximately 500 words):". Prevalence by variant (measured
# 2026-10-29): gpt_writing 86.5%, gpt_prompt2 84.5%, gpt_prompt1 66.4%, gpt 16.5%,
# claude 0.1%, human 0%. Assistant preambles ("Sure! Here's a draft for you...")
# were also present. Both are formatting artifacts of generation, not authorship
# style, and a character n-gram model learns them almost perfectly.
#
# These are stripped rather than used to drop the record: the remaining content is
# still usable, and the removal is recorded as a quality flag so it is auditable.
# ---------------------------------------------------------------------------

_PROMPT_ECHO = re.compile(
    r"^(?:introduction|title|essay|abstract|story|response|answer|draft|"
    r"here\s+is\s+(?:a|the)\s+(?:draft|essay|story))\b[^\n]{0,160}$",
    re.IGNORECASE,
)

_ASSISTANT_PREAMBLE = re.compile(
    r"^(?:sure|certainly|of\s+course|absolutely|okay|ok)[!,. ]|"
    r"^here(?:'s|\s+is)\b|^i(?:'m|\s+am)\s+sorry\b|^as\s+an\s+ai\b|"
    r"^i\s+(?:cannot|can't|am\s+unable)\b|^(?:draft|response)\s*\d*\s*:",
    re.IGNORECASE,
)

_WORD_COUNT_ECHO = re.compile(
    r"\(\s*(?:approximately|approx\.?|about|around)?\s*\d{2,4}\s*words?\s*\)",
    re.IGNORECASE,
)


def strip_contamination(text: str) -> tuple[str, list[str]]:
    """Remove prompt echoes and assistant preambles from the head of a document.

    Returns the cleaned text and a list of flags describing what was removed.
    Only the head is inspected: these artifacts occur at the start of a generation,
    and stripping aggressively in the middle of a document would destroy real content.
    """
    if not text:
        return "", []

    flags: list[str] = []
    lines = text.split("\n")

    # 1. drop leading prompt-echo / preamble lines, up to 3 of them
    dropped = 0
    while lines and dropped < 3:
        head = lines[0].strip()
        if not head:
            lines.pop(0)
            continue
        if _PROMPT_ECHO.match(head):
            lines.pop(0)
            dropped += 1
            if "prompt_echo_removed" not in flags:
                flags.append("prompt_echo_removed")
            continue
        if _ASSISTANT_PREAMBLE.match(head):
            lines.pop(0)
            dropped += 1
            if "assistant_preamble_removed" not in flags:
                flags.append("assistant_preamble_removed")
            continue
        break

    out = "\n".join(lines)

    # 2. remove inline word-count echoes, e.g. "Introduction (approximately 500 words)"
    if _WORD_COUNT_ECHO.search(out[:400]):
        out = _WORD_COUNT_ECHO.sub(" ", out, count=1)
        flags.append("word_count_echo_removed")

    return out.strip(), flags


def normalise_text(text: str) -> str:
    """Apply the full v1 normalisation chain. Order matters."""
    if text is None:
        return ""
    # 1. Unicode NFKC first: composes compatibility characters, so later maps see
    #    canonical code points rather than a mixture.
    s = unicodedata.normalize("NFKC", text)
    # 2. Remove invisible/zero-width characters.
    s = s.translate(_INVISIBLE)
    # 3. Canonicalise quotes and dashes (generator fingerprints).
    s = s.translate(_QUOTE_MAP).translate(_DASH_MAP)
    # 4. Unify line endings, collapse runs of spaces, trim.
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = _WS.sub(" ", s)
    s = _MULTISPACE.sub(" ", s)
    s = _NEWLINES.sub("\n\n", s)
    return s.strip()


def normalise_and_clean(raw: str) -> tuple[str, list[str]]:
    """Normalise, then strip prompt echoes / assistant preambles.

    Stripping happens *after* Unicode + quote canonicalisation so that the regexes
    match on a single, predictable representation.
    """
    s = normalise_text(raw)
    cleaned, flags = strip_contamination(s)
    return normalise_text(cleaned), flags


def word_count(text: str) -> int:
    return len([w for w in text.split() if w])


def shingles(text: str, k: int = 5) -> set[str]:
    """Word k-shingles over a whitespace-tokenised, lowercased text."""
    words = [w.lower() for w in re.findall(r"\w+", text)]
    if len(words) < k:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i : i + k]) for i in range(len(words) - k + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / (len(a) + len(b) - inter)
