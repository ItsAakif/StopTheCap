"""Pure functions for the E2-C1 prompt inventory.

Everything here is deterministic and side-effect free. The Wikipedia client and
the pipeline live elsewhere.
"""

from __future__ import annotations

import collections
import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Mapping, Sequence

from ml.ingest.normalise import normalise_text, word_count

from .constants import (
    BIO_CATEGORY_RE,
    BLOCKLIST_RE,
    CONTENT_BLOCKLIST_CHARS,
    FAMILY_ID_RE,
    LATIN_RATIO_MIN,
    MAX_PROMPTS_PER_PAGE,
    MAX_WORDS,
    MIN_WORDS,
    PERSON_CATEGORY_RE,
    PROMPT_HASH_RE,
    PROMPT_ID_RE,
    QUOTE_LONG_RE,
    QUOTE_RATIO_MAX,
    REVISION_ID_RE,
    SELECTION_SEED,
    SOURCE_LICENCE,
    TOPIC_RULES,
    TOPIC_TAXONOMY,
)

_BIO_RE = re.compile(BIO_CATEGORY_RE, re.IGNORECASE)
_PERSON_RE = re.compile(PERSON_CATEGORY_RE, re.IGNORECASE)
_BLOCKLIST_RE = re.compile(BLOCKLIST_RE, re.IGNORECASE)
_QUOTE_LONG_RE = re.compile(QUOTE_LONG_RE)
_PROMPT_ID_RE = re.compile(PROMPT_ID_RE)
_PROMPT_HASH_RE = re.compile(PROMPT_HASH_RE)
_REVISION_ID_RE = re.compile(REVISION_ID_RE)
_FAMILY_ID_RE = re.compile(FAMILY_ID_RE)

#: All terminal safety reasons, in rule order (§A.5).
SAFETY_REASONS: tuple[str, ...] = (
    "non_latin",
    "length_short",
    "length_long",
    "disambiguation",
    "list_page",
    "biography",
    "blocklist",
    "quote_heavy",
    "boilerplate",
)

#: The §A.5 terminal status for ``list_page`` is spelled ``excluded_list``.
_STATUS_REASON = {"list_page": "list"}


class InsufficientPoolError(RuntimeError):
    """Raised when the verified candidate pool cannot supply ``n`` prompts.

    ``shortfall`` is the number of additional prompts required; unlike the message,
    it is machine-readable so the audit can record it.
    """

    def __init__(self, message: str, *, shortfall: int = 0) -> None:
        super().__init__(message)
        self.shortfall = shortfall


def latin_ratio(text: str) -> float:
    """Fraction of alphabetic characters below U+0250 after NFKC."""
    if not text:
        return 0.0
    s = unicodedata.normalize("NFKC", text)
    letters = [c for c in s if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if ord(c) < 0x0250) / len(letters)


def script_is_latin(text: str) -> bool:
    return latin_ratio(text) >= LATIN_RATIO_MIN


def topic_category(categories: Sequence[str], title: str) -> tuple[str, str]:
    """Return ``(topic, rule)`` using ordered category-then-title matching."""
    category_blob = " ".join(categories).lower()
    title_blob = (title or "").lower()
    for topic, category_keywords, title_keywords in TOPIC_RULES:
        for keyword in category_keywords:
            if keyword.lower() in category_blob:
                return topic, f"category:{keyword}"
        for keyword in title_keywords:
            if keyword.lower() in title_blob:
                return topic, f"title:{keyword}"
    return "other", "other"


def safety_status(
    *,
    title: str,
    text: str,
    categories: Sequence[str],
    pageprops: Mapping[str, Any],
) -> tuple[str, tuple[str, ...]]:
    """Return the terminal safety status and *all* matched reason keys.

    The first matched reason determines the status (``excluded_<reason>``) but every
    match is retained so ``multi_reason`` audit counts never hide a second rule.
    """
    title = title or ""
    text = text or ""
    categories = tuple(categories or ())
    pageprops = pageprops or {}

    reasons: list[str] = []

    if latin_ratio(text) < LATIN_RATIO_MIN:
        reasons.append("non_latin")

    wc = word_count(text)
    if wc < MIN_WORDS:
        reasons.append("length_short")
    if wc > MAX_WORDS:
        reasons.append("length_long")

    if pageprops.get("disambiguation") is not None or title.strip().lower().endswith(
        "(disambiguation)"
    ):
        reasons.append("disambiguation")

    if title.lower().startswith("list of") or any("Lists" in c for c in categories):
        reasons.append("list_page")

    if any(_BIO_RE.search(c) or _PERSON_RE.search(c) for c in categories):
        reasons.append("biography")

    haystack = " ".join((title, " ".join(categories), text[:CONTENT_BLOCKLIST_CHARS]))
    if _BLOCKLIST_RE.search(haystack):
        reasons.append("blocklist")

    dquotes = text.count('"') + text.count("\u201c") + text.count("\u201d")
    if dquotes / max(1, len(text)) > QUOTE_RATIO_MAX or len(_QUOTE_LONG_RE.findall(text)) >= 2:
        reasons.append("quote_heavy")

    if text.startswith("This article is about") or text.startswith("This page is about"):
        reasons.append("boilerplate")

    if not reasons:
        return "ok", ()
    return f"excluded_{_STATUS_REASON.get(reasons[0], reasons[0])}", tuple(reasons)


def make_source_url(revision_id: str) -> str:
    return f"https://en.wikipedia.org/w/index.php?oldid={revision_id}"


def make_attribution(title: str, revision_id: str, retrieved_at: str, url: str) -> str:
    return (
        f"Wikipedia contributors. {title}. Wikipedia, The Free Encyclopedia. "
        f"Revision {revision_id}, retrieved {retrieved_at}. {url}. Licensed CC BY-SA 4.0."
    )


def make_prompt_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def make_prompt_id(revision_id: str, prompt_hash: str) -> str:
    digest = hashlib.sha256(f"{revision_id}\x1f{prompt_hash}".encode()).hexdigest()
    return "pmt:" + digest[:24]


def _field(candidate: Any, name: str) -> Any:
    if isinstance(candidate, Mapping):
        return candidate[name]
    return getattr(candidate, name)


def _rank_and_select(
    cands: Sequence[Any],
    n: int,
    seed: str,
    max_per_page: int,
) -> tuple[list[Any], int, int]:
    """Rank, apply the per-page cap, and cut off at ``n``.

    Returns ``(chosen, page_cap_removed, rank_cutoff_removed)``. The ranking key is
    order-independent (sha256 of seed + prompt_id, with prompt_id as tie-break).
    """
    cands = list(cands)
    if len(cands) < n:
        shortfall = n - len(cands)
        raise InsufficientPoolError(
            f"{len(cands)} candidates; need {n}; shortfall {shortfall}",
            shortfall=shortfall,
        )

    ranked = sorted(
        cands,
        key=lambda c: (
            hashlib.sha256(f"{seed}\x1f{_field(c, 'prompt_id')}".encode()).hexdigest(),
            _field(c, "prompt_id"),
        ),
    )

    chosen: list[Any] = []
    per_page: collections.Counter[Any] = collections.Counter()
    page_cap_removed = 0
    for candidate in ranked:
        if len(chosen) == n:
            break
        page_id = _field(candidate, "page_id")
        if per_page[page_id] >= max_per_page:
            page_cap_removed += 1
            continue
        chosen.append(candidate)
        per_page[page_id] += 1

    rank_cutoff_removed = len(cands) - len(chosen) - page_cap_removed
    return chosen, page_cap_removed, rank_cutoff_removed


def select_prompts(
    cands: Sequence[Any],
    n: int = 5000,
    seed: str = SELECTION_SEED,
    max_per_page: int = MAX_PROMPTS_PER_PAGE,
) -> list[Any]:
    """Deterministically select ``n`` prompts.

    ``cands`` may contain mappings or objects exposing ``prompt_id`` and ``page_id``.
    Raises :class:`InsufficientPoolError` (with ``shortfall``) when the pool is short;
    it never returns fewer than ``n`` silently.
    """
    chosen, _, _ = _rank_and_select(cands, n, seed, max_per_page)
    return chosen


def validate_manifest(rows: Sequence[Mapping], expected: int = 5000) -> list[str]:
    """Return human-readable violations. Never raises for malformed input."""
    violations: list[str] = []
    rows = list(rows)

    if len(rows) != expected:
        violations.append(f"row count {len(rows)} != expected {expected}")

    seen_ids: dict[str, int] = {}
    seen_hashes: dict[str, int] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            violations.append(f"row {index}: not an object")
            continue

        prompt_id = row.get("prompt_id")
        prompt_hash = row.get("prompt_hash")
        revision_id = row.get("revision_id")
        prompt_text = row.get("prompt_text")
        source_url = row.get("source_url")
        topic = row.get("topic_category")
        status = row.get("content_safety_status")
        script_latin = row.get("script_latin")
        words = row.get("words")
        family_id = row.get("family_id")

        if not isinstance(prompt_id, str) or not _PROMPT_ID_RE.match(prompt_id):
            violations.append(f"row {index}: invalid prompt_id {prompt_id!r}")
        else:
            if prompt_id in seen_ids:
                violations.append(
                    f"row {index}: duplicate prompt_id {prompt_id} (first at row {seen_ids[prompt_id]})"
                )
            else:
                seen_ids[prompt_id] = index

        if not isinstance(prompt_hash, str) or not _PROMPT_HASH_RE.match(prompt_hash):
            violations.append(f"row {index}: invalid prompt_hash {prompt_hash!r}")
        else:
            if prompt_hash in seen_hashes:
                violations.append(
                    f"row {index}: duplicate prompt_hash {prompt_hash} "
                    f"(first at row {seen_hashes[prompt_hash]})"
                )
            else:
                seen_hashes[prompt_hash] = index

        if not isinstance(revision_id, str):
            violations.append(f"row {index}: revision_id is not a string: {revision_id!r}")
        else:
            if not revision_id.isdigit():
                violations.append(f"row {index}: non-numeric revision_id {revision_id!r}")
            if not _REVISION_ID_RE.match(revision_id):
                violations.append(f"row {index}: revision_id does not match {REVISION_ID_RE}: {revision_id!r}")
            expected_url = make_source_url(revision_id)
            if source_url != expected_url or not str(source_url or "").endswith(
                f"oldid={revision_id}"
            ):
                violations.append(
                    f"row {index}: source_url {source_url!r} does not end oldid={revision_id}"
                )

        if topic not in TOPIC_TAXONOMY:
            violations.append(f"row {index}: topic_category {topic!r} not in taxonomy")

        if status != "ok":
            violations.append(f"row {index}: content_safety_status {status!r} != 'ok'")

        if script_latin is not True:
            violations.append(f"row {index}: script_latin is not True: {script_latin!r}")

        if not isinstance(words, int) or isinstance(words, bool):
            violations.append(f"row {index}: words is not an int: {words!r}")
        elif not (MIN_WORDS <= words <= MAX_WORDS):
            violations.append(f"row {index}: words {words} outside [{MIN_WORDS}, {MAX_WORDS}]")

        if not isinstance(prompt_text, str):
            violations.append(f"row {index}: prompt_text is not a string")
        else:
            recomputed = make_prompt_hash(prompt_text)
            if recomputed != prompt_hash:
                violations.append(
                    f"row {index}: prompt_hash {prompt_hash!r} does not recompute "
                    f"(expected {recomputed})"
                )
            elif isinstance(revision_id, str) and isinstance(prompt_hash, str):
                expected_id = make_prompt_id(revision_id, prompt_hash)
                if prompt_id != expected_id:
                    violations.append(
                        f"row {index}: prompt_id {prompt_id!r} does not recompute "
                        f"(expected {expected_id})"
                    )

        if row.get("source_licence") != SOURCE_LICENCE:
            violations.append(
                f"row {index}: source_licence {row.get('source_licence')!r} != {SOURCE_LICENCE!r}"
            )

        if family_id is not None:
            if not isinstance(family_id, str) or not _FAMILY_ID_RE.match(family_id):
                violations.append(f"row {index}: invalid family_id {family_id!r}")

    return violations


def read_manifest(path: Path) -> list[dict]:
    rows: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
    return rows


def write_manifest(path: Path, rows: Sequence[Mapping]) -> str:
    """Write sorted, stable JSONL and return the SHA-256 of the bytes written."""
    ordered = sorted(rows, key=lambda row: row["prompt_id"])
    if ordered:
        payload = "\n".join(
            json.dumps(row, sort_keys=True, ensure_ascii=False) for row in ordered
        ) + "\n"
    else:
        payload = ""
    data = payload.encode("utf-8")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()
