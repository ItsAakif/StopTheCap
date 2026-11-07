"""Build E2-C1 prompts directly from MediaWiki wikitext revisions.

Why this path exists
--------------------
``prop=extracts`` returns roughly four or five intros per request, so reaching
5,000 prompts takes thousands of round trips.  ``prop=revisions&rvprop=content``
returns up to 50 full articles per request, carries the revision id in the same
response, and embeds categories in the wikitext.  That cuts the request count by
an order of magnitude.

The module reuses :mod:`ml.e2c1.api_prompts` for transport (pacing, caching,
429 handling) and for deterministic selection/manifest writing.  Only the
transport payload and the wikitext parser differ.
"""

from __future__ import annotations

import collections
import json
import re
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from ml.ingest.normalise import normalise_text, word_count

from .api_prompts import (
    DEFAULT_DEADLINE_SECONDS,
    DEFAULT_MAX_REQUESTS,
    DEFAULT_MIN_INTERVAL_S,
    DEFAULT_REQUEST_TIMEOUT_S,
    DEFAULT_WORKERS,
    ApiClient,
    ApiError,
    DeadlineReached,
    MaxRequestsReached,
    _TitleQueue,
    classify_text,
    map_requested_titles,
    select_candidates,
)
from .constants import (
    PIPELINE_VERSION,
    SAFETY_RULESET_VERSION,
    SELECTION_SEED,
    SOURCE_LICENCE,
)
from .families import build_families
from .prompts import (
    make_attribution,
    make_prompt_hash,
    make_prompt_id,
    make_source_url,
    topic_category,
    write_manifest,
)

#: 50 is the API maximum for ``titles=`` and keeps enumeration/fetch aligned.
REVISION_BATCH_SIZE = 50
AP_LIMIT = 50

DEFAULT_PER_PAGE = 5
DEFAULT_MAX_PER_PAGE = 3
DEFAULT_SECTIONS = "all"

DEFAULT_OUT = "datasets/manifests/e2_c1_prompts_wikitext.jsonl"
DEFAULT_AUDIT = "datasets/manifests/e2_c1_prompt_audit_wikitext.json"

#: A partial manifest on deadline is only written when it is still useful.
MIN_PARTIAL_MANIFEST = 5000

#: Same deterministic seed order as the extracts path.
DEFAULT_SEEDS: tuple[str, ...] = (
    "0",
    *tuple(chr(ord("A") + index) for index in range(26)),
    "Ba",
    "Ca",
    "Da",
    "Fa",
    "Ha",
    "Ka",
    "Ma",
    "Na",
    "Pa",
    "Sa",
    "Ta",
    "Wa",
)

#: The API contract caps wikitext-derived prompts at 300 words
#: (``classify_text`` enforces this through ``API_MAX_WORDS``).

Fetch = Callable[[str, Mapping[str, str]], dict]


# ---------------------------------------------------------------------------
# Request construction
# ---------------------------------------------------------------------------
def build_enumeration_params(
    seed: str, *, apcontinue: str | None = None
) -> dict[str, str]:
    """Return the ``list=allpages`` title-enumeration parameters."""
    params = {
        "action": "query",
        "format": "json",
        "formatversion": "2",
        "list": "allpages",
        "apnamespace": "0",
        "aplimit": str(AP_LIMIT),
        "apfilterredir": "nonredirects",
        "maxlag": "5",
    }
    if apcontinue is not None:
        params["apcontinue"] = str(apcontinue)
    else:
        params["apfrom"] = str(seed)
    return params


def build_revisions_params(titles: Sequence[str]) -> dict[str, str]:
    """Return one ``prop=revisions`` snapshot request for up to 50 titles."""
    return {
        "action": "query",
        "format": "json",
        "formatversion": "2",
        "redirects": "1",
        "titles": "|".join(str(title) for title in titles),
        "prop": "revisions",
        "rvprop": "ids|timestamp|content",
        "rvslots": "main",
        "maxlag": "5",
    }


# ---------------------------------------------------------------------------
# Wikitext parsing (pure)
# ---------------------------------------------------------------------------
_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_REF_RE = re.compile(
    r"<ref\b[^>]*/\s*>|<ref\b[^>]*>.*?</ref\s*>", re.IGNORECASE | re.DOTALL
)
_TABLE_RE = re.compile(
    r"^[ \t]*\{\|.*?^[ \t]*\|\}[ \t]*$", re.DOTALL | re.MULTILINE
)
_FILE_LINK_RE = re.compile(
    r"\[\[\s*(?:File|Image)\s*:.*?\]\]", re.IGNORECASE | re.DOTALL
)
_CATEGORY_LINK_RE = re.compile(r"\[\[\s*Category\s*:[^\]]*\]\]", re.IGNORECASE)
_CATEGORY_CAPTURE_RE = re.compile(
    r"\[\[\s*Category\s*:\s*([^\]|]+)(?:\|[^\]]*)?\]\]", re.IGNORECASE
)
_TEMPLATE_RE = re.compile(r"\{\{[^{}]*\}\}", re.DOTALL)
_WIKILINK_LABEL_RE = re.compile(r"\[\[([^\]|]*)\|([^\]]*)\]\]")
_WIKILINK_PLAIN_RE = re.compile(r"\[\[([^\]]*)\]\]")
_EXTLINK_RE = re.compile(r"\[https?://[^\s\]]+(?:\s+([^\]]*))?\]", re.IGNORECASE)
_HEADING_RE = re.compile(r"^==+[^=]", re.MULTILINE)
_HEADING_LINE_RE = re.compile(r"^==+[^=].*==+$", re.MULTILINE)
_BLANK_LINE = re.compile(r"\r?\n\s*\r?\n")
_RESIDUAL_RE = re.compile(r"\[\[|\]\]|\{\{|\}\}|<ref|~{3,}|\|", re.IGNORECASE)
_SENTENCE_END_RE = re.compile(r"[.!?]")
_LIST_PREFIXES = ("*", "#", ";", ":")

#: Valid values for the ``sections`` parser mode.
SECTION_MODES = ("lead", "all")


def _capture_categories(wikitext: str) -> list[str]:
    categories: list[str] = []
    for match in _CATEGORY_CAPTURE_RE.finditer(wikitext):
        name = match.group(1).strip()
        if not name:
            continue
        category = "Category:" + name
        if category not in categories:
            categories.append(category)
    return categories


def _strip_templates(text: str) -> str:
    """Remove templates innermost-first until no ``{{...}}`` remains."""
    previous = None
    while previous != text:
        previous = text
        text = _TEMPLATE_RE.sub("", text)
    return text


def _strip_initial(text: str) -> str:
    """Steps 1-3: comments, refs, then tables."""
    text = _COMMENT_RE.sub("", text)
    text = _REF_RE.sub("", text)
    text = _TABLE_RE.sub("", text)
    return text


def _strip_containers(text: str) -> str:
    """Steps 4-7: containers, templates, wikilinks, emphasis, external links."""
    text = _FILE_LINK_RE.sub("", text)
    text = _CATEGORY_LINK_RE.sub("", text)
    text = _strip_templates(text)
    text = _WIKILINK_LABEL_RE.sub(lambda match: match.group(2), text)
    text = _WIKILINK_PLAIN_RE.sub(lambda match: match.group(1), text)
    text = text.replace("'''", "").replace("''", "")
    text = _EXTLINK_RE.sub(lambda match: (match.group(1) or ""), text)
    return text


def _strip_markup(text: str) -> str:
    """Run the full markup pipeline up to (excluding) the lead cutoff."""
    return _strip_containers(_strip_initial(text))


def _lead_section(text: str) -> str:
    match = _HEADING_RE.search(text)
    if match is None:
        return text
    return text[: match.start()]


def _all_sections(text: str) -> str:
    """Drop heading lines but keep every section's body text."""
    return _HEADING_LINE_RE.sub("", text)


def _is_list_like(paragraph: str) -> bool:
    """Return ``True`` for list items and fragments without sentence ending."""
    stripped = paragraph.lstrip()
    if stripped.startswith(_LIST_PREFIXES):
        return True
    return _SENTENCE_END_RE.search(paragraph) is None


def _split_paragraphs(text: str) -> list[str]:
    return [part.strip() for part in _BLANK_LINE.split(text) if part.strip()]


def _parse_wikitext(
    wikitext: str, sections: str = "all"
) -> tuple[list[str], list[str], int]:
    """Return ``(categories, paragraphs, parser_rejects)`` for one page.

    ``sections`` selects the parser mode: ``"lead"`` stops before the first
    heading (the original behaviour) and ``"all"`` parses the whole article,
    dropping heading lines from the output.

    ``parser_rejects`` counts non-empty paragraphs that still contain residual
    MediaWiki markup after the ordered strip pipeline, or that are list-like.
    They are never emitted.
    """
    if sections not in SECTION_MODES:
        raise ValueError(
            f"sections must be one of {SECTION_MODES!r}, got {sections!r}"
        )
    categories = _capture_categories(_strip_initial(wikitext))
    text = _strip_markup(wikitext)
    text = _lead_section(text) if sections == "lead" else _all_sections(text)

    paragraphs: list[str] = []
    rejects = 0
    for paragraph in _split_paragraphs(text):
        if _RESIDUAL_RE.search(paragraph) or _is_list_like(paragraph):
            rejects += 1
            continue
        paragraphs.append(paragraph)
    return categories, paragraphs, rejects


def _revision_content(revision: Mapping[str, Any]) -> str:
    slots = revision.get("slots")
    if isinstance(slots, Mapping):
        main = slots.get("main")
        if isinstance(main, Mapping):
            for key in ("content", "*"):
                value = main.get(key)
                if isinstance(value, str) and value:
                    return value
    for key in ("content", "*"):
        value = revision.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


def _page_content(page: Mapping[str, Any]) -> str:
    for key in ("wikitext", "content"):
        value = page.get(key)
        if isinstance(value, str) and value:
            return value
    revisions = page.get("revisions")
    if revisions:
        first = revisions[0]
        if isinstance(first, Mapping):
            return _revision_content(first)
    return ""


def _revision_field(page: Mapping[str, Any], name: str) -> str:
    revisions = page.get("revisions")
    if revisions:
        first = revisions[0]
        if isinstance(first, Mapping) and first.get(name) is not None:
            return str(first[name])
    alias = {"revid": "revision_id", "timestamp": "revision_timestamp"}.get(
        name, name
    )
    value = page.get(alias)
    return str(value) if value is not None else ""


def _is_disambiguation(title: str, wikitext: str, categories: Sequence[str]) -> bool:
    if title.strip().lower().endswith("(disambiguation)"):
        return True
    if "{{disambiguation" in wikitext.lower():
        return True
    return any("disambiguation" in category.lower() for category in categories)


def _parse_page_details(
    page: Mapping[str, Any], sections: str = "all"
) -> tuple[dict[str, Any], int]:
    title = str(page.get("title") or "")
    wikitext = _page_content(page)
    categories, paragraphs, rejects = _parse_wikitext(wikitext, sections=sections)
    revision_id = _revision_field(page, "revid")
    revision_timestamp = _revision_field(page, "timestamp")
    result = {
        "page_id": page.get("pageid"),
        "title": title,
        "revision_id": revision_id,
        "revision_timestamp": revision_timestamp,
        "url": make_source_url(revision_id),
        "categories": categories,
        "paragraphs": paragraphs,
        "is_disambiguation": _is_disambiguation(title, wikitext, categories),
    }
    return result, rejects


def parse_page(page: Mapping[str, Any], sections: str = "all") -> dict[str, Any]:
    """Parse one API page into the canonical wikitext page record.

    Accepts either a raw ``prop=revisions`` page (with ``revisions``/``slots``)
    or an already-finalised page carrying ``wikitext``.  ``sections`` is
    ``"all"`` (default) or ``"lead"``.
    """
    parsed, _rejects = _parse_page_details(page, sections=sections)
    return parsed


# ---------------------------------------------------------------------------
# Response merging
# ---------------------------------------------------------------------------
def _merge_revision_page(
    store: dict[Any, dict[str, Any]],
    page: Mapping[str, Any],
    requested_title: str | None = None,
) -> None:
    key = page.get("pageid")
    if key is None:
        key = page.get("title")
    if key is None:
        return

    slot = store.setdefault(key, {})
    for name in ("pageid", "title", "ns", "missing", "invalid"):
        if name in page:
            slot[name] = page[name]
    if requested_title and "requested_title" not in slot:
        slot["requested_title"] = requested_title

    revisions = page.get("revisions")
    if revisions and "_revision" not in slot:
        first = revisions[0]
        if isinstance(first, Mapping):
            slot["_revision"] = dict(first)


def _finalise_revision_pages(
    store: Mapping[Any, Mapping[str, Any]]
) -> list[dict[str, Any]]:
    pages: list[dict[str, Any]] = []
    for slot in store.values():
        page = dict(slot)
        revision = page.pop("_revision", None) or {}
        page["revision_id"] = str(revision.get("revid", ""))
        page["revision_timestamp"] = str(revision.get("timestamp", ""))
        page["wikitext"] = _revision_content(revision)
        pages.append(page)
    return pages


def _fetch_revision_batch(
    client: ApiClient, titles: Sequence[str]
) -> list[dict[str, Any]]:
    """Fetch one revision batch and follow any continuation tokens."""
    requested = [str(title) for title in titles]
    base = build_revisions_params(requested)
    params = dict(base)
    store: dict[Any, dict[str, Any]] = {}
    seen_continuations: set[tuple[tuple[str, str], ...]] = set()

    while True:
        data = client.request(params)
        query = data.get("query") or {}
        reverse = {
            target: source
            for source, target in map_requested_titles(requested, query).items()
        }
        for page in query.get("pages") or []:
            if not isinstance(page, Mapping):
                continue
            title = page.get("title")
            requested_title = reverse.get(title) if isinstance(title, str) else None
            _merge_revision_page(store, page, requested_title)

        continuation = data.get("continue") or {}
        if continuation:
            signature = tuple(
                sorted((str(key), str(value)) for key, value in continuation.items())
            )
            if signature in seen_continuations:
                raise ApiError("continuation did not advance")
            seen_continuations.add(signature)
            params = dict(base)
            params.update(
                {str(key): str(value) for key, value in continuation.items()}
            )
            continue

        return _finalise_revision_pages(store)


def _enumerate_titles_page(
    client: ApiClient, seed: str, apcontinue: str | None
) -> tuple[list[str], str | None]:
    """Fetch one ``list=allpages`` page of titles and its continuation token."""
    data = client.request(build_enumeration_params(seed, apcontinue=apcontinue))
    query = data.get("query") or {}
    titles: list[str] = []
    for entry in query.get("allpages") or []:
        if isinstance(entry, Mapping):
            title = entry.get("title")
            if isinstance(title, str) and title:
                titles.append(title)
    continuation = data.get("continue") or {}
    next_continue = continuation.get("apcontinue")
    return titles, (str(next_continue) if next_continue else None)


# ---------------------------------------------------------------------------
# Build state
# ---------------------------------------------------------------------------
class _BuildState:
    def __init__(self, n: int, per_page: int, max_per_page: int) -> None:
        self.n = int(n)
        self.per_page = max(1, int(per_page))
        self.max_per_page = max(1, int(max_per_page))
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.seen_pages: set[Any] = set()
        self.candidates: dict[str, dict[str, Any]] = {}
        self.page_counts: collections.Counter[Any] = collections.Counter()
        self.capped_total = 0
        self.pages_fetched = 0
        self.pages_seen = 0
        self.no_extract = 0
        self.no_revision = 0
        self.parser_rejects = 0
        self.length = 0
        self.non_latin = 0
        self.duplicates = 0
        self.safety: collections.Counter[str] = collections.Counter()
        self.warnings: list[str] = []
        self.deadline_hit = False
        self.max_requests_hit = False
        self.last_progress_at: float | None = None

    def add_warning(self, message: str) -> None:
        with self.lock:
            self.warnings.append(message)

    def add_page(self, page: Mapping[str, Any], parser_rejects: int = 0) -> None:
        page_id_key = page.get("page_id")
        if page_id_key is None:
            page_id_key = page.get("title")
        if page_id_key is None:
            return

        with self.lock:
            if page_id_key in self.seen_pages:
                return
            self.seen_pages.add(page_id_key)
            self.pages_fetched += 1

            rejects = max(0, int(parser_rejects))
            self.parser_rejects += rejects
            # A rejected paragraph is a removal unit, so it must also enter the
            # funnel for reconciliation to balance.
            self.pages_seen += rejects

            title = str(page.get("title") or "")
            categories = tuple(page.get("categories") or ())
            pageprops: dict[str, Any] = (
                {"disambiguation": True}
                if page.get("is_disambiguation")
                else {}
            )
            revision_id = str(page.get("revision_id") or "")
            paragraphs = list(page.get("paragraphs") or ())

            if not paragraphs:
                self.pages_seen += 1
                self.no_extract += 1
                return
            if not revision_id:
                self.pages_seen += 1
                self.no_revision += 1
                return

            page_id = page.get("page_id")
            try:
                page_id = int(page_id)
            except (TypeError, ValueError):
                page_id = -1

            accepted = 0
            for paragraph in paragraphs:
                if accepted >= self.per_page:
                    break
                self.pages_seen += 1
                text = normalise_text(paragraph)
                status = classify_text(
                    text,
                    title=title,
                    categories=categories,
                    pageprops=pageprops,
                )
                if status == "length":
                    self.length += 1
                    continue
                if status == "non_latin":
                    self.non_latin += 1
                    continue
                if status.startswith("safety:"):
                    self.safety[status.split(":", 1)[1]] += 1
                    continue

                prompt_hash = make_prompt_hash(text)
                prompt_id = make_prompt_id(revision_id, prompt_hash)
                if prompt_id in self.candidates:
                    self.duplicates += 1
                    continue

                accepted += 1

                topic, rule = topic_category(categories, title)
                self.candidates[prompt_id] = {
                    "prompt_id": prompt_id,
                    "prompt_hash": prompt_hash,
                    "prompt_text": text,
                    "words": word_count(text),
                    "page_id": page_id,
                    "source_title": title,
                    "revision_id": revision_id,
                    "revision_timestamp": str(page.get("revision_timestamp") or ""),
                    "topic_category": topic,
                    "topic_rule": rule,
                }
                self.page_counts[page_id] += 1
                if self.page_counts[page_id] <= self.max_per_page:
                    self.capped_total += 1

    def snapshot(self) -> tuple[list[dict[str, Any]], int, int]:
        with self.lock:
            return list(self.candidates.values()), self.capped_total, self.pages_seen

    def usable_and_fetched(self) -> tuple[int, int]:
        with self.lock:
            return self.capped_total, self.pages_fetched

    def maybe_progress(self, client: ApiClient, interval_s: float) -> None:
        if interval_s <= 0:
            return
        now = client.now()
        with self.lock:
            last = self.last_progress_at
            if last is not None and now - last < interval_s:
                return
            self.last_progress_at = now
            pages = self.pages_fetched
            usable = self.capped_total
        stats = client.stats()

        print(
            f"pages={pages} usable={usable} requests={stats['requests']} "
            f"rate429={stats['rate429']} elapsed={int(now)}s",
            file=sys.stderr,
            flush=True,
        )


# ---------------------------------------------------------------------------
# Worker loop
# ---------------------------------------------------------------------------
def _check_limits(
    client: ApiClient,
    start: float,
    deadline_seconds: float | None,
    max_requests: int | None,
) -> None:
    if deadline_seconds is not None and client.now() - start >= deadline_seconds:
        raise DeadlineReached
    if max_requests is not None and client.stats()["requests"] >= max_requests:
        raise MaxRequestsReached


def _worker(
    worker_index: int,
    seeds: Sequence[str],
    *,
    client: ApiClient,
    state: _BuildState,
    title_queue: _TitleQueue,
    start: float,
    deadline_seconds: float | None,
    max_requests: int | None,
    progress_interval_s: float,
    sections: str,
) -> None:
    """Enumerate titles and fetch 50-title revision batches cooperatively."""
    seed_index = 0
    seed: str | None = seeds[0] if seeds else None
    apcontinue: str | None = None

    def advance_seed() -> None:
        nonlocal seed_index, seed, apcontinue
        seed_index += 1
        seed = seeds[seed_index] if seed_index < len(seeds) else None
        apcontinue = None

    try:
        while True:
            if state.stop.is_set() or worker_index >= client.effective_workers:
                return
            _check_limits(client, start, deadline_seconds, max_requests)

            if title_queue.pending() < REVISION_BATCH_SIZE:
                if seed is not None and title_queue.should_enumerate(state):
                    try:
                        titles, next_continue = _enumerate_titles_page(
                            client, seed, apcontinue
                        )
                    except (DeadlineReached, MaxRequestsReached):
                        raise
                    except Exception as exc:  # noqa: BLE001 - one bad seed
                        state.add_warning(
                            f"seed {seed!r}: {type(exc).__name__}: {exc}"
                        )
                        advance_seed()
                        continue
                    title_queue.add(titles)
                    apcontinue = next_continue
                    if not apcontinue:
                        advance_seed()
                    continue
                if title_queue.pending() == 0:
                    return

            batch = title_queue.take(REVISION_BATCH_SIZE)
            if not batch:
                return
            try:
                pages = _fetch_revision_batch(client, batch)
            except (DeadlineReached, MaxRequestsReached):
                raise
            except Exception as exc:  # noqa: BLE001 - one bad batch
                state.add_warning(
                    f"titles {batch[0]!r}+: {type(exc).__name__}: {exc}"
                )
                continue

            for page in pages:
                parsed, rejects = _parse_page_details(page, sections=sections)
                state.add_page(parsed, rejects)
                if state.capped_total >= state.n:
                    state.stop.set()
                    return
            state.maybe_progress(client, progress_interval_s)
    except DeadlineReached:
        state.deadline_hit = True
        state.stop.set()
        return
    except MaxRequestsReached:
        state.max_requests_hit = True
        state.stop.set()
        return


# ---------------------------------------------------------------------------
# Audit / output helpers
# ---------------------------------------------------------------------------
def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _write_json(path: Path, obj: Mapping[str, Any]) -> None:
    payload = (
        json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _stage(
    stages: list[dict[str, int | str]], name: str, in_count: int, removed: int
) -> int:
    removed = max(0, min(int(removed), int(in_count)))
    out = int(in_count) - removed
    stages.append(
        {"stage": name, "in": int(in_count), "removed": removed, "out": out}
    )
    return out


def _build_stages(
    state: _BuildState,
    *,
    capped_total: int,
    selected: int,
) -> list[dict[str, int | str]]:
    stages: list[dict[str, int | str]] = [
        {
            "stage": "pages_seen",
            "in": state.pages_seen,
            "removed": 0,
            "out": state.pages_seen,
        }
    ]
    remaining = state.pages_seen
    remaining = _stage(stages, "parser_reject", remaining, state.parser_rejects)
    remaining = _stage(stages, "no_extract", remaining, state.no_extract)
    if state.no_revision:
        remaining = _stage(stages, "no_revision", remaining, state.no_revision)
    remaining = _stage(stages, "length", remaining, state.length)
    remaining = _stage(stages, "non_latin", remaining, state.non_latin)
    for status in sorted(state.safety):
        remaining = _stage(
            stages, f"safety:{status}", remaining, state.safety[status]
        )
    if state.duplicates:
        remaining = _stage(stages, "duplicate", remaining, state.duplicates)
    remaining = _stage(stages, "per_page_cap", remaining, remaining - capped_total)
    _stage(stages, "rank_cutoff", remaining, remaining - selected)
    return stages


def _manifest_row(candidate: Mapping[str, Any], retrieved_at: str) -> dict[str, Any]:
    revision_id = str(candidate["revision_id"])
    url = make_source_url(revision_id)
    return {
        "prompt_id": candidate["prompt_id"],
        "source_title": candidate["source_title"],
        "source_url": url,
        "revision_id": revision_id,
        "retrieved_at": retrieved_at,
        "source_licence": SOURCE_LICENCE,
        "attribution": make_attribution(
            str(candidate["source_title"]), revision_id, retrieved_at, url
        ),
        "prompt_text": candidate["prompt_text"],
        "prompt_hash": candidate["prompt_hash"],
        "topic_category": candidate["topic_category"],
        "content_safety_status": "ok",
        "page_id": candidate["page_id"],
        "words": candidate["words"],
        "script_latin": True,
        "topic_rule": candidate["topic_rule"],
    }


def _make_client(
    *,
    fetch: Fetch | None,
    cache_dir: Path | str | None,
    min_interval_s: float,
    refresh: bool,
    request_timeout_s: float,
    workers: int,
    sleep: Callable[[float], None],
    clock: Callable[[], float],
) -> ApiClient:
    return ApiClient(
        fetch=fetch,
        cache_dir=cache_dir,
        min_interval_s=min_interval_s,
        refresh=refresh,
        request_timeout_s=request_timeout_s,
        workers=workers,
        sleep=sleep,
        clock=clock,
    )


# ---------------------------------------------------------------------------
# Public build
# ---------------------------------------------------------------------------
def run_wikitext_build(
    *,
    out_path: Path | str = DEFAULT_OUT,
    audit_path: Path | str = DEFAULT_AUDIT,
    n: int = 5000,
    per_page: int = DEFAULT_PER_PAGE,
    max_per_page: int = DEFAULT_MAX_PER_PAGE,
    sections: str = DEFAULT_SECTIONS,
    workers: int = DEFAULT_WORKERS,
    min_interval_s: float = DEFAULT_MIN_INTERVAL_S,
    max_requests: int | None = DEFAULT_MAX_REQUESTS,
    deadline_seconds: float | None = DEFAULT_DEADLINE_SECONDS,
    refresh: bool = False,
    cache_dir: Path | str | None = None,
    seeds: Sequence[str] | None = None,
    retrieved_at: str | None = None,
    client: ApiClient | None = None,
    fetch: Fetch | None = None,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
    request_timeout_s: float = DEFAULT_REQUEST_TIMEOUT_S,
    progress_interval_s: float = 5.0,
) -> tuple[dict[str, Any], int]:
    """Enumerate pages, fetch wikitext revisions, and write the audit."""
    out_path = Path(out_path)
    audit_path = Path(audit_path)
    seed_list = list(seeds) if seeds is not None else list(DEFAULT_SEEDS)
    retrieved_at = retrieved_at or _today()

    if client is None:
        client = _make_client(
            fetch=fetch,
            cache_dir=cache_dir,
            min_interval_s=min_interval_s,
            refresh=refresh,
            request_timeout_s=request_timeout_s,
            workers=workers,
            sleep=sleep,
            clock=clock,
        )
    else:
        client.effective_workers = max(1, int(workers))

    state = _BuildState(n=n, per_page=per_page, max_per_page=max_per_page)
    start = client.now()
    actual_workers = max(1, int(workers))
    buckets = [seed_list[index::actual_workers] for index in range(actual_workers)]

    common = {
        "client": client,
        "state": state,
        "title_queue": _TitleQueue(),
        "start": start,
        "deadline_seconds": deadline_seconds,
        "max_requests": max_requests,
        "progress_interval_s": progress_interval_s,
        "sections": sections,
    }

    if actual_workers == 1:
        _worker(0, buckets[0], **common)
    else:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=actual_workers) as pool:
            futures = [
                pool.submit(_worker, index, bucket, **common)
                for index, bucket in enumerate(buckets)
                if bucket
            ]
            for future in futures:
                future.result()

    candidates, capped_total, pages_seen = state.snapshot()
    chosen, selection = select_candidates(
        candidates, n, seed=SELECTION_SEED, max_per_page=max_per_page
    )
    selected = len(chosen)
    enough = capped_total >= int(n)

    if state.deadline_hit:
        status = "deadline_reached"
    elif not enough:
        status = "insufficient_pool"
    else:
        status = "ok"

    manifest_rows = [_manifest_row(candidate, retrieved_at) for candidate in chosen]
    if manifest_rows:
        families = build_families(manifest_rows)
        for row in manifest_rows:
            row["family_id"] = families[row["prompt_id"]]

    write_output = status == "ok" or (
        status == "deadline_reached" and selected >= MIN_PARTIAL_MANIFEST
    )

    if write_output:
        output_sha = write_manifest(out_path, manifest_rows)
        output = {
            "path": str(out_path),
            "sha256": output_sha,
            "count": len(manifest_rows),
        }
        exit_code = 0
    else:
        output = {"path": str(out_path), "sha256": None, "count": 0}
        exit_code = 3

    stages = _build_stages(
        state,
        capped_total=selection["capped_total"],
        selected=selected,
    )
    removed = sum(int(stage["removed"]) for stage in stages)
    stats = client.stats()
    elapsed = max(0.0, client.now() - start)

    if status == "insufficient_pool":
        state.warnings.append(
            f"usable pool {selected} < requested {n}; manifest not written"
        )
    if status == "deadline_reached" and not write_output:
        state.warnings.append(
            f"deadline reached with {selected} prompts; manifest not written "
            f"(needs >= {MIN_PARTIAL_MANIFEST})"
        )

    audit: dict[str, Any] = {
        "schema_version": "1.0",
        "pipeline_version": PIPELINE_VERSION,
        "safety_ruleset_version": SAFETY_RULESET_VERSION,
        "status": status,
        "mode": "wikitext",
        "generated_at": _now_iso(),
        "input": {
            "source": "wikipedia-api",
            "seeds": seed_list,
            "retrieved_at": retrieved_at,
        },
        "retrieved_at": retrieved_at,
        "selection": {
            "rank_seed": SELECTION_SEED,
            "n_requested": int(n),
            "candidates": len(candidates),
            "selected": selected,
            "page_cap_removed": selection["page_cap_removed"],
            "rank_cutoff_removed": selection["rank_cutoff_removed"],
        },
        "pages_seen": pages_seen,
        "pages_fetched": state.pages_fetched,
        "paragraphs_seen": (
            pages_seen
            - state.parser_rejects
            - state.no_extract
            - state.no_revision
        ),
        "parser_rejects": state.parser_rejects,
        "stages": stages,
        "exclusions": {
            "parser_reject": state.parser_rejects,
            "no_extract": state.no_extract,
            "no_revision": state.no_revision,
            "length": state.length,
            "non_latin": state.non_latin,
            **{f"safety:{key}": value for key, value in sorted(state.safety.items())},
            "duplicate": state.duplicates,
        },
        "requests": stats["requests"],
        "pages_per_request": (
            round(state.pages_fetched / stats["requests"], 3)
            if stats["requests"]
            else 0.0
        ),
        "cache_hits": stats["cache_hits"],
        "retries": stats["retries"],
        "rate429": stats["rate429"],
        "maxlag": stats["maxlag"],
        "elapsed_s": round(elapsed, 3),
        "reconciliation": {
            "pages_seen": pages_seen,
            "removed": removed,
            "selected": selected,
            "balanced": pages_seen - removed == selected,
        },
        "output": output,
        "warnings": state.warnings,
    }

    if not audit["reconciliation"]["balanced"] and exit_code == 0:
        audit["warnings"].append(
            "reconciliation failed: pages_seen - removed != selected"
        )
        exit_code = 4
    if state.max_requests_hit:
        audit["warnings"].append(
            f"max requests ({max_requests}) reached before {n} usable prompts"
        )

    _write_json(audit_path, audit)
    return audit, exit_code
