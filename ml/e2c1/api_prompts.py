"""Build E2-C1 prompts from live Wikipedia responses.

The build derives each prompt paragraph from the *same* response that carries
the revision id, so ``revision_mismatch`` is structurally impossible.

Design notes
------------
* The default ``titles`` mode enumerates page titles with ``list=allpages``
  (cheap, no extracts) and then fetches snapshots in batches of 20 with
  ``titles=`` and ``prop=extracts``.  A probe showed a 20-title batch returns
  all 20 intro extracts in one request, while ``generator=allpages`` chunks
  extracts and yields only a couple of pages per request.
* ``redirects=1`` is allowed with ``titles=`` because the titles are explicit.
  It is forbidden with ``generator=allpages``; the legacy ``generator`` mode is
  still available behind ``--mode generator`` and :func:`build_params` keeps it
  free of ``redirects``.
* MediaWiki chunks extracts and categories.  Within a snapshot batch the
  response is followed through ``excontinue``/``clcontinue`` until the batch is
  complete; the pages are then merged and derived.
* One unit in the audit funnel is a *page/paragraph candidate*: every paragraph
  considered adds one unit, and a page with no extract adds one unit (removed by
  ``no_extract``).  This makes ``pages_seen - sum(removed) == selected`` hold
  even though a page can contribute up to ``MAX_PROMPTS_PER_PAGE`` prompts.
"""

from __future__ import annotations

import collections
import hashlib
import json
import math
import re
import socket
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from ml.ingest.normalise import normalise_text, word_count

from . import prompts
from .constants import (
    API_URL,
    CACHE_DIR,
    LATIN_RATIO_MIN,
    MAX_PROMPTS_PER_PAGE,
    MIN_WORDS,
    PIPELINE_VERSION,
    RETRY_BACKOFF_SECONDS,
    SAFETY_RULESET_VERSION,
    SELECTION_SEED,
    SOURCE_LICENCE,
    USER_AGENT,
)
from .families import build_families
from .prompts import (
    latin_ratio,
    make_attribution,
    make_prompt_hash,
    make_prompt_id,
    make_source_url,
    safety_status,
    topic_category,
    write_manifest,
)

#: This pipeline deliberately uses a tighter upper bound than the generic
#: ``MAX_WORDS`` (400).  The contract for the API path accepts 40..300 words.
API_MAX_WORDS = 300

#: A partial manifest is only written on deadline when it is still useful.
MIN_PARTIAL_MANIFEST = 5000

#: Deterministic seed order.  ``0`` then ``A``..``Z``, then a handful of
#: mid-alphabet restart points.  The list is fixed, so two runs start identically.
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

DEFAULT_MODE = "titles"
DEFAULT_PER_PAGE = 5
TITLE_BATCH_SIZE = 20
AP_LIMIT = 50
ENUMERATION_LOOKAHEAD = 200

DEFAULT_OUT = "datasets/manifests/e2_c1_prompts.jsonl"
DEFAULT_AUDIT = "datasets/manifests/e2_c1_prompt_audit.json"
DEFAULT_DEADLINE_SECONDS = 2400.0
DEFAULT_MAX_REQUESTS = 2000
DEFAULT_WORKERS = 3
DEFAULT_MIN_INTERVAL_S = 1.0
DEFAULT_REQUEST_TIMEOUT_S = 30.0
DEFAULT_RETRY_AFTER_S = 7.0
MAX_429_RETRIES = 3

Fetch = Callable[[str, Mapping[str, str]], dict]


class ApiError(RuntimeError):
    """A request failed after retries or returned a non-retryable API error."""

    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


class DeadlineReached(RuntimeError):
    """The global deadline elapsed; the build stops gracefully."""


class MaxRequestsReached(RuntimeError):
    """The configured request budget was exhausted."""


# ---------------------------------------------------------------------------
# Request construction
# ---------------------------------------------------------------------------
def build_params(seed: str, *, gapfrom: str | None = None) -> dict[str, str]:
    """Return the verified ``generator=allpages`` request parameters.

    ``redirects=1`` is intentionally absent: ``generator=allpages`` rejects it
    with an API ``params`` error.  A test pins this invariant.
    """
    return {
        "action": "query",
        "format": "json",
        "formatversion": "2",
        "generator": "allpages",
        "gapnamespace": "0",
        "gapfrom": str(gapfrom if gapfrom is not None else seed),
        "gaplimit": "20",
        "gapfilterredir": "nonredirects",
        "prop": "extracts|categories|revisions|info|pageprops",
        "exintro": "1",
        "explaintext": "1",
        "exlimit": "max",
        "rvprop": "ids|timestamp",
        "cllimit": "max",
        "clshow": "!hidden",
        "inprop": "url",
        # maxlag triggers the documented backoff path.  It is not one of the
        # parameters measured in the probe, but it is harmless and standard.
        "maxlag": "5",
    }


def build_allpages_params(
    seed: str, *, apcontinue: str | None = None
) -> dict[str, str]:
    """Return the cheap title-enumeration request parameters.

    The enumeration phase uses ``list=allpages`` and never requests extracts.
    ``redirects`` is deliberately absent: it is not needed here and some
    ``generator``/``list`` combinations reject it.
    """
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


def build_titles_params(titles: Sequence[str]) -> dict[str, str]:
    """Return snapshot parameters for one batch of up to 20 titles.

    ``redirects=1`` is valid with explicit ``titles=`` (it is only forbidden
    with ``generator=allpages``), so redirects resolve in the same request.
    """
    return {
        "action": "query",
        "format": "json",
        "formatversion": "2",
        "redirects": "1",
        "titles": "|".join(str(title) for title in titles),
        "prop": "extracts|categories|revisions|info|pageprops",
        "exintro": "1",
        "explaintext": "1",
        "exlimit": str(TITLE_BATCH_SIZE),
        "rvprop": "ids|timestamp",
        "cllimit": "max",
        "clshow": "!hidden",
        "inprop": "url",
        "maxlag": "5",
    }


def cache_key(params: Mapping[str, Any]) -> str:
    """Canonical cache key: ``sha256(canonical_params)[:16]``."""
    canonical = json.dumps(
        dict(params), sort_keys=True, ensure_ascii=False, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Paragraph derivation (pure)
# ---------------------------------------------------------------------------
_BLANK_LINE = re.compile(r"\r?\n\s*\r?\n")


def split_paragraphs(extract: str, limit: int | None = None) -> list[str]:
    """Split plaintext on blank lines and return at most ``limit`` paragraphs."""
    if not extract:
        return []
    paragraphs = [part.strip() for part in _BLANK_LINE.split(extract)]
    paragraphs = [part for part in paragraphs if part]
    if limit is not None:
        paragraphs = paragraphs[: max(0, int(limit))]
    return paragraphs


def classify_text(
    text: str,
    *,
    title: str,
    categories: Sequence[str] = (),
    pageprops: Mapping[str, Any] | None = None,
) -> str:
    """Classify an already-normalised paragraph.

    Returns ``"ok"``, ``"length"``, ``"non_latin"`` or ``"safety:<status>"``.
    Length is checked before script and safety so the audit's stage funnel is
    deterministic and matches the contract's 40..300 word window.
    """
    words = word_count(text)
    if words < MIN_WORDS or words > API_MAX_WORDS:
        return "length"
    if latin_ratio(text) < LATIN_RATIO_MIN:
        return "non_latin"
    status, _reasons = safety_status(
        title=title,
        text=text,
        categories=categories,
        pageprops=pageprops or {},
    )
    if status != "ok":
        return f"safety:{status}"
    return "ok"


def classify_paragraph(
    paragraph: str,
    *,
    title: str,
    categories: Sequence[str] = (),
    pageprops: Mapping[str, Any] | None = None,
) -> str:
    """Normalise then classify one paragraph (test-friendly wrapper)."""
    return classify_text(
        normalise_text(paragraph),
        title=title,
        categories=categories,
        pageprops=pageprops,
    )


# ---------------------------------------------------------------------------
# Transport
# ---------------------------------------------------------------------------
def _default_fetch(
    url: str, headers: Mapping[str, str], *, timeout: float = DEFAULT_REQUEST_TIMEOUT_S
) -> dict:
    request = urllib.request.Request(url, headers=dict(headers))
    with urllib.request.urlopen(  # noqa: S310 - fixed API host
        request, timeout=timeout
    ) as response:
        raw = response.read()
    return json.loads(raw.decode("utf-8"))


def _status_code(exc: BaseException) -> int | None:
    for attribute in ("code", "status", "status_code"):
        value = getattr(exc, attribute, None)
        if isinstance(value, int):
            return value
    return None


def _is_timeout(exc: BaseException) -> bool:
    if isinstance(exc, (socket.timeout, TimeoutError)):
        return True
    if isinstance(exc, urllib.error.URLError):
        reason = exc.reason
        if isinstance(reason, (socket.timeout, TimeoutError)):
            return True
        text = str(reason).lower()
        return "timed out" in text or "timeout" in text
    return False


def _retry_after(exc: BaseException, default: float = DEFAULT_RETRY_AFTER_S) -> float:
    """Read ``Retry-After`` from an HTTP exception, falling back to ``default``."""
    explicit = getattr(exc, "retry_after", None)
    if explicit is not None:
        try:
            return max(0.0, float(explicit))
        except (TypeError, ValueError):
            pass

    headers = getattr(exc, "headers", None)
    if headers is not None:
        value = None
        try:
            value = headers.get("Retry-After") or headers.get("retry-after")
        except AttributeError:
            value = None
        try:
            return max(0.0, float(value))
        except (TypeError, ValueError):
            pass
    return float(default)


class ApiClient:
    """Paced, cached, retrying MediaWiki client.

    ``fetch``, ``sleep`` and ``clock`` are injectable so tests never touch the
    network or wall-clock delays.  Every worker gets its own minimum-interval
    slot, while 429 pauses are global (shared by all workers).
    """

    def __init__(
        self,
        fetch: Fetch | None = None,
        *,
        cache_dir: Path | str | None = None,
        user_agent: str = USER_AGENT,
        min_interval_s: float = DEFAULT_MIN_INTERVAL_S,
        refresh: bool = False,
        request_timeout_s: float = DEFAULT_REQUEST_TIMEOUT_S,
        workers: int = 1,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir is not None else Path(CACHE_DIR)
        self.user_agent = user_agent
        self.min_interval_s = float(min_interval_s)
        self.refresh = bool(refresh)
        self.request_timeout_s = float(request_timeout_s)
        self._fetch = (
            fetch
            if fetch is not None
            else lambda url, headers: _default_fetch(
                url, headers, timeout=self.request_timeout_s
            )
        )
        self._sleep = sleep
        self._clock = clock
        self._local = threading.local()
        self._stats_lock = threading.Lock()
        self._pause_lock = threading.Lock()
        self._concurrency_lock = threading.Lock()
        self._pause_until = 0.0
        self._effective_workers = max(1, int(workers))
        self._stats: dict[str, int] = {
            "requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "retries": 0,
            "rate429": 0,
            "maxlag": 0,
            "timeouts": 0,
        }

    # ------------------------------------------------------------- counters
    def stats(self) -> dict[str, int]:
        with self._stats_lock:
            return dict(self._stats)

    def _bump(self, key: str, amount: int = 1) -> None:
        with self._stats_lock:
            self._stats[key] = self._stats.get(key, 0) + amount

    @property
    def effective_workers(self) -> int:
        with self._concurrency_lock:
            return self._effective_workers

    @effective_workers.setter
    def effective_workers(self, value: int) -> None:
        with self._concurrency_lock:
            self._effective_workers = max(1, int(value))

    def _reduce_concurrency(self) -> None:
        """Halve effective concurrency; persistent 429s should not amplify load."""
        with self._concurrency_lock:
            self._effective_workers = max(1, self._effective_workers // 2)

    def now(self) -> float:
        return self._clock()

    # --------------------------------------------------------------- pacing
    def pause(self, seconds: float) -> None:
        """Pause *all* workers for ``seconds`` from now."""
        if seconds <= 0:
            return
        with self._pause_lock:
            self._pause_until = max(self._pause_until, self._clock() + seconds)

    def _wait_for_slot(self) -> None:
        now = self._clock()
        with self._pause_lock:
            paused_for = max(0.0, self._pause_until - now)
        last = getattr(self._local, "last_request_at", None)
        interval_for = 0.0
        if last is not None and self.min_interval_s > 0:
            interval_for = max(0.0, self.min_interval_s - (now - last))
        wait = max(paused_for, interval_for)
        if wait > 0:
            self._sleep(wait)
        # Measuring after the sleep also lets tests inject ``sleeps.append``
        # without a clock; a real sleep advances the clock for the next call.
        self._local.last_request_at = self._clock()

    # ---------------------------------------------------------------- cache
    def _cache_path(self, params: Mapping[str, Any]) -> Path:
        return self.cache_dir / f"{cache_key(params)}.json"

    def _load_cache(self, params: Mapping[str, Any]) -> dict | None:
        path = self._cache_path(params)
        if not path.exists():
            return None
        try:
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError):
            return None
        return data if isinstance(data, dict) else None

    def _store_cache(self, params: Mapping[str, Any], data: dict) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path = self._cache_path(params)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, sort_keys=True)
            handle.write("\n")

    # ------------------------------------------------------------ transport
    def request(self, params: Mapping[str, str]) -> dict:
        params = {str(key): str(value) for key, value in params.items()}

        if not self.refresh:
            cached = self._load_cache(params)
            if cached is not None:
                self._bump("cache_hits")
                return cached

        self._bump("cache_misses")
        url = f"{API_URL}?{urllib.parse.urlencode(params)}"
        headers = {"User-Agent": self.user_agent, "Accept": "application/json"}

        rate_retries = 0
        backoff_retries = 0
        while True:
            self._wait_for_slot()
            self._bump("requests")
            try:
                data = self._fetch(url, headers)
            except Exception as exc:  # noqa: BLE001 - classified below
                status = _status_code(exc)
                if status == 429:
                    self._bump("rate429")
                    delay = _retry_after(exc)
                    self.pause(delay)
                    if rate_retries >= MAX_429_RETRIES:
                        # 429s persisted through all retries: back off globally.
                        self._reduce_concurrency()
                        raise ApiError(
                            f"HTTP 429 persisted after {rate_retries} retries",
                            status=429,
                        ) from exc
                    rate_retries += 1
                    self._bump("retries")
                    continue

                retryable = _is_timeout(exc) or (
                    status is not None and 500 <= status < 600
                )
                if _is_timeout(exc):
                    self._bump("timeouts")
                if retryable and backoff_retries < len(RETRY_BACKOFF_SECONDS):
                    self._sleep(RETRY_BACKOFF_SECONDS[backoff_retries])
                    backoff_retries += 1
                    self._bump("retries")
                    continue
                raise ApiError(
                    f"HTTP failure: {exc}", status=status
                ) from exc

            if not isinstance(data, dict):
                raise ApiError("non-object JSON response")

            error = data.get("error")
            if error:
                code = str(error.get("code", ""))
                if code == "maxlag":
                    self._bump("maxlag")
                    if backoff_retries < len(RETRY_BACKOFF_SECONDS):
                        self._sleep(RETRY_BACKOFF_SECONDS[backoff_retries])
                        backoff_retries += 1
                        self._bump("retries")
                        continue
                raise ApiError(f"API error {code}: {error.get('info', '')}")

            self._store_cache(params, data)
            return data


# ---------------------------------------------------------------------------
# Batch fetching / page merging
# ---------------------------------------------------------------------------
def _merge_page(store: dict[Any, dict[str, Any]], page: Mapping[str, Any]) -> None:
    key = page.get("pageid")
    if key is None:
        key = page.get("title")
    if key is None:
        return

    slot = store.setdefault(key, {"pageid": page.get("pageid")})
    if page.get("title"):
        slot["title"] = str(page["title"])
    requested_title = page.get("requested_title")
    if isinstance(requested_title, str) and requested_title:
        slot.setdefault("requested_title", requested_title)
    for key_name in ("pageid", "missing", "invalid", "ns"):
        if key_name in page:
            slot[key_name] = page[key_name]

    if isinstance(page.get("pageprops"), Mapping):
        merged = dict(slot.get("pageprops") or {})
        merged.update(page["pageprops"])
        slot["pageprops"] = merged

    revisions = page.get("revisions")
    if revisions and "_revision" not in slot:
        slot["_revision"] = revisions[0]

    extract = page.get("extract")
    if isinstance(extract, str) and extract:
        chunks = slot.setdefault("_extracts", [])
        if extract not in chunks:
            chunks.append(extract)

    for category in page.get("categories") or []:
        name = category.get("title") if isinstance(category, Mapping) else None
        if isinstance(name, str) and name:
            names = slot.setdefault("_category_names", [])
            if name not in names:
                names.append(name)


def _finalise_pages(store: Mapping[Any, Mapping[str, Any]]) -> list[dict[str, Any]]:
    pages: list[dict[str, Any]] = []
    for slot in store.values():
        page = dict(slot)
        page["extract"] = "\n".join(page.pop("_extracts", []))
        page["categories"] = tuple(page.pop("_category_names", []))
        revision = page.pop("_revision", None) or {}
        page["revision_id"] = str(revision.get("revid", "")) if revision else ""
        page["revision_timestamp"] = (
            str(revision.get("timestamp", "")) if revision else ""
        )
        pages.append(page)
    return pages


def _fetch_batch(
    client: ApiClient, seed: str, gapfrom: str
) -> tuple[list[dict[str, Any]], str | None]:
    """Fetch one ``gapfrom`` batch, following property continuations.

    Returns ``(pages, next_gapfrom)``.  ``next_gapfrom`` is ``None`` when the
    enumeration for this seed is complete.
    """
    base = build_params(seed, gapfrom=gapfrom)
    params = dict(base)
    store: dict[Any, dict[str, Any]] = {}
    seen_continuations: set[tuple[tuple[str, str], ...]] = set()

    while True:
        data = client.request(params)
        query = data.get("query") or {}
        for page in query.get("pages") or []:
            if isinstance(page, Mapping):
                _merge_page(store, page)

        continuation = data.get("continue") or {}
        # Follow every continuation except the batch boundary (gapcontinue).
        # The generic ``continue`` token is metadata, not a module.
        module_keys = {
            str(key) for key in continuation if key not in ("continue", "gapcontinue")
        }
        if module_keys:
            signature = tuple(
                sorted((str(k), str(v)) for k, v in continuation.items())
            )
            if signature in seen_continuations:
                raise ApiError("continuation did not advance")
            seen_continuations.add(signature)
            params = dict(base)
            params.update(
                {
                    str(key): str(value)
                    for key, value in continuation.items()
                    if key != "gapcontinue"
                }
            )
            continue

        next_gap = continuation.get("gapcontinue")
        return _finalise_pages(store), (str(next_gap) if next_gap else None)


def _enumerate_titles_page(
    client: ApiClient, seed: str, apcontinue: str | None
) -> tuple[list[str], str | None]:
    """Fetch one ``list=allpages`` page of titles and its continuation token."""
    params = build_allpages_params(seed, apcontinue=apcontinue)
    data = client.request(params)
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


def map_requested_titles(
    requested: Sequence[str], query: Mapping[str, Any]
) -> dict[str, str]:
    """Map each requested title to the final title returned by the API.

    ``query.normalized`` fixes casing/underscores and ``query.redirects``
    follows redirects; both chains are applied in that order.
    """
    normalized = {
        str(entry["from"]): str(entry["to"])
        for entry in query.get("normalized") or []
        if isinstance(entry, Mapping) and entry.get("from") and entry.get("to")
    }
    redirects = {
        str(entry["from"]): str(entry["to"])
        for entry in query.get("redirects") or []
        if isinstance(entry, Mapping) and entry.get("from") and entry.get("to")
    }

    resolved: dict[str, str] = {}
    for title in requested:
        original = str(title)
        current = normalized.get(original, original)
        seen = {current}
        while current in redirects:
            current = redirects[current]
            if current in seen:
                break
            seen.add(current)
        resolved[original] = current
    return resolved


def _fetch_title_batch(
    client: ApiClient, titles: Sequence[str]
) -> list[dict[str, Any]]:
    """Fetch one ``titles=`` batch, following property continuations.

    ``excontinue`` and ``clcontinue`` (and any other module continuation) are
    followed within the batch.  ``query.normalized``/``query.redirects`` are
    mapped back to the requested titles so redirect targets stay attributable.
    """
    requested = [str(title) for title in titles]
    base = build_titles_params(requested)
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
            merged = dict(page)
            title = merged.get("title")
            if isinstance(title, str) and title in reverse:
                merged["requested_title"] = reverse[title]
            _merge_page(store, merged)

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

        return _finalise_pages(store)


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

    def add_page(self, page: Mapping[str, Any]) -> None:
        page_id_key = page.get("pageid")
        if page_id_key is None:
            page_id_key = page.get("title")
        if page_id_key is None:
            return

        with self.lock:
            if page_id_key in self.seen_pages:
                return
            self.seen_pages.add(page_id_key)
            self.pages_fetched += 1

            title = str(page.get("title") or "")
            categories = tuple(page.get("categories") or ())
            pageprops = dict(page.get("pageprops") or {})
            revision_id = str(page.get("revision_id") or "")
            paragraphs = split_paragraphs(str(page.get("extract") or ""), self.per_page)

            if not paragraphs:
                self.pages_seen += 1
                self.no_extract += 1
                return
            if not revision_id:
                self.pages_seen += 1
                self.no_revision += 1
                return

            page_id = page.get("pageid")
            try:
                page_id = int(page_id)
            except (TypeError, ValueError):
                page_id = -1

            for paragraph in paragraphs:
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
        """Return ``(capped_total, pages_fetched)`` for runtime yield tracking."""
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


class _TitleQueue:
    """Thread-safe, deduplicated title queue shared by title-mode workers."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pending: collections.deque[str] = collections.deque()
        self._seen: set[str] = set()

    def add(self, titles: Sequence[str]) -> int:
        """Add unseen titles and return how many were newly queued."""
        added = 0
        with self._lock:
            for title in titles:
                text = str(title)
                if not text or text in self._seen:
                    continue
                self._seen.add(text)
                self._pending.append(text)
                added += 1
        return added

    def take(self, limit: int) -> list[str]:
        with self._lock:
            count = min(max(0, int(limit)), len(self._pending))
            return [self._pending.popleft() for _ in range(count)]

    def pending(self) -> int:
        with self._lock:
            return len(self._pending)

    def should_enumerate(self, state: _BuildState) -> bool:
        """Whether more titles are needed to plausibly reach ``state.n``.

        ``needed`` is the number of additional titles implied by the observed
        yield (usable prompts per fetched page).  A floor of
        :data:`ENUMERATION_LOOKAHEAD` titles keeps a buffer for pages that yield
        nothing, and prevents the enumeration phase from being too short.
        """
        usable, fetched = state.usable_and_fetched()
        if usable >= state.n:
            return False
        observed_yield = usable / fetched if fetched > 0 else 1.0
        if observed_yield <= 0:
            return True
        needed = math.ceil(max(0, state.n - usable) / observed_yield)
        return self.pending() < needed + ENUMERATION_LOOKAHEAD


# ---------------------------------------------------------------------------
# Deterministic selection
# ---------------------------------------------------------------------------
def select_candidates(
    candidates: Sequence[Mapping[str, Any]],
    n: int,
    *,
    seed: str = SELECTION_SEED,
    max_per_page: int = MAX_PROMPTS_PER_PAGE,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Hash-rank, apply the per-page cap, and return up to ``n`` prompts.

    ``n`` is clamped to the number of prompts that survive the page cap, so a
    short pool never raises here; the caller records the shortfall instead.
    """
    candidates = [dict(candidate) for candidate in candidates]
    page_counts = collections.Counter(candidate["page_id"] for candidate in candidates)
    capped_total = sum(min(count, max_per_page) for count in page_counts.values())
    effective_n = min(int(n), capped_total)
    if effective_n <= 0:
        return [], {
            "candidates": len(candidates),
            "capped_total": capped_total,
            "page_cap_removed": len(candidates) - capped_total,
            "rank_cutoff_removed": capped_total,
        }

    chosen, _page_cap, _rank_cutoff = prompts._rank_and_select(
        candidates, effective_n, seed, max_per_page
    )
    return chosen, {
        "candidates": len(candidates),
        "capped_total": capped_total,
        "page_cap_removed": len(candidates) - capped_total,
        "rank_cutoff_removed": capped_total - effective_n,
    }


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
    start: float,
    deadline_seconds: float | None,
    max_requests: int | None,
    progress_interval_s: float,
) -> None:
    for seed in seeds:
        if state.stop.is_set() or worker_index >= client.effective_workers:
            return
        gapfrom = seed
        try:
            while True:
                if state.stop.is_set() or worker_index >= client.effective_workers:
                    return
                _check_limits(client, start, deadline_seconds, max_requests)

                pages, next_gap = _fetch_batch(client, seed, gapfrom)
                for page in pages:
                    state.add_page(page)
                    if state.capped_total >= state.n:
                        state.stop.set()
                        return
                state.maybe_progress(client, progress_interval_s)

                if not next_gap:
                    break
                gapfrom = next_gap
        except DeadlineReached:
            state.deadline_hit = True
            state.stop.set()
            return
        except MaxRequestsReached:
            state.max_requests_hit = True
            state.stop.set()
            return
        except Exception as exc:  # noqa: BLE001 - one bad seed must not kill the build
            state.add_warning(f"seed {seed!r}: {type(exc).__name__}: {exc}")
            continue


def _titles_worker(
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
) -> None:
    """Enumerate titles and fetch 20-title snapshot batches cooperatively.

    Fetching takes precedence over enumeration, so the queue stays small and
    the worker stops adding titles as soon as the observed yield shows the
    target is reachable.  Any queued titles are still drained before exit.
    """
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

            # Top up to a full batch so only the final batch is partial.  Do
            # not enumerate further once the observed yield says n is
            # reachable.
            if title_queue.pending() < TITLE_BATCH_SIZE:
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

            batch = title_queue.take(TITLE_BATCH_SIZE)
            if not batch:
                return
            try:
                pages = _fetch_title_batch(client, batch)
            except (DeadlineReached, MaxRequestsReached):
                raise
            except Exception as exc:  # noqa: BLE001 - one bad batch
                state.add_warning(
                    f"titles {batch[0]!r}+: {type(exc).__name__}: {exc}"
                )
                continue

            for page in pages:
                state.add_page(page)
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
def run_api_build(
    *,
    out_path: Path | str = DEFAULT_OUT,
    audit_path: Path | str = DEFAULT_AUDIT,
    n: int = 5000,
    per_page: int = DEFAULT_PER_PAGE,
    mode: str = DEFAULT_MODE,
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
    """Run enumeration+fetch and write the audit (and manifest when eligible).

    ``mode="titles"`` is the default efficient path; ``mode="generator"`` keeps
    the original ``generator=allpages`` behaviour for compatibility.
    """
    if mode not in ("titles", "generator"):
        raise ValueError(f"unknown mode: {mode!r}")
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

    state = _BuildState(n=n, per_page=per_page, max_per_page=MAX_PROMPTS_PER_PAGE)
    start = client.now()
    actual_workers = max(1, int(workers))
    buckets = [seed_list[index::actual_workers] for index in range(actual_workers)]

    common = {
        "client": client,
        "state": state,
        "start": start,
        "deadline_seconds": deadline_seconds,
        "max_requests": max_requests,
        "progress_interval_s": progress_interval_s,
    }

    if mode == "titles":
        worker = _titles_worker
        worker_kwargs = dict(common, title_queue=_TitleQueue())
    else:
        worker = _worker
        worker_kwargs = common

    if actual_workers == 1:
        worker(0, buckets[0], **worker_kwargs)
    else:
        with ThreadPoolExecutor(max_workers=actual_workers) as pool:
            futures = [
                pool.submit(worker, index, bucket, **worker_kwargs)
                for index, bucket in enumerate(buckets)
                if bucket
            ]
            for future in futures:
                future.result()

    candidates, capped_total, pages_seen = state.snapshot()
    chosen, selection = select_candidates(
        candidates, n, seed=SELECTION_SEED, max_per_page=MAX_PROMPTS_PER_PAGE
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
        "mode": mode,
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
        # ``pages_seen`` is the reconciliation unit: paragraphs considered plus
        # pages that had no extract.  ``pages_fetched`` is the raw page count.
        "pages_seen": pages_seen,
        "pages_fetched": state.pages_fetched,
        "paragraphs_seen": pages_seen - state.no_extract - state.no_revision,
        "stages": stages,
        "exclusions": {
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
