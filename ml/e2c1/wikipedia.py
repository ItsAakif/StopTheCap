"""Wikipedia extraction client for E2-C1.

Network access happens only inside :meth:`WikipediaClient.fetch_pages`. Every raw
response is cached under ``datasets/raw/e2c1_api/`` so a dry build can be replayed
offline. Tests inject ``fetch`` and never touch the network.
"""

from __future__ import annotations

import hashlib
import json
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .constants import (
    API_URL,
    CACHE_DIR,
    MAX_TITLES_PER_REQUEST,
    RETRY_BACKOFF_SECONDS,
    USER_AGENT,
)
from .prompts import make_source_url

Fetch = Callable[[str, Mapping[str, str]], dict]


class WikipediaError(RuntimeError):
    """A request failed after retries (or returned a non-maxlag API error)."""

    def __init__(
        self,
        message: str,
        *,
        titles: Sequence[str] = (),
        status: int | None = None,
    ) -> None:
        super().__init__(message)
        self.titles = tuple(titles)
        self.status = status


class WikipediaOfflineError(WikipediaError):
    """Offline mode was requested but the response is not cached."""


@dataclass(frozen=True)
class PageSnapshot:
    page_id: int
    title: str
    revision_id: str
    revision_timestamp: str
    url: str
    extract: str
    categories: tuple[str, ...]
    pageprops: Mapping[str, Any]
    redirect_from: str | None


def _default_fetch(
    url: str,
    headers: Mapping[str, str],
    *,
    timeout: float = 30.0,
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


def _is_retryable(exc: BaseException) -> bool:
    if _is_timeout(exc):
        return True
    code = _status_code(exc)
    return code is not None and (code == 429 or 500 <= code < 600)


class WikipediaClient:
    """Resolve titles to :class:`PageSnapshot` objects.

    ``fetch`` is injectable so tests can supply deterministic JSON. ``sleep`` and
    ``clock`` are also injectable so retry/backoff behaviour is testable without
    wall-clock delays.
    """

    def __init__(
        self,
        fetch: Fetch | None = None,
        cache_dir: Path | None = None,
        user_agent: str = USER_AGENT,
        min_interval_s: float = 0.2,
        *,
        offline: bool = False,
        refresh: bool = False,
        request_timeout_s: float = 30.0,
        deadline_s: float = 120.0,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.request_timeout_s = float(request_timeout_s)
        self.deadline_s = float(deadline_s)
        if fetch is None:
            self._fetch = lambda url, headers: _default_fetch(
                url, headers, timeout=self.request_timeout_s
            )
        else:
            self._fetch = fetch
        self.cache_dir = Path(cache_dir) if cache_dir is not None else Path(CACHE_DIR)
        self.user_agent = user_agent
        self.min_interval_s = float(min_interval_s)
        self.offline = bool(offline)
        self.refresh = bool(refresh)
        self._sleep = sleep
        self._clock = clock
        self._last_request_at: float | None = None
        self._stats = {
            "requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "retries": 0,
            "timeouts": 0,
        }

    # ------------------------------------------------------------------ stats
    def stats(self) -> dict[str, int]:
        return dict(self._stats)

    # -------------------------------------------------------------- transport
    @staticmethod
    def _base_params(titles: Sequence[str], exlimit: int) -> dict[str, str]:
        return {
            "action": "query",
            "format": "json",
            "formatversion": "2",
            "redirects": "1",
            "maxlag": "5",
            "prop": "extracts|revisions|categories|info|pageprops",
            "titles": "|".join(titles),
            "exlimit": str(exlimit),
            "explaintext": "1",
            "exsectionformat": "plain",
            "rvprop": "ids|timestamp",
            "rvslots": "main",
            "cllimit": "max",
            "clshow": "!hidden",
            "ppprop": "disambiguation",
        }

    @staticmethod
    def _cache_key(params: Mapping[str, Any]) -> str:
        canonical = json.dumps(
            dict(params), sort_keys=True, ensure_ascii=False, separators=(",", ":")
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]

    def _cache_path(self, params: Mapping[str, Any]) -> Path:
        return self.cache_dir / f"{self._cache_key(params)}.json"

    def _load_cache(self, params: Mapping[str, Any]) -> dict | None:
        path = self._cache_path(params)
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _store_cache(self, params: Mapping[str, Any], data: dict) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path = self._cache_path(params)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, sort_keys=True)
            handle.write("\n")

    def _wait_for_slot(self) -> None:
        if self.min_interval_s <= 0:
            self._last_request_at = self._clock()
            return
        now = self._clock()
        if self._last_request_at is not None:
            remaining = self.min_interval_s - (now - self._last_request_at)
            if remaining > 0:
                self._sleep(remaining)
        self._last_request_at = self._clock()

    def _sleep_before_retry(
        self,
        start: float,
        attempt: int,
        titles: Sequence[str],
        *,
        slept: float,
    ) -> float:
        """Sleep before a retry without exceeding the per-request deadline."""
        remaining = self.deadline_s - (self._clock() - start)
        if remaining <= 0 or slept >= self.deadline_s:
            raise WikipediaError(
                f"request deadline of {self.deadline_s:g}s exceeded for {titles}",
                titles=titles,
            )
        sleep_for = min(
            RETRY_BACKOFF_SECONDS[attempt],
            remaining,
            self.deadline_s - slept,
        )
        if sleep_for <= 0:
            raise WikipediaError(
                f"request deadline of {self.deadline_s:g}s exceeded for {titles}",
                titles=titles,
            )
        self._sleep(sleep_for)
        return slept + sleep_for

    def _request(self, params: dict[str, str]) -> dict:
        start = self._clock()
        key = self._cache_key(params)

        if self.offline:
            cached = self._load_cache(params)
            if cached is None:
                self._stats["cache_misses"] += 1
                raise WikipediaOfflineError(
                    f"offline cache miss for {key}",
                    titles=tuple(filter(None, params.get("titles", "").split("|"))),
                )
            self._stats["cache_hits"] += 1
            return cached

        if not self.refresh:
            cached = self._load_cache(params)
            if cached is not None:
                self._stats["cache_hits"] += 1
                return cached

        self._stats["cache_misses"] += 1
        url = f"{API_URL}?{urllib.parse.urlencode(params)}"
        headers = {"User-Agent": self.user_agent, "Accept": "application/json"}
        titles = tuple(filter(None, params.get("titles", "").split("|")))

        attempts = len(RETRY_BACKOFF_SECONDS) + 1
        slept = 0.0
        for attempt in range(attempts):
            self._wait_for_slot()
            self._stats["requests"] += 1
            try:
                data = self._fetch(url, headers)
            except Exception as exc:  # noqa: BLE001 - classified below
                if _is_timeout(exc):
                    self._stats["timeouts"] += 1
                if _is_retryable(exc) and attempt < len(RETRY_BACKOFF_SECONDS):
                    slept = self._sleep_before_retry(
                        start, attempt, titles, slept=slept
                    )
                    self._stats["retries"] += 1
                    continue
                raise WikipediaError(
                    f"HTTP failure for {titles}: {exc}",
                    titles=titles,
                    status=_status_code(exc),
                ) from exc

            if not isinstance(data, dict):
                raise WikipediaError(
                    f"non-object JSON response for {titles}", titles=titles
                )

            error = data.get("error")
            if error:
                code = str(error.get("code", ""))
                if code == "maxlag" and attempt < len(RETRY_BACKOFF_SECONDS):
                    slept = self._sleep_before_retry(
                        start, attempt, titles, slept=slept
                    )
                    self._stats["retries"] += 1
                    continue
                raise WikipediaError(
                    f"API error {code}: {error.get('info', '')}", titles=titles
                )

            self._store_cache(params, data)
            return data

        raise WikipediaError(f"request failed after retries: {titles}", titles=titles)

    # ----------------------------------------------------------- page merging
    @staticmethod
    def _merge_page(store: dict[str, dict], page: Mapping[str, Any]) -> None:
        title = page.get("title")
        if not isinstance(title, str) or not title:
            return
        slot = store.setdefault(title, {})
        slot.setdefault("title", title)
        for key in ("pageid", "missing", "invalid", "ns", "contentmodel"):
            if key in page:
                slot[key] = page[key]
        pageprops = page.get("pageprops")
        if isinstance(pageprops, Mapping):
            merged = dict(slot.get("pageprops") or {})
            merged.update(pageprops)
            slot["pageprops"] = merged
        revisions = page.get("revisions")
        if revisions and "_revision" not in slot:
            slot["_revision"] = revisions[0]
        extract = page.get("extract")
        if isinstance(extract, str) and extract:
            chunks = slot.setdefault("_extracts", [])
            if extract not in chunks:
                chunks.append(extract)
        categories = page.get("categories")
        if categories:
            kept = slot.setdefault("_categories", [])
            for category in categories:
                name = category.get("title") if isinstance(category, Mapping) else None
                if isinstance(name, str) and name and name not in kept:
                    kept.append(name)

    @staticmethod
    def _resolve_title(
        title: str,
        normalized: Mapping[str, str],
        redirects: Mapping[str, str],
    ) -> str:
        current = title
        seen: set[str] = set()
        for _ in range(16):
            if current in seen:
                break
            seen.add(current)
            if current in normalized and normalized[current] != current:
                current = normalized[current]
                continue
            if current in redirects and redirects[current] != current:
                current = redirects[current]
                continue
            break
        return current

    @staticmethod
    def _is_missing(page: Mapping[str, Any]) -> bool:
        return "missing" in page or "invalid" in page

    def _snapshot(self, page: Mapping[str, Any], requested: str) -> PageSnapshot:
        revision = page.get("_revision") or {}
        revision_id = str(revision.get("revid", "")) if revision else ""
        timestamp = str(revision.get("timestamp", "")) if revision else ""
        canonical = str(page.get("title") or requested)
        return PageSnapshot(
            page_id=int(page["pageid"]) if page.get("pageid") is not None else -1,
            title=canonical,
            revision_id=revision_id,
            revision_timestamp=timestamp,
            url=make_source_url(revision_id),
            extract="\n".join(page.get("_extracts", [])),
            categories=tuple(page.get("_categories", [])),
            pageprops=dict(page.get("pageprops") or {}),
            redirect_from=requested if requested != canonical else None,
        )

    def _collect(
        self,
        params: dict[str, str],
        store: dict[str, dict],
        normalized: dict[str, str],
        redirects: dict[str, str],
    ) -> None:
        pending = dict(params)
        while True:
            data = self._request(pending)
            query = data.get("query") or {}
            for item in query.get("normalized") or []:
                if isinstance(item, Mapping) and item.get("from") and item.get("to"):
                    normalized[str(item["from"])] = str(item["to"])
            for item in query.get("redirects") or []:
                if isinstance(item, Mapping) and item.get("from") and item.get("to"):
                    redirects[str(item["from"])] = str(item["to"])
            for page in query.get("pages") or []:
                if isinstance(page, Mapping):
                    self._merge_page(store, page)

            continuation = data.get("continue")
            if not continuation:
                return
            pending = dict(params)
            pending.update({str(k): str(v) for k, v in continuation.items()})

    # ----------------------------------------------------------------- public
    def fetch_pages(self, titles: Sequence[str]) -> dict[str, PageSnapshot | None]:
        requested = list(dict.fromkeys(titles))
        result: dict[str, PageSnapshot | None] = {title: None for title in requested}
        if not requested:
            return result

        for start in range(0, len(requested), MAX_TITLES_PER_REQUEST):
            batch = requested[start : start + MAX_TITLES_PER_REQUEST]
            store: dict[str, dict] = {}
            normalized: dict[str, str] = {}
            redirects: dict[str, str] = {}
            self._collect(
                self._base_params(batch, MAX_TITLES_PER_REQUEST),
                store,
                normalized,
                redirects,
            )

            missing_extract: list[str] = []
            for title in batch:
                resolved = self._resolve_title(title, normalized, redirects)
                page = store.get(resolved)
                if page is None or self._is_missing(page):
                    result[title] = None
                    continue
                snapshot = self._snapshot(page, title)
                result[title] = snapshot
                if not snapshot.extract.strip():
                    missing_extract.append(title)

            # Full extracts may come back one-per-request even with exlimit=20 and
            # no continuation remains, so fetch those titles alone.
            for title in missing_extract:
                resolved = self._resolve_title(title, normalized, redirects)
                single_store: dict[str, dict] = {}
                single_normalized: dict[str, str] = {}
                single_redirects: dict[str, str] = {}
                self._collect(
                    self._base_params([resolved], 1),
                    single_store,
                    single_normalized,
                    single_redirects,
                )
                final = self._resolve_title(
                    resolved, single_normalized, single_redirects
                )
                page = single_store.get(final)
                if page is not None and not self._is_missing(page):
                    result[title] = self._snapshot(page, title)

        return result
