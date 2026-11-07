"""Offline tests for the Wikipedia prompt builder.

Covers the default two-phase ``titles`` path and the legacy
``generator=allpages`` path.  Every test injects ``fetch``; nothing here touches
the network or the repo's real API cache.
"""

from __future__ import annotations

import json
import random
import re
import urllib.parse

import pytest

from ml.e2c1.api_prompts import (
    ApiClient,
    ApiError,
    _fetch_title_batch,
    build_allpages_params,
    build_params,
    build_titles_params,
    classify_paragraph,
    run_api_build,
    select_candidates,
    split_paragraphs,
)
from ml.e2c1.constants import MAX_PROMPTS_PER_PAGE
from ml.e2c1.prompts import read_manifest, validate_manifest

FAMILY_RE = re.compile(r"^fam:[0-9a-f]{16}$")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def word_text(seed: str, n: int = 60) -> str:
    return " ".join(f"{seed}{index}" for index in range(n))


def make_page(
    index: int,
    *,
    extract: str | None = None,
    title: str | None = None,
    categories: tuple[str, ...] = (),
    pageprops: dict | None = None,
) -> dict:
    return {
        "pageid": 1000 + index,
        "title": title or f"Topic Page {index}",
        "extract": word_text(f"page{index}") if extract is None else extract,
        "revisions": [
            {"revid": 1_000_000 + index, "timestamp": "2026-01-01T00:00:00Z"}
        ],
        "categories": [{"title": category} for category in categories],
        "pageprops": dict(pageprops or {}),
    }


def api_response(pages: list[dict], continuation: dict | None = None) -> dict:
    data: dict = {"query": {"pages": pages}}
    if continuation:
        data["continue"] = dict(continuation)
    else:
        data["batchcomplete"] = True
    return data


class SequenceFetch:
    def __init__(self, outcomes) -> None:
        self.outcomes = list(outcomes)
        self.calls = 0

    def __call__(self, url: str, headers):
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class FixedFetch:
    """Serve one fixed response, then empty pages (no continuation)."""

    def __init__(self, pages: list[dict], continuation: dict | None = None) -> None:
        self.pages = pages
        self.continuation = continuation
        self.calls = 0
        self.urls: list[str] = []

    def __call__(self, url: str, headers):
        self.calls += 1
        self.urls.append(url)
        if self.calls == 1:
            return api_response(self.pages, self.continuation)
        return api_response([])


class FakeClock:
    def __init__(self, start: float = 0.0) -> None:
        self.value = float(start)

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += float(seconds)


class HttpFailure(Exception):
    def __init__(self, code: int, *, retry_after: str | None = None) -> None:
        super().__init__(f"HTTP {code}")
        self.code = code
        self.headers = {}
        if retry_after is not None:
            self.headers["Retry-After"] = retry_after


# ---------------------------------------------------------------------------
# Paragraph splitting and acceptance
# ---------------------------------------------------------------------------
def test_split_paragraphs_on_blank_lines_and_limit():
    assert split_paragraphs("one\n\ntwo\n\n\nthree\n\nfour") == [
        "one",
        "two",
        "three",
        "four",
    ]
    assert split_paragraphs("one\n\ntwo\n\nthree\n\nfour", limit=3) == [
        "one",
        "two",
        "three",
    ]
    assert split_paragraphs("a\r\n\r\nb") == ["a", "b"]
    assert split_paragraphs("   \n\n  ") == []
    assert split_paragraphs("") == []


def test_classify_paragraph_accepts_and_rejects_each_rule():
    good = word_text("good")
    assert classify_paragraph(good, title="Neutral topic") == "ok"

    assert (
        classify_paragraph("only a few words", title="Neutral topic") == "length"
    )
    too_long = word_text("long", 301)
    assert classify_paragraph(too_long, title="Neutral topic") == "length"

    non_latin = " ".join(["日本語"] * 60)
    assert classify_paragraph(non_latin, title="Neutral topic") == "non_latin"

    assert (
        classify_paragraph(
            good,
            title="Neutral topic",
            categories=("Category:1990 births",),
        )
        == "safety:excluded_biography"
    )


# ---------------------------------------------------------------------------
# Request shape / transport behaviour
# ---------------------------------------------------------------------------
def test_redirects_is_never_sent_with_generator_allpages():
    for seed in ("0", "A", "M", "Wa"):
        request = build_params(seed)
        assert request["generator"] == "allpages"
        assert "redirects" not in request
        assert request["gapfilterredir"] == "nonredirects"
        assert request["gapfrom"] == seed
        assert request["exintro"] == "1"
        assert request["exlimit"] == "max"

    override = build_params("A", gapfrom="B")
    assert override["gapfrom"] == "B"
    assert "redirects" not in override


def test_429_honours_retry_after_with_global_pause(tmp_path):
    clock = FakeClock()
    sleeps: list[float] = []

    def sleep(seconds: float) -> None:
        sleeps.append(float(seconds))
        clock.advance(seconds)

    fetch = SequenceFetch(
        [
            HttpFailure(429, retry_after="3"),
            api_response([make_page(1)]),
        ]
    )
    client = ApiClient(
        fetch=fetch,
        cache_dir=tmp_path / "cache",
        min_interval_s=0,
        sleep=sleep,
        clock=clock,
    )

    result = client.request(build_params("A"))

    assert result["query"]["pages"][0]["pageid"] == 1001
    assert sleeps == [3.0]
    assert client.stats()["rate429"] == 1
    assert client.stats()["retries"] == 1

    # The pause is shared: a second worker waits for the same deadline.
    client.pause(5)
    client._wait_for_slot()
    assert sleeps == [3.0, 5.0]


def test_persistent_429_reduces_effective_concurrency(tmp_path):
    clock = FakeClock()
    sleeps: list[float] = []

    def sleep(seconds: float) -> None:
        sleeps.append(float(seconds))
        clock.advance(seconds)

    fetch = SequenceFetch([HttpFailure(429, retry_after="7") for _ in range(8)])
    client = ApiClient(
        fetch=fetch,
        cache_dir=tmp_path / "cache",
        min_interval_s=0,
        workers=4,
        sleep=sleep,
        clock=clock,
    )

    with pytest.raises(ApiError):
        client.request(build_params("A"))

    assert client.effective_workers == 2
    assert client.stats()["rate429"] == 4
    assert all(seconds == 7.0 for seconds in sleeps)


def test_5xx_uses_exponential_backoff(tmp_path):
    clock = FakeClock()
    sleeps: list[float] = []

    def sleep(seconds: float) -> None:
        sleeps.append(float(seconds))
        clock.advance(seconds)

    fetch = SequenceFetch(
        [HttpFailure(503), api_response([make_page(1)])]
    )
    client = ApiClient(
        fetch=fetch,
        cache_dir=tmp_path / "cache",
        min_interval_s=0,
        sleep=sleep,
        clock=clock,
    )

    client.request(build_params("A"))
    assert sleeps == [1.0]
    assert client.stats()["retries"] == 1


def test_maxlag_uses_exponential_backoff(tmp_path):
    clock = FakeClock()
    sleeps: list[float] = []

    def sleep(seconds: float) -> None:
        sleeps.append(float(seconds))
        clock.advance(seconds)

    fetch = SequenceFetch(
        [
            {"error": {"code": "maxlag", "info": "lagged"}},
            api_response([make_page(1)]),
        ]
    )
    client = ApiClient(
        fetch=fetch,
        cache_dir=tmp_path / "cache",
        min_interval_s=0,
        sleep=sleep,
        clock=clock,
    )

    client.request(build_params("A"))
    assert sleeps == [1.0]
    assert client.stats()["maxlag"] == 1
    assert client.stats()["retries"] == 1


def test_cache_is_read_before_the_network(tmp_path):
    cache_dir = tmp_path / "cache"
    request = build_params("A")
    first = ApiClient(
        fetch=FixedFetch([make_page(1)]),
        cache_dir=cache_dir,
        min_interval_s=0,
    )
    first.request(request)
    assert len(list(cache_dir.glob("*.json"))) == 1
    assert len(next(cache_dir.glob("*.json")).stem) == 16

    def exploding(url, headers):
        raise AssertionError("cache hit must not call the network")

    second = ApiClient(fetch=exploding, cache_dir=cache_dir, min_interval_s=0)
    result = second.request(request)
    assert result["query"]["pages"][0]["pageid"] == 1001
    assert second.stats()["cache_hits"] == 1
    assert second.stats()["requests"] == 0


# ---------------------------------------------------------------------------
# Selection determinism
# ---------------------------------------------------------------------------
def _candidates(count: int = 40, pages: int = 10) -> list[dict]:
    return [
        {
            "prompt_id": f"pmt:{index:024x}",
            "page_id": index % pages,
        }
        for index in range(count)
    ]


def test_selection_is_deterministic_under_shuffled_input():
    candidates = _candidates()
    first, first_stats = select_candidates(candidates, 10)

    shuffled = list(candidates)
    random.Random(17).shuffle(shuffled)
    second, second_stats = select_candidates(shuffled, 10)

    assert [row["prompt_id"] for row in first] == [
        row["prompt_id"] for row in second
    ]
    assert first_stats == second_stats
    assert len(first) == 10


def test_selection_applies_two_per_page_cap():
    candidates = [{"prompt_id": f"pmt:{index:024x}", "page_id": 7} for index in range(9)]
    chosen, stats = select_candidates(candidates, 8)
    assert len(chosen) == MAX_PROMPTS_PER_PAGE
    assert stats["page_cap_removed"] == 7


# ---------------------------------------------------------------------------
# End-to-end offline builds
# ---------------------------------------------------------------------------
def test_manifest_rows_are_unique_and_have_valid_family_ids(tmp_path):
    pages = [make_page(index) for index in range(8)]
    out_path = tmp_path / "e2_c1_prompts.jsonl"
    audit_path = tmp_path / "e2_c1_prompt_audit.json"

    audit, code = run_api_build(
        out_path=out_path,
        audit_path=audit_path,
        n=4,
        per_page=1,
        mode="generator",
        workers=1,
        min_interval_s=0,
        max_requests=10,
        deadline_seconds=100,
        seeds=("0",),
        fetch=FixedFetch(pages),
        cache_dir=tmp_path / "cache",
        retrieved_at="2026-11-06",
    )

    assert code == 0
    assert audit["status"] == "ok"
    rows = read_manifest(out_path)
    assert len(rows) == 4
    assert validate_manifest(rows, expected=4) == []
    assert len({row["prompt_id"] for row in rows}) == 4
    assert len({row["prompt_hash"] for row in rows}) == 4
    for row in rows:
        assert FAMILY_RE.match(row["family_id"])


def test_audit_stage_counts_and_reconciliation_balance(tmp_path):
    pages = [
        make_page(0, extract=""),  # no extract
        make_page(1, extract=word_text("short", 5)),  # length
        make_page(2, extract=" ".join(["日本語"] * 60)),  # non_latin
        make_page(3, categories=("Category:1990 births",)),  # safety
        make_page(4),  # survivor
    ]
    out_path = tmp_path / "e2_c1_prompts.jsonl"
    audit_path = tmp_path / "e2_c1_prompt_audit.json"

    audit, code = run_api_build(
        out_path=out_path,
        audit_path=audit_path,
        n=1,
        per_page=1,
        mode="generator",
        workers=1,
        min_interval_s=0,
        max_requests=10,
        deadline_seconds=100,
        seeds=("0",),
        fetch=FixedFetch(pages),
        cache_dir=tmp_path / "cache",
    )

    assert code == 0
    by_name = {stage["stage"]: stage for stage in audit["stages"]}
    assert by_name["no_extract"]["removed"] == 1
    assert by_name["length"]["removed"] == 1
    assert by_name["non_latin"]["removed"] == 1
    assert by_name["safety:excluded_biography"]["removed"] == 1
    assert any(name.startswith("safety:") for name in by_name)

    removed = sum(stage["removed"] for stage in audit["stages"])
    reconciliation = audit["reconciliation"]
    assert reconciliation["pages_seen"] - removed == reconciliation["selected"]
    assert reconciliation["selected"] == len(read_manifest(out_path))
    assert reconciliation["balanced"] is True


def test_deadline_stops_cleanly_without_a_short_manifest(tmp_path):
    out_path = tmp_path / "e2_c1_prompts.jsonl"
    audit_path = tmp_path / "e2_c1_prompt_audit.json"
    fetch = FixedFetch([make_page(index) for index in range(5)])

    audit, code = run_api_build(
        out_path=out_path,
        audit_path=audit_path,
        n=10,
        per_page=3,
        mode="generator",
        workers=1,
        min_interval_s=0,
        max_requests=10,
        deadline_seconds=0,
        seeds=("0",),
        fetch=fetch,
        cache_dir=tmp_path / "cache",
    )

    assert code == 3
    assert audit["status"] == "deadline_reached"
    assert audit["output"]["count"] == 0
    assert not out_path.exists()
    assert audit_path.exists()
    assert fetch.calls == 0
    stored = json.loads(audit_path.read_text(encoding="utf-8"))
    assert stored["status"] == "deadline_reached"


def test_build_stops_after_enough_usable_prompts(tmp_path):
    # Each request returns five pages and a gapcontinue.  Two batches are enough
    # for n=6, so the third page range must never be requested.
    class BatchedFetch:
        def __init__(self) -> None:
            self.calls = 0

        def __call__(self, url, headers):
            self.calls += 1
            start = self.calls * 10
            pages = [make_page(start + offset) for offset in range(5)]
            return api_response(
                pages,
                {"continue": "||", "gapcontinue": str(start + 5)},
            )

    fetch = BatchedFetch()
    audit, code = run_api_build(
        out_path=tmp_path / "out.jsonl",
        audit_path=tmp_path / "audit.json",
        n=6,
        per_page=1,
        mode="generator",
        workers=1,
        min_interval_s=0,
        max_requests=50,
        deadline_seconds=100,
        seeds=("0",),
        fetch=fetch,
        cache_dir=tmp_path / "cache",
    )

    assert code == 0
    assert audit["status"] == "ok"
    assert audit["selection"]["selected"] == 6
    assert fetch.calls == 2


# ---------------------------------------------------------------------------
# Two-phase ``titles`` mode
# ---------------------------------------------------------------------------
def request_params(url: str) -> dict[str, str]:
    query = urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)
    return {key: values[0] for key, values in query.items()}


def titled_page(title: str, pageid: int, *, paragraphs: int = 1) -> dict:
    extract = "\n\n".join(
        word_text(f"page{pageid}p{index}") for index in range(paragraphs)
    )
    return {
        "pageid": pageid,
        "title": title,
        "extract": extract,
        "revisions": [
            {"revid": 2_000_000 + pageid, "timestamp": "2026-01-01T00:00:00Z"}
        ],
        "categories": [],
        "pageprops": {},
    }


class TitleFetch:
    """Param-aware fake for ``titles`` mode.

    ``enumeration`` maps ``("from", seed)`` or ``("continue", token)`` to
    ``(titles, next_apcontinue)``.  Snapshot requests return one page per title.
    """

    def __init__(self, enumeration, page_factory) -> None:
        self.enumeration = dict(enumeration)
        self.page_factory = page_factory
        self.enumerations: list[tuple[str, str]] = []
        self.title_batches: list[list[str]] = []

    def __call__(self, url, headers):
        params = request_params(url)
        if params.get("list") == "allpages":
            if "apcontinue" in params:
                key = ("continue", params["apcontinue"])
            else:
                key = ("from", params["apfrom"])
            self.enumerations.append(key)
            titles, next_continue = self.enumeration[key]
            data = {"query": {"allpages": [{"title": title} for title in titles]}}
            if next_continue is not None:
                data["continue"] = {"apcontinue": next_continue}
            return data

        batch = params["titles"].split("|")
        self.title_batches.append(batch)
        return {"query": {"pages": [self.page_factory(title) for title in batch]}}


def test_enumeration_params_omit_redirects_while_title_params_include_it():
    enumeration = build_allpages_params("A")
    assert enumeration["list"] == "allpages"
    assert enumeration["apfrom"] == "A"
    assert enumeration["aplimit"] == "50"
    assert enumeration["apfilterredir"] == "nonredirects"
    assert "redirects" not in enumeration
    assert "prop" not in enumeration

    continued = build_allpages_params("A", apcontinue="B|7")
    assert continued["apcontinue"] == "B|7"
    assert "apfrom" not in continued
    assert "redirects" not in continued

    snapshot = build_titles_params([f"Title {index}" for index in range(20)])
    assert snapshot["redirects"] == "1"
    assert snapshot["exlimit"] == "20"
    assert "generator" not in snapshot
    assert len(snapshot["titles"].split("|")) == 20


def test_titles_are_batched_twenty_at_a_time(tmp_path):
    all_titles = [f"Title {index}" for index in range(45)]
    page_ids = {title: 3000 + index for index, title in enumerate(all_titles)}

    def page_factory(title):
        return titled_page(title, page_ids[title])

    fetch = TitleFetch({("from", "0"): (all_titles, None)}, page_factory)
    audit, code = run_api_build(
        out_path=tmp_path / "out.jsonl",
        audit_path=tmp_path / "audit.json",
        n=45,
        per_page=1,
        mode="titles",
        workers=1,
        min_interval_s=0,
        max_requests=50,
        deadline_seconds=100,
        seeds=("0",),
        fetch=fetch,
        cache_dir=tmp_path / "cache",
    )

    assert code == 0
    assert [len(batch) for batch in fetch.title_batches] == [20, 20, 5]
    # One titles request returned all 20 pages in its batch.
    assert len(fetch.title_batches[0]) == 20
    assert audit["pages_fetched"] == 45
    assert audit["requests"] == 4
    assert audit["pages_per_request"] == round(45 / 4, 3)


def test_batches_stay_full_across_enumeration_pages(tmp_path):
    # Two allpages pages of 35 titles.  Draining greedily would send a 15-title
    # batch between them; the top-up rule keeps every batch full but the last.
    enumeration = {
        ("from", "A"): ([f"First {index}" for index in range(35)], "A2"),
        ("continue", "A2"): ([f"Second {index}" for index in range(35)], None),
    }
    page_ids: dict[str, int] = {}

    def page_factory(title):
        page_ids.setdefault(title, 4000 + len(page_ids))
        return titled_page(title, page_ids[title])

    fetch = TitleFetch(enumeration, page_factory)
    audit, code = run_api_build(
        out_path=tmp_path / "out.jsonl",
        audit_path=tmp_path / "audit.json",
        n=70,
        per_page=1,
        mode="titles",
        workers=1,
        min_interval_s=0,
        max_requests=50,
        deadline_seconds=100,
        seeds=("A",),
        fetch=fetch,
        cache_dir=tmp_path / "cache",
    )

    assert code == 0
    assert [len(batch) for batch in fetch.title_batches] == [20, 20, 20, 10]
    assert audit["pages_fetched"] == 70


def test_apcontinue_is_followed_and_duplicate_titles_are_deduplicated(tmp_path):
    enumeration = {
        ("from", "A"): (["Alpha", "Shared"], "A2"),
        ("continue", "A2"): (["Beta", "Shared"], None),
        ("from", "B"): (["Shared", "Gamma"], None),
    }
    page_ids: dict[str, int] = {}

    def page_factory(title):
        page_ids.setdefault(title, 5000 + len(page_ids))
        return titled_page(title, page_ids[title])

    fetch = TitleFetch(enumeration, page_factory)
    audit, code = run_api_build(
        out_path=tmp_path / "out.jsonl",
        audit_path=tmp_path / "audit.json",
        n=4,
        per_page=1,
        mode="titles",
        workers=1,
        min_interval_s=0,
        max_requests=50,
        deadline_seconds=100,
        seeds=("A", "B"),
        fetch=fetch,
        cache_dir=tmp_path / "cache",
    )

    assert code == 0
    assert fetch.enumerations == [
        ("from", "A"),
        ("continue", "A2"),
        ("from", "B"),
    ]
    fetched = [title for batch in fetch.title_batches for title in batch]
    assert fetched.count("Shared") == 1
    assert set(fetched) == {"Alpha", "Shared", "Beta", "Gamma"}
    assert audit["pages_fetched"] == 4


def test_title_fetch_follows_extract_and_category_continuations(tmp_path):
    class ContinuationFetch:
        def __init__(self) -> None:
            self.calls: list[dict[str, str]] = []

        def __call__(self, url, headers):
            params = request_params(url)
            self.calls.append(params)
            if "excontinue" in params or "clcontinue" in params:
                return {
                    "query": {
                        "pages": [
                            {
                                "pageid": 1,
                                "title": "Topic",
                                "extract": word_text("second"),
                                "categories": [{"title": "Category:Second"}],
                                "revisions": [
                                    {
                                        "revid": 42,
                                        "timestamp": "2026-01-01T00:00:00Z",
                                    }
                                ],
                            }
                        ]
                    }
                }
            return {
                "query": {
                    "pages": [
                        {
                            "pageid": 1,
                            "title": "Topic",
                            "extract": word_text("first"),
                            "categories": [{"title": "Category:First"}],
                            "revisions": [
                                {
                                    "revid": 42,
                                    "timestamp": "2026-01-01T00:00:00Z",
                                }
                            ],
                        }
                    ]
                },
                "continue": {
                    "continue": "||",
                    "excontinue": "1",
                    "clcontinue": "C2",
                },
            }

    fetch = ContinuationFetch()
    client = ApiClient(
        fetch=fetch,
        cache_dir=tmp_path / "cache",
        min_interval_s=0,
    )

    pages = _fetch_title_batch(client, ["Topic"])

    assert len(pages) == 1
    assert "first" in pages[0]["extract"]
    assert "second" in pages[0]["extract"]
    assert set(pages[0]["categories"]) == {"Category:First", "Category:Second"}
    assert len(fetch.calls) == 2
    assert fetch.calls[1]["excontinue"] == "1"
    assert fetch.calls[1]["clcontinue"] == "C2"


def test_normalized_and_redirected_titles_map_back_to_requested(tmp_path):
    class RedirectFetch:
        def __call__(self, url, headers):
            return {
                "query": {
                    "normalized": [{"from": "topic", "to": "Topic"}],
                    "redirects": [{"from": "Topic", "to": "Target"}],
                    "pages": [
                        {
                            "pageid": 77,
                            "title": "Target",
                            "extract": word_text("redirect"),
                            "categories": [],
                            "revisions": [
                                {
                                    "revid": 77,
                                    "timestamp": "2026-01-01T00:00:00Z",
                                }
                            ],
                        }
                    ],
                }
            }

    client = ApiClient(
        fetch=RedirectFetch(),
        cache_dir=tmp_path / "cache",
        min_interval_s=0,
    )

    pages = _fetch_title_batch(client, ["topic"])

    assert len(pages) == 1
    assert pages[0]["title"] == "Target"
    assert pages[0]["requested_title"] == "topic"


def test_enumeration_stops_early_once_yield_shows_target_is_reachable(tmp_path):
    seeds = ("A", "B", "C", "D", "E", "F")
    enumeration = {
        ("from", seed): ([f"{seed} Topic {index}" for index in range(50)], None)
        for seed in seeds
    }
    page_ids: dict[str, int] = {}

    def page_factory(title):
        page_ids.setdefault(title, 7000 + len(page_ids))
        return titled_page(title, page_ids[title], paragraphs=5)

    fetch = TitleFetch(enumeration, page_factory)
    audit, code = run_api_build(
        out_path=tmp_path / "out.jsonl",
        audit_path=tmp_path / "audit.json",
        n=40,
        per_page=5,
        mode="titles",
        workers=1,
        min_interval_s=0,
        max_requests=100,
        deadline_seconds=100,
        seeds=seeds,
        fetch=fetch,
        cache_dir=tmp_path / "cache",
    )

    assert code == 0
    # The high yield from the first 20-title batch stops enumeration at seed A.
    assert fetch.enumerations == [("from", "A")]
    assert len(fetch.title_batches) == 1
    assert len(fetch.title_batches[0]) == 20
    assert audit["pages_fetched"] == 20
    assert audit["selection"]["selected"] == 40


def test_pages_per_request_is_recorded_and_reconciles(tmp_path):
    all_titles = [f"Reconcile {index}" for index in range(10)]
    page_ids = {title: 9000 + index for index, title in enumerate(all_titles)}

    def page_factory(title):
        return titled_page(title, page_ids[title])

    fetch = TitleFetch({("from", "0"): (all_titles, None)}, page_factory)
    audit, code = run_api_build(
        out_path=tmp_path / "out.jsonl",
        audit_path=tmp_path / "audit.json",
        n=10,
        per_page=1,
        mode="titles",
        workers=1,
        min_interval_s=0,
        max_requests=20,
        deadline_seconds=100,
        seeds=("0",),
        fetch=fetch,
        cache_dir=tmp_path / "cache",
    )

    assert code == 0
    assert audit["requests"] == 2
    assert audit["pages_fetched"] == 10
    assert audit["pages_per_request"] == 5.0
    assert audit["pages_per_request"] == round(
        audit["pages_fetched"] / audit["requests"], 3
    )
