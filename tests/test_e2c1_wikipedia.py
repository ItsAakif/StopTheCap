"""Wikipedia client: injected fetch, continuations, fallbacks, cache and retries."""

from __future__ import annotations

import json
import socket
import urllib.error
import urllib.parse

import pytest

from ml.e2c1 import wikipedia
from ml.e2c1.constants import USER_AGENT
from ml.e2c1.wikipedia import (
    PageSnapshot,
    WikipediaClient,
    WikipediaError,
    WikipediaOfflineError,
)


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    """Never read or write the repo's real API cache from tests."""
    monkeypatch.setattr(wikipedia, "CACHE_DIR", str(tmp_path / "default_cache"))


class HttpFailure(Exception):
    def __init__(self, code: int) -> None:
        super().__init__(f"HTTP {code}")
        self.code = code


def page(
    title: str,
    *,
    pageid: int = 1,
    revid: int = 100,
    extract: str | None = None,
    categories: tuple[str, ...] = (),
    pageprops: dict | None = None,
    missing: bool = False,
) -> dict:
    result: dict = {"pageid": pageid, "ns": 0, "title": title}
    if missing:
        result["missing"] = True
        return result
    if extract is not None:
        result["extract"] = extract
    result["revisions"] = [{"revid": revid, "timestamp": "2020-01-01T00:00:00Z"}]
    if categories:
        result["categories"] = [{"title": category} for category in categories]
    if pageprops is not None:
        result["pageprops"] = pageprops
    return result


def response(
    pages: list[dict],
    *,
    normalized: list[dict] | None = None,
    redirects: list[dict] | None = None,
    continuation: dict | None = None,
) -> dict:
    query: dict = {"pages": pages}
    if normalized is not None:
        query["normalized"] = normalized
    if redirects is not None:
        query["redirects"] = redirects
    data: dict = {"query": query}
    if continuation is not None:
        data["continue"] = continuation
    else:
        data["batchcomplete"] = True
    return data


def params(url: str) -> dict[str, str]:
    parsed = urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)
    return {key: values[0] for key, values in parsed.items()}


class FakeFetch:
    def __init__(self, responder) -> None:
        self.responder = responder
        self.calls: list[tuple[str, dict]] = []

    def __call__(self, url: str, headers) -> dict:
        self.calls.append((url, dict(headers)))
        return self.responder(url, len(self.calls))


class SequenceFetch:
    def __init__(self, outcomes) -> None:
        self.outcomes = list(outcomes)
        self.calls = 0

    def __call__(self, url: str, headers) -> dict:
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def first_page(url: str, _call: int, *, extract: str = "text") -> dict:
    titles = params(url)["titles"].split("|")
    pages = [
        page(title, pageid=index + 1, revid=1000 + index, extract=f"{extract} {title}")
        for index, title in enumerate(titles)
    ]
    return response(pages)


def test_stats_start_at_zero_with_required_counters():
    client = WikipediaClient(fetch=lambda url, headers: {}, min_interval_s=0)
    assert client.stats() == {
        "requests": 0,
        "cache_hits": 0,
        "cache_misses": 0,
        "retries": 0,
        "timeouts": 0,
    }


class FakeHttpResponse:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def read(self) -> bytes:
        return self._payload


@pytest.mark.parametrize(("configured", "expected"), [(None, 30.0), (7.5, 7.5)])
def test_urlopen_receives_configured_timeout(tmp_path, monkeypatch, configured, expected):
    seen: list[float | None] = []
    payload = json.dumps(response([page("Alpha", extract="text")])).encode("utf-8")

    def fake_urlopen(request, timeout=None):
        seen.append(timeout)
        return FakeHttpResponse(payload)

    monkeypatch.setattr(wikipedia.urllib.request, "urlopen", fake_urlopen)
    kwargs = {} if configured is None else {"request_timeout_s": configured}
    client = WikipediaClient(
        cache_dir=tmp_path / "cache", min_interval_s=0, **kwargs
    )

    client.fetch_pages(["Alpha"])

    assert seen == [expected]


def test_empty_request_does_not_call_fetch():
    calls = []

    def fetch(url, headers):
        calls.append(url)
        return {}

    assert WikipediaClient(fetch=fetch).fetch_pages([]) == {}
    assert calls == []


def test_batches_titles_at_twenty_and_sends_user_agent():
    fake = FakeFetch(first_page)
    client = WikipediaClient(fetch=fake, min_interval_s=0)
    titles = [f"Title {index}" for index in range(25)]

    result = client.fetch_pages(titles)

    assert set(result) == set(titles)
    assert all(isinstance(snapshot, PageSnapshot) for snapshot in result.values())
    assert len(fake.calls) == 2
    assert all(len(params(url)["titles"].split("|")) <= 20 for url, _ in fake.calls)
    assert all(headers["User-Agent"] == USER_AGENT for _, headers in fake.calls)


def test_follows_excontinue_and_concatenates_chunks():
    def responder(url, _):
        current = params(url)
        if "excontinue" not in current:
            return response(
                [page("Alpha", extract="part one")],
                continuation={"continue": "||", "excontinue": "2"},
            )
        return response([page("Alpha", extract="part two")])

    fake = FakeFetch(responder)
    client = WikipediaClient(fetch=fake, min_interval_s=0)

    result = client.fetch_pages(["Alpha"])

    assert result["Alpha"].extract == "part one\npart two"
    assert client.stats()["requests"] == 2


def test_follows_clcontinue_and_deduplicates_categories():
    def responder(url, _):
        current = params(url)
        if "clcontinue" not in current:
            return response(
                [
                    page(
                        "Alpha",
                        extract="text",
                        categories=("Category:A", "Category:B"),
                    )
                ],
                continuation={"continue": "||", "clcontinue": "x"},
            )
        return response([page("Alpha", categories=("Category:B", "Category:C"))])

    client = WikipediaClient(fetch=FakeFetch(responder), min_interval_s=0)
    result = client.fetch_pages(["Alpha"])

    assert result["Alpha"].categories == (
        "Category:A",
        "Category:B",
        "Category:C",
    )


def test_single_title_refetch_when_extract_missing():
    def responder(url, _):
        current = params(url)
        if current["titles"] == "Alpha|Beta":
            return response(
                [
                    page("Alpha", pageid=1, revid=11, extract="alpha text"),
                    page("Beta", pageid=2, revid=22),
                ]
            )
        assert current["titles"] == "Beta"
        assert current["exlimit"] == "1"
        return response([page("Beta", pageid=2, revid=22, extract="beta text")])

    client = WikipediaClient(fetch=FakeFetch(responder), min_interval_s=0)
    result = client.fetch_pages(["Alpha", "Beta"])

    assert result["Alpha"].extract == "alpha text"
    assert result["Beta"].extract == "beta text"


def test_maps_normalized_and_redirects_back_to_requested_title():
    def responder(url, _):
        return response(
            [page("Foo Bar", extract="canonical text")],
            normalized=[{"from": "foo bar", "to": "Foo bar"}],
            redirects=[{"from": "Foo bar", "to": "Foo Bar"}],
        )

    client = WikipediaClient(fetch=FakeFetch(responder), min_interval_s=0)
    result = client.fetch_pages(["foo bar"])

    assert set(result) == {"foo bar"}
    assert result["foo bar"].title == "Foo Bar"
    assert result["foo bar"].redirect_from == "foo bar"


def test_missing_page_returns_none():
    client = WikipediaClient(
        fetch=FakeFetch(lambda url, _: response([page("Gone", missing=True)])),
        min_interval_s=0,
    )
    assert client.fetch_pages(["Gone"]) == {"Gone": None}


def test_cache_hit_and_miss(tmp_path):
    cache_dir = tmp_path / "cache"
    first_fake = FakeFetch(first_page)
    first = WikipediaClient(fetch=first_fake, cache_dir=cache_dir, min_interval_s=0)
    first.fetch_pages(["Alpha"])

    assert first.stats()["cache_misses"] == 1
    assert first.stats()["requests"] == 1
    cache_files = list(cache_dir.glob("*.json"))
    assert len(cache_files) == 1
    assert len(cache_files[0].stem) == 16
    json.loads(cache_files[0].read_text(encoding="utf-8"))

    second_fake = FakeFetch(first_page)
    second = WikipediaClient(fetch=second_fake, cache_dir=cache_dir, min_interval_s=0)
    result = second.fetch_pages(["Alpha"])

    assert result["Alpha"].extract == "text Alpha"
    assert second_fake.calls == []
    assert second.stats()["cache_hits"] == 1
    assert second.stats()["requests"] == 0


def test_offline_cache_miss_raises_without_network(tmp_path):
    fake = FakeFetch(first_page)
    client = WikipediaClient(
        fetch=fake,
        cache_dir=tmp_path / "empty",
        offline=True,
        min_interval_s=0,
    )
    with pytest.raises(WikipediaOfflineError):
        client.fetch_pages(["Alpha"])
    assert fake.calls == []
    assert client.stats()["cache_misses"] == 1


def test_refresh_bypasses_cached_response(tmp_path):
    cache_dir = tmp_path / "cache"
    WikipediaClient(
        fetch=FakeFetch(first_page), cache_dir=cache_dir, min_interval_s=0
    ).fetch_pages(["Alpha"])

    refresh_fake = FakeFetch(first_page)
    client = WikipediaClient(
        fetch=refresh_fake,
        cache_dir=cache_dir,
        refresh=True,
        min_interval_s=0,
    )
    client.fetch_pages(["Alpha"])
    assert len(refresh_fake.calls) == 1
    assert client.stats()["requests"] == 1


def test_retries_429_with_one_second_backoff():
    sleeps: list[float] = []
    fake = SequenceFetch(
        [HttpFailure(429), response([page("Alpha", extract="ok")])]
    )
    client = WikipediaClient(
        fetch=fake, min_interval_s=0, sleep=sleeps.append
    )
    result = client.fetch_pages(["Alpha"])

    assert result["Alpha"].extract == "ok"
    assert sleeps == [1.0]
    assert client.stats()["retries"] == 1
    assert client.stats()["requests"] == 2


def test_retries_maxlag_with_exponential_backoff():
    sleeps: list[float] = []
    fake = SequenceFetch(
        [
            {"error": {"code": "maxlag", "info": "lagged"}},
            HttpFailure(503),
            response([page("Alpha", extract="ok")]),
        ]
    )
    client = WikipediaClient(
        fetch=fake, min_interval_s=0, sleep=sleeps.append
    )
    client.fetch_pages(["Alpha"])

    assert sleeps == [1.0, 2.0]
    assert client.stats()["retries"] == 2


def test_gives_up_after_three_retries():
    sleeps: list[float] = []
    fake = SequenceFetch([HttpFailure(500) for _ in range(4)])
    client = WikipediaClient(
        fetch=fake, min_interval_s=0, sleep=sleeps.append
    )
    with pytest.raises(WikipediaError):
        client.fetch_pages(["Alpha"])

    assert sleeps == [1.0, 2.0, 4.0]
    assert client.stats()["retries"] == 3
    assert client.stats()["requests"] == 4


def test_timeout_is_retried_and_counted_in_stats():
    sleeps: list[float] = []
    fake = SequenceFetch(
        [socket.timeout("timed out"), response([page("Alpha", extract="ok")])]
    )
    client = WikipediaClient(
        fetch=fake, min_interval_s=0, sleep=sleeps.append
    )
    result = client.fetch_pages(["Alpha"])

    assert result["Alpha"].extract == "ok"
    assert sleeps == [1.0]
    assert client.stats()["retries"] == 1
    assert client.stats()["requests"] == 2
    assert client.stats()["timeouts"] == 1


def test_url_error_with_timeout_reason_is_retried_and_counted():
    sleeps: list[float] = []
    fake = SequenceFetch(
        [
            urllib.error.URLError(socket.timeout("timed out")),
            response([page("Alpha", extract="ok")]),
        ]
    )
    client = WikipediaClient(
        fetch=fake, min_interval_s=0, sleep=sleeps.append
    )
    client.fetch_pages(["Alpha"])

    assert sleeps == [1.0]
    assert client.stats()["timeouts"] == 1


def test_deadline_stops_timeout_retry_loop():
    sleeps: list[float] = []
    calls: list[str] = []

    class AdvancingClock:
        def __init__(self) -> None:
            self.value = 0.0

        def __call__(self) -> float:
            current = self.value
            self.value += 30.0
            return current

    def fetch(url, headers):
        calls.append(url)
        raise socket.timeout("timed out")

    client = WikipediaClient(
        fetch=fetch,
        min_interval_s=0,
        sleep=sleeps.append,
        clock=AdvancingClock(),
        deadline_s=120.0,
    )

    with pytest.raises(WikipediaError, match="deadline"):
        client.fetch_pages(["Alpha"])

    assert sum(sleeps) <= 120.0
    assert len(calls) < 10
    assert client.stats()["timeouts"] >= 1
