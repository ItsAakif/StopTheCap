"""Offline tests for the wikitext-revision prompt builder.

Every test injects ``fetch``; nothing touches the network or the real API cache.
"""

from __future__ import annotations

import random
import urllib.parse

from ml.e2c1.api_prompts import ApiClient
from ml.e2c1.constants import MAX_PROMPTS_PER_PAGE
from ml.e2c1.prompts import read_manifest, validate_manifest
from ml.e2c1.wikitext_prompts import (
    _fetch_revision_batch,
    _parse_wikitext,
    build_enumeration_params,
    build_revisions_params,
    parse_page,
    run_wikitext_build,
    select_candidates,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def word_text(seed: str, n: int = 60) -> str:
    return " ".join(f"{seed}{index}" for index in range(n)) + "."


def revision_page(
    title: str,
    pageid: int,
    wikitext: str,
    *,
    revision_id: int | None = None,
    timestamp: str = "2026-01-01T00:00:00Z",
) -> dict:
    return {
        "pageid": pageid,
        "title": title,
        "ns": 0,
        "revisions": [
            {
                "revid": revision_id if revision_id is not None else 2_000_000 + pageid,
                "timestamp": timestamp,
                "slots": {"main": {"contentmodel": "wikitext", "content": wikitext}},
            }
        ],
    }


def parse_text(
    wikitext: str, title: str = "Example topic", sections: str = "all"
) -> dict:
    return parse_page(revision_page(title, 1, wikitext), sections=sections)


def squash(text: str) -> str:
    return " ".join(text.split())


def request_params(url: str) -> dict[str, str]:
    query = urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)
    return {key: values[0] for key, values in query.items()}


class WikiFetch:
    """Param-aware fake for enumeration and revision batches.

    ``enumeration`` maps ``("from", seed)`` or ``("continue", token)`` to
    ``(titles, next_apcontinue)``.  ``page_factory`` turns a title into a raw
    API page dict.
    """

    def __init__(self, enumeration, page_factory) -> None:
        self.enumeration = dict(enumeration)
        self.page_factory = page_factory
        self.enumerations: list[tuple[str, str]] = []
        self.revision_batches: list[list[str]] = []
        self.revision_params: list[dict[str, str]] = []

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
        self.revision_batches.append(batch)
        self.revision_params.append(params)
        return {"query": {"pages": [self.page_factory(title) for title in batch]}}


# ---------------------------------------------------------------------------
# Request shape
# ---------------------------------------------------------------------------
def test_enumeration_and_revision_request_shapes():
    enumeration = build_enumeration_params("A")
    assert enumeration["list"] == "allpages"
    assert enumeration["apnamespace"] == "0"
    assert enumeration["apfrom"] == "A"
    assert enumeration["aplimit"] == "50"
    assert enumeration["apfilterredir"] == "nonredirects"

    continued = build_enumeration_params("A", apcontinue="B|7")
    assert continued["apcontinue"] == "B|7"
    assert "apfrom" not in continued

    titles = [f"Title {index}" for index in range(50)]
    snapshot = build_revisions_params(titles)
    assert snapshot["redirects"] == "1"
    assert snapshot["prop"] == "revisions"
    assert snapshot["rvprop"] == "ids|timestamp|content"
    assert snapshot["rvslots"] == "main"
    assert snapshot["formatversion"] == "2"
    assert len(snapshot["titles"].split("|")) == 50


# ---------------------------------------------------------------------------
# Parser: individual rules
# ---------------------------------------------------------------------------
def test_comments_are_stripped():
    parsed = parse_text("Visible<!-- hidden --> text.")
    assert parsed["paragraphs"] == ["Visible text."]


def test_refs_are_stripped_in_both_forms():
    parsed = parse_text(
        'Alpha<ref name="a">a citation</ref> beta.<ref name="b"/> Gamma.'
    )
    assert parsed["paragraphs"] == ["Alpha beta. Gamma."]


def test_nested_templates_are_removed_iteratively():
    parsed = parse_text("Before {{outer|{{inner|deep}}}} after.")
    assert squash(parsed["paragraphs"][0]) == "Before after."


def test_infobox_is_removed_and_lead_survives():
    parsed = parse_text(
        "{{Infobox person\n| name = X\n| birth_date = Y\n}}\n\nLead paragraph."
    )
    assert parsed["paragraphs"] == ["Lead paragraph."]


def test_tables_are_removed():
    parsed = parse_text(
        'Lead.\n\n{| class="wikitable"\n|-\n| cell\n|}\n\nAfter.'
    )
    assert parsed["paragraphs"] == ["Lead.", "After."]


def test_wikilinks_with_and_without_labels_are_unwrapped():
    parsed = parse_text("See [[Target|the label]] and [[Other]].")
    assert parsed["paragraphs"] == ["See the label and Other."]


def test_bold_and_italic_markers_are_stripped():
    parsed = parse_text("This is '''bold''' and ''italic''.")
    assert parsed["paragraphs"] == ["This is bold and italic."]


def test_external_links_become_their_label():
    parsed = parse_text("Read [https://example.com the source] now.")
    assert parsed["paragraphs"] == ["Read the source now."]


def test_categories_are_captured_and_removed_from_text():
    parsed = parse_text("Text [[Category:Physics]] and [[Category:Science|sort]].")
    assert parsed["categories"] == ["Category:Physics", "Category:Science"]
    paragraph = parsed["paragraphs"][0]
    assert "Category:" not in paragraph
    assert "[[" not in paragraph
    assert squash(paragraph) == "Text and ."


def test_lead_section_cutoff_drops_later_sections():
    parsed = parse_text(
        "Lead paragraph.\n\n== History ==\nHidden paragraph.", sections="lead"
    )
    assert parsed["paragraphs"] == ["Lead paragraph."]


def test_all_sections_extend_past_headings_to_body_paragraphs():
    wikitext = (
        "Short lead."
        "\n\n== First section ==\n"
        + word_text("bodyone", 60)
        + "\n\n== Second section ==\n"
        + word_text("bodytwo", 60)
    )
    parsed = parse_text(wikitext)
    assert parsed["paragraphs"] == [
        "Short lead.",
        word_text("bodyone", 60),
        word_text("bodytwo", 60),
    ]


def test_heading_lines_are_never_emitted_as_paragraphs():
    categories, paragraphs, rejects = _parse_wikitext(
        "Lead line.\n\n== History ==\nBody line.\n\n=== Subsection ===\nMore body."
    )
    assert categories == []
    assert paragraphs == ["Lead line.", "Body line.", "More body."]
    assert rejects == 0
    assert all("==" not in paragraph for paragraph in paragraphs)


def test_list_like_and_unpunctuated_paragraphs_are_rejected():
    keep = word_text("keep", 60)
    wikitext = (
        keep
        + "\n\n* item "
        + word_text("star", 55)
        + "\n\n# item "
        + word_text("hash", 55)
        + "\n\n"
        + " ".join(f"punctless{index}" for index in range(60))
    )
    categories, paragraphs, rejects = _parse_wikitext(wikitext)
    assert categories == []
    assert paragraphs == [keep]
    assert rejects == 3


def test_lead_mode_still_matches_the_old_lead_only_behaviour():
    wikitext = (
        word_text("lead", 60)
        + "\n\n== History ==\n"
        + word_text("hidden", 60)
    )
    parsed = parse_text(wikitext, sections="lead")
    assert parsed["paragraphs"] == [word_text("lead", 60)]


def test_disambiguation_is_detected_three_ways():
    assert parse_text("Text.", title="Mercury (disambiguation)")["is_disambiguation"]
    assert parse_text("{{Disambiguation}}\n\nText.")["is_disambiguation"]
    assert parse_text("Text [[Category:Disambiguation pages]].")["is_disambiguation"]
    assert not parse_text("Text.")["is_disambiguation"]


def test_paragraphs_with_residual_markup_are_rejected():
    categories, paragraphs, rejects = _parse_wikitext(
        "Keep this paragraph.\n\nReject [[bad link.\n\nAlso reject | pipe."
    )
    assert categories == []
    assert paragraphs == ["Keep this paragraph."]
    assert rejects == 2


def test_categories_inside_comments_and_tables_are_not_captured():
    parsed = parse_text(
        "Lead paragraph.\n\n"
        "<!-- [[Category:Hidden]] -->\n\n"
        "{| class=\"wikitable\"\n| [[Category:TableOnly]]\n|}\n\n"
        "[[Category:Visible]]"
    )
    assert parsed["categories"] == ["Category:Visible"]


# ---------------------------------------------------------------------------
# Parser: realistic fixture
# ---------------------------------------------------------------------------
REALISTIC_WIKITEXT = """'''Example''' is a synthetic topic in [[Physics]].<ref name="a">{{cite web|url=https://example.com}}</ref>

It has [[Second|two]] distinct paragraphs and a [[Third]] reference.{{Infobox topic
| name = Example
| image = Example.png
}}

[[File:Example.png|thumb|A caption]]

{| class="wikitable"
|-
! Header
|-
| Ignored cell
|}

== History ==
This history paragraph must never become a prompt.

[[Category:Physics]]
[[Category:Science|sort key]]
[[Category:History]]
"""


def test_realistic_fixture_yields_two_clean_paragraphs():
    parsed = parse_page(
        revision_page("Example", 42, REALISTIC_WIKITEXT), sections="lead"
    )
    assert parsed["paragraphs"] == [
        "Example is a synthetic topic in Physics.",
        "It has two distinct paragraphs and a Third reference.",
    ]
    assert parsed["categories"] == [
        "Category:Physics",
        "Category:Science",
        "Category:History",
    ]
    assert parsed["page_id"] == 42
    assert parsed["title"] == "Example"
    assert parsed["revision_id"] == "2000042"
    assert parsed["revision_timestamp"] == "2026-01-01T00:00:00Z"
    assert parsed["url"].endswith("oldid=2000042")
    assert parsed["is_disambiguation"] is False


# ---------------------------------------------------------------------------
# Transport: batching, continuation, dedupe, redirects
# ---------------------------------------------------------------------------
def test_batches_are_exactly_fifty_titles_and_redirects_is_sent(tmp_path):
    titles = [f"Topic {index}" for index in range(120)]
    enumeration = {
        ("from", "A"): (titles[:50], "A2"),
        ("continue", "A2"): (titles[50:100], "A3"),
        ("continue", "A3"): (titles[100:], None),
    }
    page_ids = {title: 3000 + index for index, title in enumerate(titles)}

    def page_factory(title):
        return revision_page(title, page_ids[title], word_text(f"t{page_ids[title]}"))

    fetch = WikiFetch(enumeration, page_factory)
    audit, code = run_wikitext_build(
        out_path=tmp_path / "out.jsonl",
        audit_path=tmp_path / "audit.json",
        n=120,
        per_page=1,
        workers=1,
        min_interval_s=0,
        max_requests=100,
        deadline_seconds=100,
        seeds=("A",),
        fetch=fetch,
        cache_dir=tmp_path / "cache",
    )

    assert code == 0
    assert audit["status"] == "ok"
    assert [len(batch) for batch in fetch.revision_batches] == [50, 50, 20]
    for params in fetch.revision_params:
        assert params["redirects"] == "1"
        assert params["prop"] == "revisions"
        assert params["rvslots"] == "main"
    assert fetch.enumerations == [
        ("from", "A"),
        ("continue", "A2"),
        ("continue", "A3"),
    ]


def test_apcontinue_is_followed_and_titles_are_deduplicated_across_seeds(tmp_path):
    enumeration = {
        ("from", "A"): (["Alpha", "Shared"], None),
        ("from", "B"): (["Shared", "Gamma"], None),
    }
    page_ids = {"Alpha": 5001, "Shared": 5002, "Gamma": 5003}

    def page_factory(title):
        return revision_page(title, page_ids[title], word_text(f"w{page_ids[title]}"))

    fetch = WikiFetch(enumeration, page_factory)
    audit, code = run_wikitext_build(
        out_path=tmp_path / "out.jsonl",
        audit_path=tmp_path / "audit.json",
        n=3,
        per_page=1,
        workers=1,
        min_interval_s=0,
        max_requests=50,
        deadline_seconds=100,
        seeds=("A", "B"),
        fetch=fetch,
        cache_dir=tmp_path / "cache",
    )

    assert code == 0
    assert fetch.enumerations == [("from", "A"), ("from", "B")]
    fetched = [title for batch in fetch.revision_batches for title in batch]
    assert fetched == ["Alpha", "Shared", "Gamma"]
    assert fetched.count("Shared") == 1
    assert audit["pages_fetched"] == 3


def test_normalized_and_redirected_titles_map_back_to_requested(tmp_path):
    class RedirectFetch:
        def __call__(self, url, headers):
            return {
                "query": {
                    "normalized": [{"from": "topic", "to": "Topic"}],
                    "redirects": [{"from": "Topic", "to": "Target"}],
                    "pages": [
                        revision_page("Target", 77, word_text("redirect", 60))
                    ],
                }
            }

    client = ApiClient(
        fetch=RedirectFetch(),
        cache_dir=tmp_path / "cache",
        min_interval_s=0,
    )
    pages = _fetch_revision_batch(client, ["topic"])

    assert len(pages) == 1
    assert pages[0]["title"] == "Target"
    assert pages[0]["requested_title"] == "topic"
    assert pages[0]["revision_id"] == "2000077"
    assert "redirect0" in pages[0]["wikitext"]


# ---------------------------------------------------------------------------
# Lead-only end-to-end
# ---------------------------------------------------------------------------
def test_content_after_history_heading_never_reaches_a_prompt(tmp_path):
    wikitext = (
        word_text("lead", 60)
        + "\n\n== History ==\n"
        + word_text("hidden", 60)
    )

    def page_factory(title):
        return revision_page(title, 7001, wikitext)

    fetch = WikiFetch({("from", "A"): (["Lead only"], None)}, page_factory)
    audit, code = run_wikitext_build(
        out_path=tmp_path / "out.jsonl",
        audit_path=tmp_path / "audit.json",
        n=1,
        per_page=5,
        workers=1,
        min_interval_s=0,
        max_requests=10,
        deadline_seconds=100,
        seeds=("A",),
        fetch=fetch,
        cache_dir=tmp_path / "cache",
        sections="lead",
        retrieved_at="2026-11-06",
    )

    assert code == 0
    rows = read_manifest(tmp_path / "out.jsonl")
    assert len(rows) == 1
    assert "hidden0" not in rows[0]["prompt_text"]
    assert "lead0" in rows[0]["prompt_text"]
    assert audit["parser_rejects"] == 0


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------
def _candidates(count: int = 40, pages: int = 10) -> list[dict]:
    return [
        {"prompt_id": f"pmt:{index:024x}", "page_id": index % pages}
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


def test_selection_applies_the_global_page_cap():
    candidates = [
        {"prompt_id": f"pmt:{index:024x}", "page_id": 7} for index in range(9)
    ]
    chosen, stats = select_candidates(candidates, 8)
    assert len(chosen) == MAX_PROMPTS_PER_PAGE
    assert stats["page_cap_removed"] == 7


def test_per_page_caps_accepted_paragraphs(tmp_path):
    wikitext = "\n\n".join(word_text(f"paragraph{index}", 60) for index in range(6))

    def page_factory(title):
        return revision_page(title, 9200, wikitext)

    fetch = WikiFetch({("from", "A"): (["Many paragraphs"], None)}, page_factory)
    audit, code = run_wikitext_build(
        out_path=tmp_path / "out.jsonl",
        audit_path=tmp_path / "audit.json",
        n=1,
        per_page=2,
        workers=1,
        min_interval_s=0,
        max_requests=10,
        deadline_seconds=100,
        seeds=("A",),
        fetch=fetch,
        cache_dir=tmp_path / "cache",
    )

    assert code == 0
    assert audit["selection"]["candidates"] == 2
    assert audit["paragraphs_seen"] == 2


def test_max_per_page_allows_three_prompts_from_one_page(tmp_path):
    wikitext = "\n\n".join(word_text(f"cap{index}", 60) for index in range(4))

    def page_factory(title):
        return revision_page(title, 9300, wikitext)

    fetch = WikiFetch({("from", "A"): (["One page"], None)}, page_factory)
    audit, code = run_wikitext_build(
        out_path=tmp_path / "out.jsonl",
        audit_path=tmp_path / "audit.json",
        n=3,
        per_page=5,
        max_per_page=3,
        workers=1,
        min_interval_s=0,
        max_requests=10,
        deadline_seconds=100,
        seeds=("A",),
        fetch=fetch,
        cache_dir=tmp_path / "cache",
        retrieved_at="2026-11-06",
    )

    assert code == 0
    rows = read_manifest(tmp_path / "out.jsonl")
    assert len(rows) == 3
    assert {row["page_id"] for row in rows} == {9300}
    assert len({row["family_id"] for row in rows}) == 1


def test_headings_do_not_reach_prompt_text(tmp_path):
    wikitext = (
        word_text("lead", 60)
        + "\n\n== History ==\n"
        + word_text("body", 60)
    )

    def page_factory(title):
        return revision_page(title, 9350, wikitext)

    fetch = WikiFetch({("from", "A"): (["Headings"], None)}, page_factory)
    audit, code = run_wikitext_build(
        out_path=tmp_path / "out.jsonl",
        audit_path=tmp_path / "audit.json",
        n=2,
        per_page=5,
        max_per_page=3,
        workers=1,
        min_interval_s=0,
        max_requests=10,
        deadline_seconds=100,
        seeds=("A",),
        fetch=fetch,
        cache_dir=tmp_path / "cache",
        retrieved_at="2026-11-06",
    )

    assert code == 0
    rows = read_manifest(tmp_path / "out.jsonl")
    assert len(rows) == 2
    assert any("body0" in row["prompt_text"] for row in rows)
    assert all("==" not in row["prompt_text"] for row in rows)


def test_second_build_with_the_same_client_issues_zero_network_calls(tmp_path):
    calls: list[str] = []

    class CountingFetch(WikiFetch):
        def __call__(self, url, headers):
            calls.append(url)
            return super().__call__(url, headers)

    enumeration = {("from", "A"): (["Cached page"], None)}

    def page_factory(title):
        return revision_page(title, 9400, word_text("cached", 60))

    fetch = CountingFetch(enumeration, page_factory)
    client = ApiClient(fetch=fetch, cache_dir=tmp_path / "cache", min_interval_s=0)

    for index in range(2):
        audit, code = run_wikitext_build(
            out_path=tmp_path / f"out{index}.jsonl",
            audit_path=tmp_path / f"audit{index}.json",
            n=1,
            per_page=1,
            max_per_page=3,
            workers=1,
            min_interval_s=0,
            max_requests=10,
            deadline_seconds=100,
            seeds=("A",),
            client=client,
            retrieved_at="2026-11-06",
        )
        assert code == 0
        assert audit["status"] == "ok"
        if index == 0:
            calls_after_first = len(calls)

    assert calls_after_first > 0
    assert len(calls) == calls_after_first


# ---------------------------------------------------------------------------
# End-to-end audit / manifest
# ---------------------------------------------------------------------------
def test_audit_stage_counts_and_reconciliation_balance(tmp_path):
    pages = {
        "Empty": revision_page("Empty", 1, ""),
        "Short": revision_page("Short", 2, word_text("short", 5)),
        "NonLatin": revision_page("NonLatin", 3, " ".join(["日本語"] * 60) + "."),
        "Birth": revision_page(
            "Birth", 4, word_text("birth", 60) + "\n\n[[Category:1990 births]]"
        ),
        "Survivor": revision_page("Survivor", 5, word_text("survivor", 60)),
    }

    def page_factory(title):
        return pages[title]

    fetch = WikiFetch({("from", "0"): (list(pages), None)}, page_factory)
    audit, code = run_wikitext_build(
        out_path=tmp_path / "out.jsonl",
        audit_path=tmp_path / "audit.json",
        n=1,
        per_page=1,
        workers=1,
        min_interval_s=0,
        max_requests=10,
        deadline_seconds=100,
        seeds=("0",),
        fetch=fetch,
        cache_dir=tmp_path / "cache",
        retrieved_at="2026-11-06",
    )

    assert code == 0
    assert audit["status"] == "ok"
    assert audit["mode"] == "wikitext"
    assert audit["parser_rejects"] == 0
    by_name = {stage["stage"]: stage for stage in audit["stages"]}
    assert by_name["no_extract"]["removed"] == 1
    assert by_name["length"]["removed"] == 1
    assert by_name["non_latin"]["removed"] == 1
    assert by_name["safety:excluded_biography"]["removed"] == 1

    removed = sum(stage["removed"] for stage in audit["stages"])
    reconciliation = audit["reconciliation"]
    assert reconciliation["pages_seen"] - removed == reconciliation["selected"]
    assert reconciliation["selected"] == len(read_manifest(tmp_path / "out.jsonl"))
    assert reconciliation["balanced"] is True
    assert audit["pages_per_request"] == round(
        audit["pages_fetched"] / audit["requests"], 3
    )


def test_parser_rejects_are_counted_and_reconciled(tmp_path):
    wikitext = word_text("bad", 60) + " | residue"
    page = revision_page("Rejected", 9, wikitext)
    fetch = WikiFetch({("from", "A"): (["Rejected"], None)}, lambda title: page)

    audit, code = run_wikitext_build(
        out_path=tmp_path / "out.jsonl",
        audit_path=tmp_path / "audit.json",
        n=1,
        per_page=1,
        workers=1,
        min_interval_s=0,
        max_requests=10,
        deadline_seconds=100,
        seeds=("A",),
        fetch=fetch,
        cache_dir=tmp_path / "cache",
    )

    # The single paragraph is rejected, the page yields no usable prompt, and the
    # pool is short; the audit still reconciles.
    assert code == 3
    assert audit["status"] == "insufficient_pool"
    assert audit["parser_rejects"] == 1
    by_name = {stage["stage"]: stage for stage in audit["stages"]}
    assert by_name["parser_reject"]["removed"] == 1
    removed = sum(stage["removed"] for stage in audit["stages"])
    assert audit["reconciliation"]["pages_seen"] - removed == 0
    assert audit["reconciliation"]["balanced"] is True


def test_build_stops_early_once_n_is_reachable(tmp_path):
    titles = [f"Topic {index}" for index in range(200)]
    enumeration = {
        ("from", "A"): (titles[:50], "A2"),
        ("continue", "A2"): (titles[50:100], "A3"),
        ("continue", "A3"): (titles[100:150], "A4"),
        ("continue", "A4"): (titles[150:], None),
    }

    wikitext = word_text("p1", 60) + "\n\n" + word_text("p2", 60)
    page_ids = {title: 8000 + index for index, title in enumerate(titles)}

    def page_factory(title):
        return revision_page(title, page_ids[title], wikitext)

    fetch = WikiFetch(enumeration, page_factory)
    audit, code = run_wikitext_build(
        out_path=tmp_path / "out.jsonl",
        audit_path=tmp_path / "audit.json",
        n=10,
        per_page=5,
        workers=1,
        min_interval_s=0,
        max_requests=100,
        deadline_seconds=100,
        seeds=("A",),
        fetch=fetch,
        cache_dir=tmp_path / "cache",
    )

    assert code == 0
    assert audit["status"] == "ok"
    assert len(fetch.revision_batches) == 1
    assert len(fetch.revision_batches[0]) == 50
    assert fetch.enumerations == [("from", "A")]
    assert audit["selection"]["selected"] == 10


def test_manifest_contains_no_residual_wikitext_markup(tmp_path):
    wikitext = (
        "'''Marked''' text with [[a link|a label]] and a ref.<ref>cite</ref>\n\n"
        + word_text("second", 60)
        + "\n\n[[Category:Physics]]\n"
    )

    def page_factory(title):
        return revision_page(title, 9001, wikitext)

    fetch = WikiFetch({("from", "A"): (["Clean page"], None)}, page_factory)
    audit, code = run_wikitext_build(
        out_path=tmp_path / "out.jsonl",
        audit_path=tmp_path / "audit.json",
        n=1,
        per_page=5,
        workers=1,
        min_interval_s=0,
        max_requests=10,
        deadline_seconds=100,
        seeds=("A",),
        fetch=fetch,
        cache_dir=tmp_path / "cache",
        retrieved_at="2026-11-06",
    )

    assert code == 0
    rows = read_manifest(tmp_path / "out.jsonl")
    assert rows
    for row in rows:
        for marker in ("[[", "]]", "{{", "}}", "<ref", "|"):
            assert marker not in row["prompt_text"]


def test_manifest_schema_matches_the_api_builder(tmp_path):
    def page_factory(title):
        return revision_page(title, 9100, word_text("schema", 60))

    fetch = WikiFetch({("from", "A"): (["Schema page"], None)}, page_factory)
    audit, code = run_wikitext_build(
        out_path=tmp_path / "out.jsonl",
        audit_path=tmp_path / "audit.json",
        n=1,
        per_page=1,
        workers=1,
        min_interval_s=0,
        max_requests=10,
        deadline_seconds=100,
        seeds=("A",),
        fetch=fetch,
        cache_dir=tmp_path / "cache",
        retrieved_at="2026-11-06",
    )

    assert code == 0
    rows = read_manifest(tmp_path / "out.jsonl")
    assert validate_manifest(rows, expected=1) == []
    row = rows[0]
    for key in (
        "page_id",
        "words",
        "script_latin",
        "family_id",
        "topic_rule",
        "content_safety_status",
        "prompt_text",
        "source_url",
        "revision_id",
    ):
        assert key in row
    assert row["content_safety_status"] == "ok"
    assert row["script_latin"] is True
    assert audit["output"]["count"] == 1
