"""Pure prompt-inventory functions: taxonomy, safety, hashes, selection, manifest."""

from __future__ import annotations

import copy
import hashlib
import json
import random

import pytest

from ml.e2c1 import prompts
from ml.e2c1.constants import (
    MAX_WORDS,
    MIN_WORDS,
    SELECTION_SEED,
    TOPIC_RULES,
    TOPIC_TAXONOMY,
)
from ml.ingest.normalise import word_count


def good_text(seed: str = "alpha", n: int = 60) -> str:
    return " ".join(f"{seed}{i}" for i in range(n))


def make_row(
    text: str | None = None,
    *,
    revision_id: str = "1234567890",
    page_id: int = 1,
    title: str = "Example",
    topic: str = "other",
    rule: str = "other",
) -> dict:
    text = text if text is not None else good_text()
    prompt_hash = prompts.make_prompt_hash(text)
    return {
        "prompt_id": prompts.make_prompt_id(revision_id, prompt_hash),
        "source_title": title,
        "source_url": prompts.make_source_url(revision_id),
        "revision_id": revision_id,
        "retrieved_at": "2026-11-06",
        "source_licence": "CC-BY-SA-4.0",
        "attribution": prompts.make_attribution(
            title, revision_id, "2026-11-06", prompts.make_source_url(revision_id)
        ),
        "prompt_text": text,
        "prompt_hash": prompt_hash,
        "topic_category": topic,
        "content_safety_status": "ok",
        "page_id": page_id,
        "words": word_count(text),
        "script_latin": True,
        "topic_rule": rule,
    }


# ---------------------------------------------------------------------------
# Taxonomy
# ---------------------------------------------------------------------------
def test_taxonomy_has_exactly_ten_ordered_unique_values():
    assert len(TOPIC_TAXONOMY) == 10
    assert len(set(TOPIC_TAXONOMY)) == 10
    assert TOPIC_TAXONOMY[-1] == "other"
    assert [rule[0] for rule in TOPIC_RULES] == list(TOPIC_TAXONOMY)


def test_every_non_other_rule_has_keywords():
    for topic, category_keywords, title_keywords in TOPIC_RULES:
        if topic == "other":
            assert category_keywords == () and title_keywords == ()
        else:
            assert category_keywords, f"{topic} has no category keywords"
            assert title_keywords, f"{topic} has no title keywords"


# ---------------------------------------------------------------------------
# Latin ratio / script
# ---------------------------------------------------------------------------
def test_latin_ratio_ascii_is_one():
    assert prompts.latin_ratio("The quick brown fox.") == 1.0


def test_latin_ratio_cjk_is_zero():
    assert prompts.latin_ratio("日本語のテキスト") == 0.0


def test_latin_ratio_mixed_and_nfkc():
    assert prompts.latin_ratio("abcあ") == pytest.approx(0.75)
    # Fullwidth Latin composes to ASCII under NFKC.
    assert prompts.latin_ratio("ＡＢＣ") == 1.0
    assert prompts.latin_ratio("1234 — !!!") == 0.0
    assert prompts.latin_ratio("") == 0.0


def test_script_is_latin_threshold():
    assert prompts.script_is_latin("a" * 19 + "あ") is True  # 19/20 = 0.95
    assert prompts.script_is_latin("a" * 18 + "ああ") is False


# ---------------------------------------------------------------------------
# Topic category
# ---------------------------------------------------------------------------
def test_category_match_wins_over_title_and_uses_rule_prefix():
    topic, rule = prompts.topic_category(["Category:Medicine"], "History of Rome")
    assert topic == "medicine_health"
    assert rule == "category:Medicine"


def test_topic_rules_are_ordered():
    topic, rule = prompts.topic_category(
        ["Category:Physics", "Category:History"], "An unrelated title"
    )
    assert topic == "stem"
    assert rule == "category:Physics"


def test_title_match_used_when_no_category_matches():
    topic, rule = prompts.topic_category([], "A river in France")
    assert topic == "geography_places"
    assert rule == "title:river"


def test_topic_falls_back_to_other():
    assert prompts.topic_category(["Category:Unrelated"], "Nothing here") == (
        "other",
        "other",
    )


# ---------------------------------------------------------------------------
# Safety
# ---------------------------------------------------------------------------
def test_safety_ok():
    status, reasons = prompts.safety_status(
        title="Neutral article",
        text=good_text(),
        categories=(),
        pageprops={},
    )
    assert status == "ok"
    assert reasons == ()


def test_safety_non_latin():
    text = " ".join(["日本語"] * 60)
    status, reasons = prompts.safety_status(
        title="Japanese", text=text, categories=(), pageprops={}
    )
    assert status == "excluded_non_latin"
    assert reasons == ("non_latin",)


def test_safety_length_short_and_long():
    status, reasons = prompts.safety_status(
        title="Short", text="only a few words", categories=(), pageprops={}
    )
    assert status == "excluded_length_short"
    assert reasons == ("length_short",)

    status, reasons = prompts.safety_status(
        title="Long", text=good_text(n=MAX_WORDS + 1), categories=(), pageprops={}
    )
    assert status == "excluded_length_long"
    assert reasons == ("length_long",)


def test_safety_disambiguation_from_pageprops_and_title():
    status, reasons = prompts.safety_status(
        title="Mercury",
        text=good_text(),
        categories=(),
        pageprops={"disambiguation": ""},
    )
    assert status == "excluded_disambiguation"
    assert reasons == ("disambiguation",)

    status, _ = prompts.safety_status(
        title="Mercury (disambiguation)",
        text=good_text(),
        categories=(),
        pageprops={},
    )
    assert status == "excluded_disambiguation"


def test_safety_list_page_from_title_and_category():
    assert prompts.safety_status(
        title="List of rivers", text=good_text(), categories=(), pageprops={}
    )[0] == "excluded_list"
    assert prompts.safety_status(
        title="Rivers",
        text=good_text(),
        categories=("Category:Lists of rivers",),
        pageprops={},
    )[0] == "excluded_list"


def test_safety_biography():
    assert prompts.safety_status(
        title="Someone",
        text=good_text(),
        categories=("Category:1990 births",),
        pageprops={},
    )[0] == "excluded_biography"
    # The person-category regex intentionally over-excludes occupations.
    assert prompts.safety_status(
        title="Film directors",
        text=good_text(),
        categories=("Category:Film directors",),
        pageprops={},
    )[0] == "excluded_biography"


def test_safety_blocklist():
    status, reasons = prompts.safety_status(
        title="Neutral title",
        text=good_text() + " massacre",
        categories=(),
        pageprops={},
    )
    assert status == "excluded_blocklist"
    assert reasons == ("blocklist",)


def test_safety_quote_heavy_and_apostrophes_are_not_quotes():
    quoted = (
        f'"{("alpha " * 12).strip()}" '
        f'{("ordinary " * 35)}'
        f'"{("beta " * 13).strip()}"'
    )
    status, reasons = prompts.safety_status(
        title="Quoted", text=quoted, categories=(), pageprops={}
    )
    assert status == "excluded_quote_heavy"
    assert reasons == ("quote_heavy",)

    apostrophes = "don't won't can't " * 30
    status, _ = prompts.safety_status(
        title="Apostrophes", text=apostrophes, categories=(), pageprops={}
    )
    assert status == "ok"


def test_safety_boilerplate():
    status, reasons = prompts.safety_status(
        title="Boilerplate",
        text="This article is about " + good_text(),
        categories=(),
        pageprops={},
    )
    assert status == "excluded_boilerplate"
    assert reasons == ("boilerplate",)


def test_safety_records_all_reasons_but_first_wins():
    status, reasons = prompts.safety_status(
        title="A person",
        text=good_text() + " massacre",
        categories=("Category:1990 births",),
        pageprops={},
    )
    assert status == "excluded_biography"
    assert reasons == ("biography", "blocklist")

    # list_page comes before biography in rule order.
    status, reasons = prompts.safety_status(
        title="List of people",
        text=good_text(),
        categories=("Category:1990 births",),
        pageprops={},
    )
    assert status == "excluded_list"
    assert reasons == ("list_page", "biography")


# ---------------------------------------------------------------------------
# §A.6 golden values
# ---------------------------------------------------------------------------
def test_source_url_golden():
    assert (
        prompts.make_source_url("1234567890")
        == "https://en.wikipedia.org/w/index.php?oldid=1234567890"
    )


def test_attribution_golden_and_quotes_are_safe():
    url = prompts.make_source_url("1234567890")
    assert prompts.make_attribution('A "quoted" title', "1234567890", "2026-11-06", url) == (
        'Wikipedia contributors. A "quoted" title. Wikipedia, The Free Encyclopedia. '
        "Revision 1234567890, retrieved 2026-11-06. "
        "https://en.wikipedia.org/w/index.php?oldid=1234567890. Licensed CC BY-SA 4.0."
    )


def test_prompt_hash_golden():
    assert (
        prompts.make_prompt_hash("Example paragraph text.")
        == "c237207cdd4bf724f91d9a2cd3a6206faa24c1f45e5bd4acc2bf08f2456363ca"
    )


def test_prompt_id_golden():
    assert (
        prompts.make_prompt_id(
            "1234567890",
            "c237207cdd4bf724f91d9a2cd3a6206faa24c1f45e5bd4acc2bf08f2456363ca",
        )
        == "pmt:9536fe5833e94a3a5bc9bb64"
    )


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------
def candidates(count: int = 20, pages: int = 20) -> list[dict]:
    return [
        {
            "prompt_id": f"pmt:{index:024x}",
            "page_id": index % pages,
            "n": index,
        }
        for index in range(count)
    ]


def test_select_is_order_independent():
    cands = candidates(40)
    first = prompts.select_prompts(cands, n=10, seed=SELECTION_SEED)
    shuffled = list(cands)
    random.Random(7).shuffle(shuffled)
    second = prompts.select_prompts(shuffled, n=10, seed=SELECTION_SEED)
    assert [c["prompt_id"] for c in first] == [c["prompt_id"] for c in second]
    assert len({c["prompt_id"] for c in first}) == 10


def test_select_rank_salt_changes_order():
    cands = candidates(40)
    a = prompts.select_prompts(cands, n=5, seed="seed-a")
    b = prompts.select_prompts(cands, n=5, seed="seed-b")
    assert [c["prompt_id"] for c in a] != [c["prompt_id"] for c in b]


def test_select_applies_page_cap():
    cands = [{"prompt_id": f"pmt:{i:024x}", "page_id": 1} for i in range(8)]
    chosen = prompts.select_prompts(cands, n=4, max_per_page=2)
    assert len(chosen) == 2


def test_select_raises_insufficient_pool_with_shortfall():
    with pytest.raises(prompts.InsufficientPoolError) as info:
        prompts.select_prompts(candidates(3), n=5)
    assert info.value.shortfall == 2
    assert "shortfall 2" in str(info.value)


# ---------------------------------------------------------------------------
# validate_manifest
# ---------------------------------------------------------------------------
def valid_rows() -> list[dict]:
    return [
        make_row(good_text("alpha"), revision_id="1234567890", page_id=1, title="Alpha"),
        make_row(good_text("beta"), revision_id="1234567891", page_id=2, title="Beta"),
    ]


def test_validate_manifest_accepts_valid_rows():
    assert prompts.validate_manifest(valid_rows(), expected=2) == []


def test_validate_manifest_catches_each_violation():
    rows = valid_rows()

    violations = prompts.validate_manifest(rows, expected=3)
    assert any("row count" in v for v in violations)

    broken = copy.deepcopy(rows)
    broken[1]["prompt_id"] = broken[0]["prompt_id"]
    assert any("duplicate prompt_id" in v for v in prompts.validate_manifest(broken, expected=2))

    broken = copy.deepcopy(rows)
    broken[1]["prompt_hash"] = broken[0]["prompt_hash"]
    assert any("duplicate prompt_hash" in v for v in prompts.validate_manifest(broken, expected=2))

    broken = copy.deepcopy(rows)
    broken[0]["prompt_id"] = "pmt:NOTHEX"
    assert any("invalid prompt_id" in v for v in prompts.validate_manifest(broken, expected=2))

    broken = copy.deepcopy(rows)
    broken[0]["prompt_hash"] = "deadbeef"
    assert any("invalid prompt_hash" in v for v in prompts.validate_manifest(broken, expected=2))

    broken = copy.deepcopy(rows)
    broken[0]["revision_id"] = "abcdef"
    violations = prompts.validate_manifest(broken, expected=2)
    assert any("non-numeric revision_id" in v for v in violations)
    assert any("revision_id does not match" in v for v in violations)

    broken = copy.deepcopy(rows)
    broken[0]["source_url"] = "https://en.wikipedia.org/w/index.php?oldid=999"
    assert any("source_url" in v and "oldid" in v for v in prompts.validate_manifest(broken, expected=2))

    broken = copy.deepcopy(rows)
    broken[0]["topic_category"] = "not_a_topic"
    assert any("not in taxonomy" in v for v in prompts.validate_manifest(broken, expected=2))

    broken = copy.deepcopy(rows)
    broken[0]["content_safety_status"] = "excluded_biography"
    assert any("content_safety_status" in v for v in prompts.validate_manifest(broken, expected=2))

    broken = copy.deepcopy(rows)
    broken[0]["script_latin"] = False
    assert any("script_latin" in v for v in prompts.validate_manifest(broken, expected=2))

    broken = copy.deepcopy(rows)
    broken[0]["words"] = MIN_WORDS - 1
    assert any("outside" in v for v in prompts.validate_manifest(broken, expected=2))

    broken = copy.deepcopy(rows)
    broken[0]["words"] = MAX_WORDS + 1
    assert any("outside" in v for v in prompts.validate_manifest(broken, expected=2))

    broken = copy.deepcopy(rows)
    broken[0]["prompt_text"] = good_text("changed")
    assert any("does not recompute" in v for v in prompts.validate_manifest(broken, expected=2))

    broken = copy.deepcopy(rows)
    broken[0]["source_licence"] = "MIT"
    assert any("source_licence" in v for v in prompts.validate_manifest(broken, expected=2))


# ---------------------------------------------------------------------------
# Manifest I/O
# ---------------------------------------------------------------------------
def test_write_manifest_is_sorted_stable_and_hashed(tmp_path):
    rows = list(reversed(valid_rows()))
    rows[0]["source_title"] = "Café"
    path = tmp_path / "manifest.jsonl"

    digest = prompts.write_manifest(path, rows)
    data = path.read_bytes()

    assert digest == hashlib.sha256(data).hexdigest()
    assert data.endswith(b"\n")
    lines = data.decode("utf-8").splitlines()
    assert len(lines) == 2
    decoded = [json.loads(line) for line in lines]
    assert [row["prompt_id"] for row in decoded] == sorted(row["prompt_id"] for row in decoded)
    assert "Café" in data.decode("utf-8")
    # sort_keys=True: each object is serialised with sorted keys.
    assert lines[0] == json.dumps(decoded[0], sort_keys=True, ensure_ascii=False)


def test_read_manifest_round_trips(tmp_path):
    rows = valid_rows()
    path = tmp_path / "manifest.jsonl"
    prompts.write_manifest(path, rows)
    assert prompts.read_manifest(path) == sorted(rows, key=lambda r: r["prompt_id"])
