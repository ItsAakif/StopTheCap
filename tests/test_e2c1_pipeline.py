"""End-to-end prompt pipeline with a synthetic corpus and a fake Wikipedia client."""

from __future__ import annotations

import hashlib
import json

from ml.e2c1 import build_prompts
from ml.e2c1.build_prompts import STAGE_NAMES, main, read_input, run_build
from ml.e2c1.constants import MAX_WORDS, TOPIC_TAXONOMY
from ml.e2c1.prompts import make_source_url, read_manifest, validate_manifest
from ml.e2c1.wikipedia import PageSnapshot, WikipediaError, WikipediaOfflineError


def word_text(seed: str, n: int = 60) -> str:
    return " ".join(f"{seed}{index}" for index in range(n))


class FixtureBuilder:
    def __init__(self) -> None:
        self.rows: list[dict[str, str]] = []
        self.fixtures: dict[str, dict | None] = {}
        self.revision = 100000
        self.page = 0

    def add(
        self,
        title: str,
        text: str,
        *,
        page_id: int | None = None,
        categories: tuple[str, ...] = (),
        pageprops: dict | None = None,
        extract: str | None = None,
        missing: bool = False,
        fixture: bool = True,
    ) -> None:
        self.rows.append({"page_title": title, "human_text": text})
        if not fixture:
            return
        if missing:
            self.fixtures[title] = None
            return
        self.revision += 1
        self.page += 1
        self.fixtures[title] = {
            "revid": self.revision,
            "page_id": page_id if page_id is not None else self.page,
            "extract": text if extract is None else extract,
            "categories": categories,
            "pageprops": pageprops or {},
        }


def synthetic_corpus() -> tuple[list[dict[str, str]], dict[str, dict | None]]:
    builder = FixtureBuilder()

    # Stage 1: exact duplicate. "Dup A" is the lexicographically smaller title.
    duplicate_text = word_text("dup")
    builder.add("Dup A", duplicate_text)
    builder.add("Dup B", duplicate_text)

    # Stages 2-5: offline filters.
    builder.add("Short Page", "not enough words here")
    builder.add("Long Page", word_text("long", MAX_WORDS + 1))
    builder.add("Japanese Page", " ".join(["日本語"] * 60))
    builder.add("Murder of a Person", word_text("titleblock"))

    # Stages 6-9.
    builder.add("Missing Page", word_text("missing"), missing=True)
    builder.add("Error Page", word_text("error"), fixture=False)
    builder.add("No Extract Page", word_text("noextract"), extract="")
    builder.add("Mismatch Page", word_text("mismatch"), extract=word_text("other"))

    # Stages 10-15.
    builder.add(
        "Mercury",
        word_text("disamb"),
        pageprops={"disambiguation": ""},
    )
    builder.add("List of rivers", word_text("list"))
    builder.add(
        "A Biographical Subject",
        word_text("bio"),
        categories=("Category:1990 births",),
    )
    builder.add(
        "Notable Person",
        word_text("biomulti") + " massacre",
        categories=("Category:Living people",),
    )
    builder.add("Neutral Event", word_text("block") + " massacre")
    quoted = (
        f'"{("quoted words " * 6).strip()}" '
        f'{word_text("middle", 30)} '
        f'"{("tail span " * 7).strip()}"'
    )
    builder.add("Quoted Thing", quoted)
    builder.add("Boilerplate Article", "This article is about " + word_text("boiler", 55))

    # Survivors: 12 unique pages + 6 prompts sharing one page (page cap).
    for index in range(12):
        builder.add(f"Good Topic {chr(ord('A') + index)}", word_text(f"good{index}"))
    for index in range(6):
        builder.add(
            f"Cap {index + 1}",
            word_text(f"cap{index}"),
            page_id=900,
        )

    return builder.rows, builder.fixtures


class FakeClient:
    def __init__(self, fixtures: dict[str, dict | None]) -> None:
        self.fixtures = fixtures
        self.calls: list[list[str]] = []

    def fetch_pages(self, titles):
        self.calls.append(list(titles))
        result = {}
        for title in titles:
            if title == "Error Page":
                raise WikipediaError("simulated API failure", titles=[title])
            fixture = self.fixtures.get(title)
            if fixture is None:
                result[title] = None
                continue
            revision_id = str(fixture["revid"])
            result[title] = PageSnapshot(
                page_id=fixture["page_id"],
                title=title,
                revision_id=revision_id,
                revision_timestamp="2020-01-01T00:00:00Z",
                url=make_source_url(revision_id),
                extract=fixture["extract"],
                categories=tuple(fixture["categories"]),
                pageprops=dict(fixture["pageprops"]),
                redirect_from=None,
            )
        return result


def write_input(path, rows) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_offline_stage_counts_match_the_expected_shape():
    rows, _fixtures = synthetic_corpus()
    survivors, stages = build_prompts.run_offline_stages(rows)
    by_name = {stage["stage"]: stage for stage in stages}

    assert [stage["stage"] for stage in stages] == list(STAGE_NAMES[:5])
    assert by_name["exact_duplicate"]["removed"] == 1
    assert by_name["length_short"]["removed"] == 1
    assert by_name["length_long"]["removed"] == 1
    assert by_name["non_latin"]["removed"] == 1
    assert by_name["title_blocklist"]["removed"] == 1
    assert len(survivors) == len(rows) - 5


def test_pipeline_builds_manifest_and_balanced_audit(tmp_path):
    rows, fixtures = synthetic_corpus()
    input_path = tmp_path / "input.jsonl"
    out_path = tmp_path / "e2_c1_prompts.jsonl"
    audit_path = tmp_path / "e2_c1_prompt_audit.json"
    write_input(input_path, rows)

    audit, code = run_build(
        input_path=input_path,
        out_path=out_path,
        audit_path=audit_path,
        n=12,
        client=FakeClient(fixtures),
        retrieved_at="2026-11-06",
    )

    assert code == 0
    assert audit["status"] == "ok"
    assert out_path.exists() and audit_path.exists()

    manifest = read_manifest(out_path)
    assert len(manifest) == 12
    assert validate_manifest(manifest, expected=12) == []
    assert [row["prompt_id"] for row in manifest] == sorted(
        row["prompt_id"] for row in manifest
    )

    required = {
        "prompt_id",
        "source_title",
        "source_url",
        "revision_id",
        "retrieved_at",
        "source_licence",
        "attribution",
        "prompt_text",
        "prompt_hash",
        "topic_category",
        "content_safety_status",
        "page_id",
        "words",
        "script_latin",
        "topic_rule",
    }
    for row in manifest:
        assert required <= set(row)
        assert row["source_url"] == make_source_url(row["revision_id"])
        assert row["source_url"].endswith(f"oldid={row['revision_id']}")
        assert row["revision_id"].isdigit()
        assert row["content_safety_status"] == "ok"
        assert row["script_latin"] is True
        assert row["source_licence"] == "CC-BY-SA-4.0"
        assert row["retrieved_at"] == "2026-11-06"

    # Every stage is recorded, in order, and its arithmetic is self-consistent.
    assert [stage["stage"] for stage in audit["stages"]] == list(STAGE_NAMES)
    for stage in audit["stages"]:
        assert stage["in"] - stage["removed"] == stage["out"]
        assert stage["removed"] > 0, f"reason not exercised: {stage['stage']}"

    # Reconciliation: input - sum(removed) == selected.
    removed = sum(stage["removed"] for stage in audit["stages"])
    assert audit["reconciliation"]["input"] - removed == len(manifest)
    assert audit["reconciliation"]["selected"] == len(manifest)
    assert audit["reconciliation"]["balanced"] is True

    # Every exclusion reason is represented.
    for key, count in audit["exclusions"].items():
        assert count > 0, f"exclusion not exercised: {key}"
    assert audit["multi_reason"].get("biography+blocklist", 0) >= 1

    assert sum(audit["by_topic"].values()) == len(manifest)
    assert set(audit["by_topic"]) == set(TOPIC_TAXONOMY)
    assert audit["selection"]["selected"] == len(manifest)
    assert audit["selection"]["page_cap_removed"] > 0
    assert audit["selection"]["rank_cutoff_removed"] > 0

    assert audit["output"]["sha256"] == hashlib.sha256(out_path.read_bytes()).hexdigest()
    assert audit["output"]["count"] == len(manifest)

    # Only the requested e2_c1_* artefacts were written.
    written = {path.name for path in tmp_path.iterdir()} - {"input.jsonl"}
    assert written == {"e2_c1_prompts.jsonl", "e2_c1_prompt_audit.json"}


def test_pipeline_is_order_independent(tmp_path):
    rows, fixtures = synthetic_corpus()
    first_input = tmp_path / "first.jsonl"
    second_input = tmp_path / "second.jsonl"
    write_input(first_input, rows)
    write_input(second_input, list(reversed(rows)))

    first_out = tmp_path / "e2_c1_first.jsonl"
    second_out = tmp_path / "e2_c1_second.jsonl"
    first_audit, first_code = run_build(
        input_path=first_input,
        out_path=first_out,
        audit_path=tmp_path / "e2_c1_first_audit.json",
        n=12,
        client=FakeClient(fixtures),
    )
    second_audit, second_code = run_build(
        input_path=second_input,
        out_path=second_out,
        audit_path=tmp_path / "e2_c1_second_audit.json",
        n=12,
        client=FakeClient(fixtures),
    )

    assert first_code == second_code == 0
    assert first_out.read_bytes() == second_out.read_bytes()
    assert first_audit["stages"] == second_audit["stages"]
    assert first_audit["exclusions"] == second_audit["exclusions"]


def test_insufficient_pool_writes_audit_but_no_manifest(tmp_path):
    builder = FixtureBuilder()
    for index in range(3):
        builder.add(f"Good {index}", word_text(f"good{index}"))
    input_path = tmp_path / "input.jsonl"
    out_path = tmp_path / "e2_c1_prompts.jsonl"
    audit_path = tmp_path / "e2_c1_prompt_audit.json"
    write_input(input_path, builder.rows)

    audit, code = run_build(
        input_path=input_path,
        out_path=out_path,
        audit_path=audit_path,
        n=5,
        client=FakeClient(builder.fixtures),
    )

    assert code == 3
    assert audit["status"] == "insufficient_pool"
    assert audit["shortfall"] == 2
    assert audit["selection"]["selected"] == 0
    assert not out_path.exists()
    assert audit_path.exists()


def test_dry_run_prints_stages_and_writes_nothing(tmp_path, monkeypatch, capsys):
    class ExplodingClient:
        def __init__(self, **kwargs):
            raise AssertionError("dry run must not construct a network client")

    monkeypatch.setattr(build_prompts, "WikipediaClient", ExplodingClient)
    rows, _fixtures = synthetic_corpus()
    input_path = tmp_path / "input.jsonl"
    out_path = tmp_path / "e2_c1_prompts.jsonl"
    audit_path = tmp_path / "e2_c1_prompt_audit.json"
    write_input(input_path, rows)

    code = main(
        [
            "--dry-run",
            "--input",
            str(input_path),
            "--out",
            str(out_path),
            "--audit",
            str(audit_path),
        ]
    )

    output = capsys.readouterr().out
    assert code == 0
    for stage in ("exact_duplicate", "length_short", "non_latin", "title_blocklist"):
        assert stage in output
    assert "api_missing" not in output
    assert not out_path.exists()
    assert not audit_path.exists()


def test_offline_cache_miss_fails_without_manifest(tmp_path, monkeypatch):
    class OfflineClient:
        def __init__(self, **kwargs):
            pass

        def fetch_pages(self, titles):
            raise WikipediaOfflineError("offline cache miss")

    monkeypatch.setattr(build_prompts, "WikipediaClient", OfflineClient)
    builder = FixtureBuilder()
    builder.add("Good Page", word_text("good"))
    input_path = tmp_path / "input.jsonl"
    out_path = tmp_path / "e2_c1_prompts.jsonl"
    audit_path = tmp_path / "e2_c1_prompt_audit.json"
    write_input(input_path, builder.rows)

    code = main(
        [
            "--offline",
            "--input",
            str(input_path),
            "--out",
            str(out_path),
            "--audit",
            str(audit_path),
        ]
    )

    assert code == 2
    assert not out_path.exists()


def test_read_input_limit(tmp_path):
    rows, _fixtures = synthetic_corpus()
    path = tmp_path / "input.jsonl"
    write_input(path, rows)
    assert len(read_input(path, limit=3)) == 3
    assert [row["page_title"] for row in read_input(path, limit=3)] == [
        row["page_title"] for row in rows[:3]
    ]
