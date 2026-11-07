# E2-C1 prompt provenance

**Built:** 2026-11-06
**Source:** English Wikipedia, fetched through the MediaWiki API
**Manifest:** `datasets/manifests/e2_c1_prompts.jsonl` (5,000 rows)
**Manifest SHA-256:** `0c3c8b7a72196e2fc2d5454869a66bfcac8b785c19f8cc619c59d95892b6788a`
**Audit:** `datasets/manifests/e2_c1_prompt_audit.json`
**Split manifest:** `datasets/manifests/e2_c1_splits.json`
**Split digest:** `adfe9df5900dac1342ff241ef1eff7ed5cf933cdcc728dfe4e69e0c00d94fb97`

---

## 1. Why this was rebuilt

The first design reused `research/_evidence/wikipedia.jsonl`, a scraped corpus of 9,970
human/AI pairs. Measurement killed it:

- Only **3.5%** of its stored paragraphs could be verified against a current revision.
- **35.6%** of paragraphs no longer appear in the page's current revision at all, so any
  `revision_id` attached to them would have been false attribution.
- Full plaintext extracts are served **one page per request**, so verifying 7,370 candidates
  would have taken roughly eight hours.

The prompt inventory is therefore built from **wikitext at a pinned revision**. The paragraph and
the revision id come from the same API response, so a revision mismatch cannot occur by
construction.

## 2. Source and retrieval

| Property | Value |
|---|---|
| Site | `en.wikipedia.org` |
| Interface | `action=query`, `prop=revisions`, `rvprop=ids|timestamp|content`, `rvslots=main` |
| Enumeration | `list=allpages`, namespace 0, non-redirects, 50 titles per request |
| Fetch batch | 50 titles per request |
| Seed starts | `0`-`9` replaced by `0`, then `A`-`Z`, then `Ba`, `Ca`, `Da`, `Fa`, `Ha`, `Ka`, `Ma`, `Na`, `Pa`, `Sa`, `Ta`, `Wa` |
| Pages fetched | **5,088** |
| API requests | **331** (15.37 pages/request) |
| Rate-limit pauses | 47 (`Retry-After` honoured, global pause) |
| Wall clock | ~28 minutes |
| `retrieved_at` | `2026-11-06` (UTC date, recorded on every row) |

Every request carried an identifying `User-Agent` and was paced. The 47 rate-limit responses were
handled by pausing all workers for the server-supplied `Retry-After`, not by retrying immediately.

## 3. What a prompt actually is

`prompt_text` is the **plain-text form of one paragraph** taken from the page's wikitext at
revision `revision_id`, after a deterministic markup strip:

1. remove HTML comments;
2. remove `<ref>…</ref>` and self-closing refs;
3. remove tables;
4. capture and remove `[[Category:…]]`, `[[File:…]]`, `[[Image:…]]`;
5. remove templates iteratively, including nested infoboxes;
6. unwrap `[[Target|Label]]` to `Label` and `[[Target]]` to `Target`;
7. strip bold/italic markers and external-link syntax;
8. split into paragraphs on blank lines;
9. reject any paragraph still containing `[[`, `]]`, `{{`, `}}`, `<ref`, `|
`, or `~~~`;
10. reject list-like fragments and paragraphs without sentence-ending punctuation.

This is a **derivation, not a byte copy of rendered HTML**. `prompt_hash` is the SHA-256 of the
stored plain text, so the hash pins exactly what was used. The revision id pins the source text
from which it was derived. `source_url` is the permanent `oldid` link.

| Field | Definition |
|---|---|
| `prompt_id` | `pmt:` + first 24 hex of `sha256(revision_id + U+001F + prompt_hash)` |
| `prompt_hash` | `sha256(prompt_text)` |
| `family_id` | `fam:` + first 16 hex of `sha256(min(member prompt_id))` |
| `source_licence` | `CC-BY-SA-4.0` (constant) |
| `source_url` | `https://en.wikipedia.org/w/index.php?oldid=<revision_id>` |

## 4. Selection funnel

Every removal is counted. Reconciliation: `38,216 − 33,216 = 5,000`.

| Stage | Considered | Removed | Remaining |
|---|---:|---:|---:|
| paragraphs parsed from 5,088 pages | 38,216 | 0 | 38,216 |
| `parser_reject` (residual markup, list fragments, no terminal punctuation) | 38,216 | 11,018 | 27,198 |
| `no_extract` (page had no usable revision content) | 27,198 | 202 | 26,996 |
| `length` (<40 or >300 words) | 26,996 | 10,628 | 16,368 |
| `non_latin` (Latin ratio < 0.95) | 16,368 | 23 | 16,345 |
| `safety:excluded_biography` | 16,345 | 9,280 | 7,065 |
| `safety:excluded_blocklist` | 7,065 | 109 | 6,956 |
| `safety:excluded_disambiguation` | 6,956 | 5 | 6,951 |
| `safety:excluded_list` | 6,951 | 61 | 6,890 |
| `safety:excluded_quote_heavy` | 6,890 | 388 | 6,502 |
| `per_page_cap` (max 3 prompts per page) | 6,502 | 1,501 | 5,001 |
| `rank_cutoff` (deterministic hash rank) | 5,001 | 1 | **5,000** |

The biography rule removes 24.3% of otherwise-valid paragraphs. That is deliberate: the cost of
losing a biography is lower than the cost of putting personal data into a training corpus. It is
the largest single exclusion and the main reason the build needs 5,088 pages to produce 5,000
prompts.

## 5. Near-duplicate policy

The task requires both "deduplicate exact and near-duplicate prompts" and "near-duplicate prompt
families must remain in one partition". Those pull in opposite directions if near-duplicates are
deleted: a deleted pair cannot form a family, and the family test becomes vacuous.

Resolution, recorded explicitly:

- **Exact duplicates are removed.** Uniqueness is enforced on `prompt_hash` and `prompt_id`;
  `validate_manifest` returned no violations, and both columns have 5,000 distinct values.
- **Near-duplicates at Jaccard >= 0.85 are clustered, not deleted.** They are joined into families
  together with all prompts from the same `page_id`.
- **No family may cross a partition**, so a near-duplicate pair can never sit in train and test.

| Family property | Value |
|---|---|
| Families | **2,302** (across 2,306 pages) |
| Size 1 | 740 |
| Size 2 | 436 |
| Size 3 | 1,122 |
| Size 4 | 1 |
| Size 6 | 3 |
| Max size | 6 |

Family ids are `sha256(min(member prompt_id))`, so they are order-independent for a frozen
manifest. They are **not** stable if members are added or removed; an incremental rebuild must be
treated as a new manifest version.

## 6. Topic distribution

The taxonomy is a fixed, ordered keyword table over titles and categories (10 values, first match
wins). It is a convenience stratification, not a curated subject classification.

| Topic | Prompts | Share |
|---|---:|---:|
| `other` | 2,237 | 44.7% |
| `arts_culture` | 1,022 | 20.4% |
| `sports` | 576 | 11.5% |
| `society_politics` | 470 | 9.4% |
| `business_economics` | 335 | 6.7% |
| `history` | 140 | 2.8% |
| `stem` | 124 | 2.5% |
| `geography_places` | 61 | 1.2% |
| `medicine_health` | 23 | 0.5% |
| `religion_philosophy` | 12 | 0.2% |

**Known skew.** `other` dominates and `medicine_health`/`stem` are thin. Two causes: the
alphabetical enumeration over-samples pop-culture and sport articles, and the keyword table is
coarse. This is a property of the prompt inventory, not of the generator. It is recorded rather
than hidden, and it bounds any claim that E2-C1 is topic-balanced.
## 7. Prompt-level split (assigned before generation)

Assignment happens **before any generation**. A prompt is the unit: every generation from that
prompt lands in the same partition regardless of generator or decoding setting.

| Property | Value |
|---|---|
| Algorithm | family-atomic, topic-stratified, hash-ranked greedy fill + deterministic repair |
| Salt | `e2c1-prompt-split-2026-11-06` |
| Fractions | 80 / 10 / 10 |
| Tolerance | 10 prompts per partition |
| Partitions | `train`, `val`, `test_holdout` - **never `test_id`** |
| Split digest | `adfe9df5900dac1342ff241ef1eff7ed5cf933cdcc728dfe4e69e0c00d94fb97` |
| Prompt manifest SHA-256 | `0c3c8b7a72196e2fc2d5454869a66bfcac8b785c19f8cc619c59d95892b6788a` |

| Partition | Prompts | Planned samples |
|---|---:|---:|
| `train` | 4,004 | 16,016 |
| `val` | 504 | 2,016 |
| `test_holdout` | 492 | 1,968 |
| **Total** | **5,000** | **20,000** |

Planned samples = prompts x 2 generators x 2 decoding settings. Every partition x generator x
decoding cell is non-zero and recorded explicitly in the split manifest, including the zeros in the
full topic matrix.

`test_id` is deliberately unused. `ml/data/splits.py:build_split_plan` maps a 10% bucket to
`Split.TEST_ID`, which is E1's frozen in-domain test set. Reusing it for E2-C1 would contaminate the
only in-domain benchmark the project has. E2-C1 therefore owns a separate manifest and a separate
digest; `COMBINED.yaml` is untouched.

## 8. Licensing layers

These are four different things and are recorded separately.

| Layer | What it is | Governing terms | Redistribution |
|---|---|---|---|
| Source prompt | Wikipedia paragraph, plaintext, at a pinned revision | **CC BY-SA 4.0**; attribution required; share-alike on adaptations | Permitted with attribution; the dataset stays private, so share-alike is not triggered |
| Model weights | Mistral-7B-Instruct-v0.2 / Llama-3.2-3B-Instruct | Apache-2.0 / Llama 3.2 Community License | Weights are not redistributed |
| Generated output | Text produced by those weights | **No licence granted by either weight licence**; copyright status uncertain | **Not redistributed.** Private research use only; `outputs_licensed: false` on every record |
| Derived record | ids, hashes, partition, topic, annotations | Our own; no third-party licence attaches | Internal only |

The manifest carries attribution on every row, which is what makes committing it to the private
repository compatible with CC BY-SA 4.0. Nothing in the repository claims a blanket licence over
the generated corpus.

Attribution template:

```
Wikipedia contributors. <title>. Wikipedia, The Free Encyclopedia.
Revision <revision_id>, retrieved <retrieved_at>. <oldid URL>. Licensed CC BY-SA 4.0.
```

## 9. Limitations and residual risks

1. **Topic skew** (section 6): `other` 44.7%, `medicine_health` 0.5%. Any per-topic claim must be
   read against this distribution.
2. **Safety filtering is best-effort, not moderation.** It excludes biographies, disambiguation
   pages, list pages, quote-heavy paragraphs and a keyword blocklist. It does not guarantee that
   every surviving paragraph is free of sensitive material.
3. **Prompts are derived plaintext, not rendered HTML.** The markup strip is deterministic and
   tested, but it is a transformation. `prompt_hash` pins the stored text; `revision_id` pins the
   source.
4. **`retrieved_at` is a date, not a timestamp.** Revisions are immutable, so the revision id is
   the stronger pin.
5. **Near-duplicates are clustered, not deleted** (section 5). They consume prompt budget but
   cannot cross partitions.
6. **Llama is a family-level proxy only.** RAID's labels are `llama-chat`, `mistral` and
   `mistral-chat`; RAID does not contain Llama-3.2-3B-Instruct or Mistral-7B-Instruct-v0.2 outputs.
   No exact-generator parity is claimed.
7. **Mistral generation will be 4-bit NF4**, not the published fp16 checkpoint, because 7B fp16
   does not fit a 15.64 GB T4 alongside activations and KV cache. Every Mistral record carries
   `deviation_from_published_checkpoint: true`.
8. **The generated corpus is not decontaminated yet.** That is a pre-training gate: run
   `build_reference_index` and `classify` from `ml/data/decontaminate.py` over the generated JSONL
   against the frozen Ghostbuster, HC3 and RAID references, and record the digest before training.
   The prompts are source text, not training samples; the generated text is what must be checked.

## 10. Reproduction

```powershell
# rebuild the inventory (network; paced; honours Retry-After)
python -m ml.e2c1.build_prompts_wikitext --n 5000 --workers 2 --min-interval 1.2 --deadline-seconds 1800 --sections all --max-per-page 3 --out datasets/manifests/e2_c1_prompts.jsonl --audit datasets/manifests/e2_c1_prompt_audit.json

# rebuild the split (offline)
python -m ml.e2c1.build_splits --prompts datasets/manifests/e2_c1_prompts.jsonl --out datasets/manifests/e2_c1_splits.json --config ml/e2c1/generation_config.json

# verify: must print [] and 5000
python -c "from pathlib import Path; from ml.e2c1.prompts import read_manifest, validate_manifest; rows = read_manifest(Path('datasets/manifests/e2_c1_prompts.jsonl')); print(validate_manifest(rows)); print(len(rows))"
```

Expected manifest hash: `0c3c8b7a72196e2fc2d5454869a66bfcac8b785c19f8cc619c59d95892b6788a`.

Rebuilding is not bit-reproducible against a live wiki: Wikipedia changes, the enumeration seed
walk may return different pages, and the rate limiter changes timing. The committed manifest and
its digest are the frozen artefact; a rebuild is a new version and must be treated as such.
