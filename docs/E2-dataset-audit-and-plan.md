# E2 dataset audit, decontamination and experiment plan

**Date:** 2026-11-06. E1 is untouched: committed eval hash unchanged, split digest still
`932d20f1dca64c21537bed89739c633ace1b627c1a25f7d78c45a17915cf27a2`.

---

## 1. Licence audit

| Corpus | Licence | Evidence | Covers the actual text? |
|---|---|---|---|
| Ghostbuster | **CC BY 3.0** | `ghostbuster-data/LICENSE` | Yes |
| ai-text-detection-pile | **MIT** | HF card front-matter only | **Unverified.** MIT is the uploader's claim over a compilation of Reddit WritingPrompts, OpenAI Webtext, HC3, GPT-2/3/J and ChatGPT outputs. Each upstream has its own terms |
| AI-and-Human-Generated-Text | **MIT** | Upstream `panagiotisanagnostou/AI-GA/LICENSE` (GitHub API) + HF tag | **No.** Human abstracts come from **CORD-19** (PubMed/bioRxiv/medRxiv/WHO), which MIT cannot relicense |
| HC3 | CC BY-SA 4.0 | HF card | Yes (evaluation only) |
| RAID | MIT | HF card | Yes (evaluation only) |

**Position:** the pile is usable for non-commercial research with attribution, but its provenance is
a compilation and the MIT claim should not be repeated as if it covered every row. AI-GA's human
half is third-party text held under CORD-19 terms; fine to train on for internal research, not fine
to redistribute.

## 2. Provenance audit

### ai-text-detection-pile

| Property | Verified |
|---|---|
| Rows | **1,392,522** across 7 parquet shards (~1.5 GB) |
| Columns | `source`, `id`, `text` - **no generator, no domain, no provenance** |
| Sharding | **by class**: one shard is 198,932 rows, all `human` |
| Human sources (card) | Reddit WritingPrompts 570k, OpenAI Webtext 260k, HC3 58k, ivypanda essays |
| Generators (card) | GPT-2, GPT-3 (davinci), GPT-J, ChatGPT |

**Consequence:** per-row generator and domain attribution is impossible. Any E2 report on this corpus
can only cite the card's aggregate composition, not measured per-generator counts.

### AI-and-Human-Generated-Text

| Property | Verified |
|---|---|
| Rows | **28,662** (22,930 train + 5,732 test), balanced **14,331 / 14,331** |
| Columns | `title`, `abstract`, `label` |
| Human provenance | **CORD-19** |
| AI provenance | GPT-3, generated from the titles |
| Domain | biomedical abstracts; only **6.5%** mention COVID/SARS |
| Overlap with our evaluation sets | **zero** (exact and near) |

**Two weaknesses.** The generated class is templated - the six most common openings are
"the title of this article is" (84, a prompt echo), "this paper presents a novel approach" (69),
"this paper presents an analysis of" (66). A classifier could win on three phrases. And its human
half carries CORD-19 terms.

---

## 3. Decontamination results

Content-based, because the pile has no row-level provenance. Two stages: SHA-256 exact match after
normalisation, then MinHash (64 perms, 5-word shingles, 16x4 LSH bands) with **exact Jaccard
verification at >= 0.85**.

### Round 1 vs round 2 - a corrected measurement

Round 1 built its reference from our *canonical* records, which hold **8,000 of 85,449 HC3 rows -
9.4% coverage**. It reported 5,391 HC3 removals. That was not a contamination rate; it was an
artefact of the reference set.

| | Round 1 (8k HC3) | **Round 2 (full HC3)** |
|---|---:|---:|
| Reference records | 29,592 | **108,448** |
| Rows examined | 1,392,522 | 1,392,522 |
| Removed exact | 5,246 | **63,484** |
| Removed near (>=0.85) | 1,099 | **2,253** |
| Removed empty | 342 | 342 |
| **Total removed** | 6,687 | **66,079** |
| **Removal rate** | 0.48% | **4.75%** |
| Matched to HC3 | 5,391 | **64,783** |
| Matched to Ghostbuster | 954 | 954 |

**Round 2 is the measurement.** 64,783 HC3 rows is close to the card's own claim of ~85k
(58k human + 27k ChatGPT); the ~20k gap is consistent with rows that were truncated or edited
enough to fall below 0.85 Jaccard - which is a residual risk, not a clean bill of health.

### Final counts

| | |
|---|---:|
| Examined | 1,392,522 |
| Removed | 66,079 (4.75%) |
| **Kept** | **1,326,443** |
| Kept human | **982,502** |
| Kept AI | **343,941** |
| Corpus digest | **`fd012688e17cd95d064507ca17180cd093abc86ea4a5e0c5404adfb4f249253a`** |

Artefacts: `datasets/raw/aitextpile/decon_full/{decontamination.json,kept.parquet}` (1,337 MB, 27 row
groups). Parameters: shingle k=5, 64 permutations, 16x4 bands, Jaccard >= 0.85, NFKC + quote/dash
canonicalisation. Runtime 2,341.9 s at 594.6 rows/s.

### Residual contamination that dedup cannot catch

The first kept row is *"12 Years a Slave: An Analysis of the Film Essay"* - the same film Ghostbuster
uses. I checked 20,000 human rows for first-line overlap with Ghostbuster's essay prompts: **0 hits
at >= 0.60**, because those prompt files are meta-instructions, not topic titles. So the overlap is
**topical, not textual**: a 954 rows caught as near-duplicates, plus an unmeasured number of
same-topic-different-wording essays.

**Why this is tolerable:** it affects only `test_id`, the in-domain column we already discount
(0.999). HC3 and RAID are separate corpora and are unaffected.

---

## 4. E2-A / E2-B / E2-C compositions

To keep training inside a Colab session and the comparison interpretable, the pile is **capped at
20,000 rows, class-stratified 10,000 human / 10,000 AI**, sampled deterministically by row id.

| | E2-A | E2-B | E2-C |
|---|---|---|---|
| Ghostbuster | 22,100 | 22,100 | 22,100 |
| pile (decontaminated, capped) | 20,000 | - | 20,000 |
| AI-GA | - | 22,930 | 22,930 |
| **Total records** | **42,100** | **45,030** | **65,030** |
| Human : machine (approx) | 11,617 : 19,533 | 13,082 : 16,068 | 21,617 : 25,533 |
| Class ratio | 1.7 : 1 | 1.2 : 1 | 1.2 : 1 |
| Generators | GPT-3, Claude + GPT-2/3/J, ChatGPT | GPT-3, Claude + GPT-3 | all of the above |
| Domains | essays, news, Reuters + Reddit, webtext | essays, news, Reuters + biomedical | all |
| Licence | CC BY 3.0 + MIT(claim) | CC BY 3.0 + MIT(CORD-19 caveat) | mixed |

Numbers are pre-split. Exact per-split counts and the **training manifest digest** for each
configuration are produced at build time by the existing pipeline; they do **not** exist yet and must
not be quoted until generated. E1's digest stays `932d20f1...` and E2's will differ - the two are
comparable only on the evaluation sets, never by training digest.

**E2-B is the weakest design.** AI-GA is templated, so a win there would likely measure the template
rather than transfer.

---

## 5. Evaluation leakage assessment

| Risk | Status |
|---|---|
| HC3 rows inside the pile | **Addressed** - 64,783 removed against the full corpus |
| Ghostbuster duplicates inside the pile | 954 removed |
| RAID rows inside the pile | **Only against our 2,000-row evaluation sample.** Contamination of other RAID rows would not be detected - but also would not affect our measurement, because our evaluation set is exactly that sample |
| Truncated/edited HC3 rows below 0.85 Jaccard | **Residual, unquantified** |
| Same-topic Ghostbuster essays | **Residual**, affects `test_id` only |
| Evaluation sets used for tuning | None - HC3 and RAID are quarantine-only; the existing gate enforces it |

**Verdict:** safe to train on, with the two residuals documented. The HC3 residual is the one that
could still inflate the HC3 column; it should be reported alongside any E2 HC3 result.

---

## 6. Training configuration

Identical to E1 for a controlled comparison: `distilbert-base-cased`, 512 tokens, 3 epochs, lr 5e-5,
batch 16 x grad-accum 2, fp16, class-weighted loss recomputed per configuration. Threshold from
`val` at a 5% human-FPR budget, frozen before test. Same evaluation harness
(`ml/train/evaluate_encoder.py`).

## 7. Expected GPU requirement (ESTIMATE from E1)

E1: 2,088 steps, ~40 min on a T4 at 512 tokens, batch 16.

| Config | Train rows (approx) | Steps | Estimated T4 time |
|---|---:|---:|---|
| E1 (measured) | 11,150 | 2,088 | ~40 min |
| E2-A | ~31,000 | ~5,800 | **~1.9 h** |
| E2-B | ~33,000 | ~6,200 | **~2.0 h** |
| E2-C | ~47,000 | ~8,800 | **~2.8 h** |

Plus ~6 min evaluation each. E2-C sits at roughly 60% of a 4 h 40 m Colab session - viable, but it
should be run with checkpointing to Drive as E1 was.

## 8. Outstanding risks

1. **HC3 residual**: rows edited below the Jaccard threshold would still leak. Unquantified.
2. **Pile provenance**: MIT is the uploader's claim over a compilation; no per-row rights record.
3. **No per-generator labels** in the pile, so E2 cannot report generator-specific training
   composition - only card-level claims.
4. **AI-GA template shortcut**: its AI class may be separable on three phrases.
5. **Digest comparability**: E2 digests will differ from E1's; only evaluation sets are shared.
6. **Domain confound in the pile**: Reddit fiction prompts vs instructed prose.

## 9. Recommendation

**Run E2-A first.**

It adds the most domain and generator breadth per unit of risk, its licence position is no worse than
E2-C, and it avoids AI-GA's template shortcut entirely. If E2-A fails to move RAID, E2-C will not
rescue it - and that negative result is the more valuable finding.

**Do not run E2-B** except as an ablation after E2-A, and if you do, strip the templated openers
first and re-run the artefact audit.

The decontamination gate is met. The training gate is **not**: the E2 training manifest and its
digest do not exist yet, and the audit must pass on the new combined corpus before any run.
