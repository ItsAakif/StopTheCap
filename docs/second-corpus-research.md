# Second training corpus research (for E2)

**Date:** 2026-11-05. Candidate metadata read from Hugging Face dataset cards and the Hub API.
Nothing downloaded.

**Purpose:** E1 transfers poorly across corpora (RAID AUC 0.532, CI includes chance). E2 asks whether
adding an independently sourced corpus improves that. This document shortlists candidates; the
experiment design is in `E2-design.md`.

---

## Requirements

| # | Requirement |
|---|---|
| R1 | Permissive or clearly documented licence for research **and** training |
| R2 | Both classes present, with enough human examples to evaluate honestly (>= 1,000) |
| R3 | Generators or domains that Ghostbuster does not already cover |
| R4 | Documented provenance, so contamination can be detected |
| R5 | No overlap with HC3 or RAID, or overlap that can be removed |
| R6 | Bounded download |

---

## Shortlist

### 1. `artem9k/ai-text-detection-pile` - strongest candidate, with a serious caveat

| | |
|---|---|
| Licence | **MIT** (card front-matter) |
| Size | ~1.39M rows: ~990k human, ~340k AI. 7 parquet shards, **~980 MB** |
| Human sources | Reddit WritingPrompts (570k), OpenAI Webtext (260k), **HC3 human responses (58k)**, ivypanda essays |
| Generators | **GPT-2, GPT-3 (davinci), GPT-J, ChatGPT** |
| Domains | Reddit fiction prompts, web text, essays, QA |
| Ghostbuster overlap | Low for Ghostbuster itself (different corpora) |
| **Eval overlap** | **Contains 58k HC3 human + 27k HC3 ChatGPT rows** |

**Caveat, and it is a real one:** the dataset includes HC3, which is one of our *independent
evaluation* sets. Training on it would silently contaminate the HC3 column of every future result.

Mitigation is possible but not free: the dataset has only `text` and `source` columns, no row-level
provenance. Decontamination must therefore be **content-based** - MinHash near-duplicate removal
against our committed HC3 canonical records - and it removes what it can find, not what it cannot.

**Value:** adds GPT-J and GPT-2 (new generator families), and Reddit/webtext/essay domains that
Ghostbuster lacks. All generators are 2019-2023 vintage, so it does **not** address the models RAID
actually tests.

### 2. `Ateeqq/AI-and-Human-Generated-Text` - clean, narrow

| | |
|---|---|
| Licence | **MIT** |
| Size | **28,662 rows**, balanced 14,331 / 14,331 |
| Human source | Original academic abstracts (AI-GA project) |
| Generator | **GPT-3 only** |
| Domain | Academic abstracts |
| Eval overlap | None identified |

**Value:** a clean, small, permissive corpus in a domain Ghostbuster does not cover (abstracts).
**Limitation:** a single generator, a single domain, and quite short texts. Unlikely to move RAID
much on its own.

### 3. `SJTU-CL/ArguGPT` - generator diversity, licence ambiguity

| | |
|---|---|
| Licence | **`cc`** - the variant is not specified on the card |
| Size | small (1K-10K); 4,708 valid machine essays reported in the README |
| Generators | **7 GPT-family models**: gpt2-xl, text-babbage-001, text-curie-001, text-davinci-001/002/003, gpt-3.5-turbo |
| Domain | Argumentative student essays (human side from TOEFL/essay corpora) |
| Eval overlap | None identified |

**Blocked by policy:** our rule is that an ambiguous licence is treated as *not permitted*. `cc`
without a version is ambiguous. This would be a good corpus if the variant were pinned to CC BY or
CC BY-SA.

### 4. `M4GT-Bench` / `SemEval-2024 Task 8` - excluded

No licence file in either repository; the Hub mirrors carry uploader-chosen tags. Already recorded as
excluded in `DATASET_REGISTRY.md`.

### 5. `LLM-DetectAIve` - not located

The LLM-DetectAIve paper (EMNLP 2024 demo) describes a 4-class corpus with modern generators built on
M4GT-Bench. **I could not find a canonical Hub dataset id** for it under the names tried; searching
returned nothing. Marked BLOCKED rather than guessed - and since it extends M4GT-Bench, its
inheritance of that unlicensed base would need checking anyway.

---

## The gap that matters most

RAID's generator set is: gpt2, gpt3, gpt4, chatgpt, **llama-chat, mistral, mistral-chat, mpt,
mpt-chat, cohere, cohere-chat**.

Ghostbuster covers GPT-3-davinci and Claude. None of the shortlisted corpora contains **Llama,
Mistral, MPT or Cohere** output. So no candidate on this list can teach the model the families that
RAID actually tests - which is the most plausible explanation for the 0.532.

**The only routes to those generators are:**

1. **Generate them ourselves** with Apache-2.0 models (Mistral 7B and Qwen3.5-4B; Llama 3.2 was
   considered but is **not** Apache-2.0, and its access is pending) on Colab -
   licence-clean, exact control, but 10-20k generations of compute.
2. Use RAID itself for training - **rejected**: it destroys RAID's value as an independent benchmark.

Route 1 is the honest answer to the generator-coverage problem. It is a generation project, not a
download, so it belongs in E3 rather than E2.

---

## Recommendation

**E2 composition: Ghostbuster + decontaminated `ai-text-detection-pile` + `Ateeqq` abstracts.**

Rationale: MIT on both, adds four domains (Reddit, webtext, essays, abstracts) and three generators
(GPT-2, GPT-J, ChatGPT) that Ghostbuster lacks, and stays within a ~1 GB download.

**Expected outcome, stated in advance:** a modest RAID improvement at best. The corpora add *old*
generators, not the ones RAID tests. If RAID stays near chance, that is a clean falsification of the
"more data fixes transfer" hypothesis, which is worth knowing before spending E3 on generation.

**Explicitly not doing:** training on ArguGPT (ambiguous licence), M4GT-Bench (no licence), HC3 or
RAID (evaluation sets).

---

## Unresolved

| Item | Status |
|---|---|
| Whether `ai-text-detection-pile` per-row provenance exists anywhere | Unknown - would remove the need for content-based decontamination |
| Exact CC variant for ArguGPT | Unread |
| Canonical id for LLM-DetectAIve | Not found |
| Whether the pile's Reddit fiction prompts confound human/AI with genre | Cannot tell before audit |

---

# Audit results (2026-11-05)

## `artem9k/ai-text-detection-pile`

**Structure, verified by downloading all 7 shards:**

| Property | Value |
|---|---|
| Rows | **1,392,522** (matches the card's ~1.39M) |
| Columns | `source`, `id`, `text` - **no generator, no domain, no provenance** |
| Sharding | **by class**: shard 00002 is 198,932 rows, **100% human** |
| Size on disk | ~1.5 GB parquet |

**Contamination, measured on a 3,001-row timing sample from the all-human shard:**

| Reason | Count | Share |
|---|---:|---:|
| Exact match vs HC3 | **230** | 7.7% of sampled rows |
| Near-duplicate | 0 | - |
| Total removed | 230 | 7.7% |

Every match was Jaccard **1.0** - the pile reproduces HC3 **byte-for-byte** after normalisation, not
loosely. The removal pipeline therefore works, and the removal rate is not a rounding error: at 7.7%
of human rows, training on the raw pile would place a large block of the HC3 evaluation set into
training.

The full 1.39M-row run is in progress; final counts and digest are recorded in
`reports/e2/pile_decontamination.json`.

**Verdict:** usable **only** after decontamination, and the decontamination must be verified against
the evaluation sets before any training.

## `Ateeqq/AI-and-Human-Generated-Text` (AI-GA)

| Property | Value |
|---|---|
| Rows | **28,662** (22,930 train + 5,732 test), balanced **14,331 / 14,331** |
| Columns | `title`, `abstract`, `label` |
| Licence | **MIT** - corroborated upstream (`panagiotisanagnostou/AI-GA/LICENSE`, GitHub API) |
| Human provenance | **CORD-19** (per the upstream README) - COVID-19 research corpus |
| AI provenance | GPT-3, generated from the titles |
| Domain | biomedical abstracts; only **6.5%** mention COVID/SARS, so not narrowly topical |

**Overlap with evaluation sets: zero.** Exact and near-duplicate matching against Ghostbuster, HC3
and RAID returned no matches. Five exact duplicates exist between its own train and test splits.

### Two findings that change its value

1. **The MIT tag cannot cover the human text.** The abstracts come from CORD-19, which aggregates
   PubMed, bioRxiv, medRxiv and WHO material under per-paper terms. MIT on the wrapper does not
   relicense the underlying text. This is a provenance limitation, not a research blocker, but it
   means the corpus cannot be redistributed as MIT.
2. **The generated side is templated.** The six most common openings in the AI class:

   | Opening | Count |
   |---|---:|
   | "the title of this article is" | 84 |
   | "this paper presents a novel approach" | 69 |
   | "this paper presents an analysis of" | 66 |

   "The title of this article is" is a **prompt echo** - the instruction leaked into the output. The
   rest are stock GPT-3 academic openers. A classifier trained on this corpus can score well by
   learning three phrases, which is precisely the failure mode the audit exists to catch.

**Verdict:** clean on overlap, weak on signal. Usable as a **small, secondary** addition at most, and
only if the templated openers are stripped and the artefact audit confirms they are gone.
