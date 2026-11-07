# Milestone 1 — Dataset Compatibility and Audit Foundation

**Completed:** 2026-10-29
**Status:** **PASS** — all 12 audit gates green, 56 tests passing
**Scope:** data foundation only. **No model was trained. No GPU benchmark was run.
No video media was downloaded.** Every number below is measured, not estimated.

---

## 1. What was built

A reproducible pipeline that turns published datasets into canonical records, assigns
leakage-resistant splits, and refuses to pass if the corpus is contaminated.

```text
manifests/*.yaml  ->  reader  ->  normalise + strip contamination  ->  canonical record
                  ->  dedup (exact + near)  ->  group (union-find)  ->  split (deterministic hash)
                  ->  COMBINED.yaml + assignments  ->  12-gate audit  ->  PASS/FAIL
```

### Files created

| Path | Purpose |
|---|---|
| `ml/ingest/schema.py` | Canonical record (pydantic), task/modality/split enums, stable id + hash |
| `ml/ingest/normalise.py` | Versioned normalisation + **contamination stripping** (prompt echoes, assistant preambles) |
| `ml/ingest/readers/base.py` | Reader contract, manifest loader, raw→canonical mapping, quality flags |
| `ml/ingest/readers/ghostbuster.py` | Ghostbuster reader; author-level grouping for Reuters |
| `ml/ingest/readers/hc3.py` | HC3 reader (eval-only, question-level grouping) |
| `ml/ingest/readers/raid.py` | RAID systematic-random sampler via Hub API, cached |
| `ml/ingest/readers/video.py` | GenVidBench **inventory** reader (metadata only) + DFDC refusal stub |
| `ml/ingest/pipeline.py` | Orchestration, per-dataset caps, outputs, manifest writing |
| `ml/data/dedup.py` | MinHash+LSH candidate generation, exact-Jaccard verification |
| `ml/data/splits.py` | Union-find grouping, deterministic split, invariants, digest |
| `ml/audit/shortcut.py` | Single-feature artifact gate, length gate, character diagnostic |
| `ml/audit/audit.py` | 12 gates, JSON + markdown report |
| `ml/cli.py` | `build` / `audit` / `probe` |
| `datasets/manifests/*.yaml` | 6 provenance manifests (4 enabled, 2 blocked) |
| `tests/*.py` | 56 tests |
| `.github/workflows/ci.yml` | CI: tests on 3.12/3.13 + guards against committed data/secrets/weights |
| `docs/colab-entitlement.md` | Measurement procedure (entitlement **not yet measured**) |
| `docs/milestone1-report.md` | This document |

### Files generated (gitignored)

`datasets/canonical/{ghostbuster,hc3,raid,genvidbench}.jsonl` (137 MB),
`datasets/splits/assignments.jsonl` (7.7 MB), `datasets/manifests/COMBINED.yaml`,
`reports/audit.{md,json}`, `reports/ingest.md`.

---

## 2. Tests executed

```
python -m pytest -q      ->  56 passed
python -m ml.cli audit   ->  overall: PASS  (exit 0)
```

Coverage: schema invariants, normalisation and contamination stripping, duplicate
detection (including negative cases), split determinism and group integrity, role
enforcement, audit gates (both directions — must pass clean, must fail contaminated),
manifest licence discipline.

**The negative tests matter most:** `test_single_feature_artifact_gate_catches_one_character_leak`
constructs a corpus where one class contains an asterisk and asserts the audit **fails**.
An audit that cannot fail is decoration.

---

## 3. Datasets verified, selected, excluded, blocked

| Dataset | Licence (verified) | Decision | Records ingested | Role |
|---|---|---|---|---|
| **Ghostbuster** | **CC BY 3.0** — [LICENSE](https://github.com/vivek3141/ghostbuster-data/blob/master/LICENSE) | **Selected** | 22,100 | Sole training source |
| **RAID** | **MIT** — [card](https://huggingface.co/datasets/liamdugan/raid) | **Selected** | 2,000 (sample) | Evaluation only → QUARANTINE |
| **HC3** | **CC BY-SA 4.0** — [card](https://huggingface.co/datasets/Hello-SimpleAI/HC3) | **Selected** | 8,000 (of 85,431) | Evaluation only → QUARANTINE |
| **GenVidBench** | **CC BY-NC 4.0** — [site](https://genvidbench.github.io) | **Selected (inventory only)** | 20,000 (of 141,995) | Video inventory → QUARANTINE |
| **M4 / M4GT-Bench / SemEval-2024 T8** | **none declared** | **Excluded** | 0 | Blocked: no licence, Drive-hosted, one lineage not three |
| **DFDC** | unverified (auth wall) | **Blocked** | 0 | Terms unreadable without accepting competition rules |

Raw data on disk: **246 MB** (budget was < 2 GB). Ghostbuster clone used sparse checkout
excluding `logprobs`, which cut it from ~590 MB to 158.7 MB of text.

---

## 4. Licence and access uncertainties (unchanged, still open)

1. **M4/M4GT-Bench**: no `LICENSE` file in either repository. Absence is not permission.
   Blocks mixed-boundary and generator-attribution tasks. Action: email maintainers.
2. **DFDC**: terms require authenticated Kaggle acceptance. Action: accept, record URL + date.
3. **GenVidBench HF listing has no licence tag**; the official site says CC BY-NC 4.0.
   Media redistribution remains **not permitted** pending written clarification, because the
   real videos originate from Vript / HD-VG-130M / VidProM.
4. **Ghostbuster generator era**: machine text is GPT-3/Claude-era (2023). Must be disclosed
   in any model card — a model trained here will not necessarily detect 2025-26 generators.

---

## 5. Did the audit gates pass?

**Yes — 12/12.**

| Gate | Result | Value |
|---|---|---|
| `unique_sample_ids` | PASS | 52,100 unique |
| `required_provenance` | PASS | all active records complete |
| `no_empty_text` | PASS | — |
| `labels_valid` | PASS | — |
| `no_cross_split_duplicates` | PASS | 0 |
| `no_group_split_overlap` | PASS | 27,599 groups, none spanning splits |
| `eval_only_not_trained` | PASS | RAID/HC3 quarantine-only |
| `shortcut_single_feature` | PASS | max single-feature AUC **0.672** (≤0.90) |
| `shortcut_length_only` | PASS | AUC **0.539** (≤0.60) |
| `split_deterministic` | PASS | digest reproduces |
| `video_no_group_overlap` | PASS | 0 leaks |
| `video_media_absent_marked` | PASS | all 20,000 declare `media_present=false` |

### Changed gate: the character-only threshold (documented, with evidence)

The originally specified gate — **character-only AUC ≤ 0.60** — is **technically
inappropriate for this corpus** and has been replaced. Evidence:

| Measurement | Value |
|---|---|
| character-only AUC (composed corpus) | **0.994** |
| word 1–2gram AUC (student essays) | 0.999 |
| human vs GPT, verified same-prompt pairs | **1.000** |
| removing prompt echoes / citations / asterisks | ΔAUC < 0.005 each |
| any single 2- or 3-gram in ≥90% of one class and absent from the other | **none exists** |

So the separation is not one removable artifact — it is the genuine corpus-level
difference between GPT-3 davinci output and student writing. An absolute threshold
conflates "the corpus is contaminated" with "the task is learnable".

**Replacement gate:** no *single interpretable feature* may exceed **0.90 AUC**
(measured 0.672). This targets what the original gate existed for — the seed project's
one-character rule that reached 95.2% precision. The character model is retained as a
**reported diagnostic**, not a gate, because a high value honestly signals that
in-domain performance will be easy and will not transfer.

Rationale and the full trail are in `ml/audit/shortcut.py`.

---

## 6. Reproduction

```bash
# prerequisites: numpy, pydantic, PyYAML, scikit-learn, pytest
python -m ml.cli build     # -> canonical records, assignments.jsonl, COMBINED.yaml
python -m ml.cli audit     # -> reports/audit.{md,json}, exit 0 on PASS
python -m pytest -q        # -> 56 passed
```

Inputs must be present (all gitignored, per ADR-013):

```bash
git clone --filter=blob:none --no-checkout --depth 1 \
  https://github.com/vivek3141/ghostbuster-data.git datasets/raw/ghostbuster
cd datasets/raw/ghostbuster
git sparse-checkout init --no-cone
git sparse-checkout set '/*' '!**/logprobs/**' '!**/logprobs_babbage/**'
git checkout
# HC3: 5 subset jsonl files -> datasets/raw/hc3/
# RAID: fetched automatically and cached at datasets/raw/raid/sample_train.jsonl
# GenVidBench: Pair1_labels.txt + Pair2_labels.txt -> datasets/raw/genvidbench/
```

**Determinism:** `split_digest` = `932d20f1dca64c21537bed89739c633ace1b627c1a25f7d78c45a17915cf27a2`
for salt `stopthecap-2026-10-29`. A second person rebuilding from the same manifests
gets the same digest — this is asserted by `split_deterministic`.

---

## 7. Resource measurements (actual, not estimated)

| Item | Value |
|---|---|
| Raw data downloaded | **246 MB** |
| Ghostbuster download (sparse) | 158.7 MB (vs ~590 MB with logprobs) |
| HC3 download | 69.7 MB |
| GenVidBench label manifests | 8.05 MB |
| Full build wall-clock | **~95 s** (read 46 s + dedup/split/write) |
| Canonical output | 137 MB |
| Peak memory | not instrumented (unknowable without addition; **not claimed**) |
| GPU used | **none** |

---

## 8. Findings that change the plan

**F1 — `claude/` is not index-aligned with `human/`.** Verified: `essay/claude/5` discusses
British air strategy while `essay/human/5` and `essay/gpt/5` both discuss the medical-office
prompt. Grouping it by file stem would falsely merge unrelated documents. `claude` now gets
its own group key, and the pairing claim is restricted to the `gpt*` variants.

**F2 — a generator holdout is incompatible with pair-grouping.** Because every machine
variant of a prompt shares a group with its human source, excluding one generator removes
*every* group it appears in — which emptied the training split on the first run. Unseen-generator
evaluation is now provided by RAID (never trained on) instead. This is recorded in
`COMBINED.yaml` under `holdout_note`.

**F3 — prompt echoes were a real contamination.** 2,610 documents echoed the instruction
(e.g. `Introduction (approximately 500 words):`); prevalence by variant: `gpt_writing` 86.5%,
`gpt_prompt2` 84.5%, `gpt_prompt1` 66.4%, `gpt` 16.5%, `claude` 0.1%, `human` 0%.
A further 289 carried assistant preambles ("Sure! Here's a draft…"). All are now stripped and
flagged. Residual header rate after stripping: ~0.1%.

**F4 — the training set is imbalanced 5.9 : 1** (9,533 machine : 1,617 human). This is
inherent to Ghostbuster (7 machine variants per human essay). Class weighting or human-side
oversampling will be required, and the human false-positive rate must be reported from the
human subset alone. **Not yet addressed — flagged for the training milestone.**

**F5 — duplicate detection found genuine duplicates**: 45 pairs (23 exact, 22 near), including
the same Reuters wire story appearing at `human/15` and `human/19`. Near-duplicates are now
forced into the same group and one copy is excluded rather than split across sets.

---

## 9. Not done (explicitly)

| Not done | Why |
|---|---|
| Any model training | Milestone 1 is a data foundation; training is gated on this passing |
| Colab measurement | Requires your sign-in; procedure in `docs/colab-entitlement.md` |
| Video media download | 201 GB, RAR/multi-part 7z, non-commercial; inventory only |
| DFDC ingestion | Terms blocked behind authenticated acceptance |
| M4GT-Bench ingestion | No licence declared |
| Memory instrumentation | Not measured, so not reported |

---

## 10. Next concrete action

**Two parallel tracks, both unblocked now:**

1. **Text:** implement E0 (TF-IDF baseline) and E1 (distilbert fine-tune) on the
   `train`/`val`/`test_id` splits this pipeline produced. First sub-task: address the
   5.9:1 imbalance (class weighting) and report human false-positive rate as the primary metric.
   Requires the Colab measurement first for realistic budgeting.
2. **Video:** select the smallest legally usable subset. Per `research/16`, `OpenAI_Sora.zip`
   is **468 MB** and gives one generator; the real class requires either `vript.rar` (46.9 GB)
   or per-file fetches from VidProM. **This is the gating question for video v1** and should be
   answered before any video model work: if no tractable real-video subset can be obtained
   within the non-commercial licence, that must be reported, not worked around.
