# E2 experiment design - does a second corpus improve cross-corpus transfer?

**Status:** proposed. Not started. Training is gated on the new corpus passing the existing audit.

**E1** = Ghostbuster only. AUC 0.999 in-domain, 0.817 HC3, **0.532 RAID** (CI includes chance).
**E2** = Ghostbuster + independently sourced second corpus, everything else held constant.

---

## Hypothesis

> Adding text from different domains and different generator families will improve performance on
> RAID, which E1 cannot separate at all.

**Pre-registered expectation: a modest gain at best.** The shortlisted corpora
(`docs/second-corpus-research.md`) add GPT-2, GPT-J and ChatGPT, but RAID's hardest families are
Llama, Mistral, MPT and Cohere. If RAID stays near chance, the hypothesis is falsified and the
conclusion is that *more of the same era of data does not fix corpus transfer* - which is a result
worth having before spending compute on generating new corpora.

---

## What is held constant

Deliberately identical to E1, so any difference is attributable to data:

| Item | Value |
|---|---|
| Architecture | `distilbert-base-cased`, 2-class head |
| Sequence length | 512 |
| Epochs / schedule | 3, lr 5e-5, batch 16 x grad-accum 2, fp16 |
| Class weighting | inverse-frequency, recomputed from the new train split |
| Threshold policy | 5% human-FPR budget, chosen on `val`, **frozen before test** |
| Evaluation sets | `test_id`, `test_ood_domain`, HC3, RAID - **unchanged, still quarantine-only** |
| Metrics | ROC-AUC with bootstrap CI, PR-AUC, human FPR with Wilson CI, P/R/F1, confusion, Brier, ECE |
| Split safeguards | group-based, deterministic, digest-pinned |

The E1 checkpoint and its results are kept untouched; E2 writes a separate checkpoint and report.

---

## Data composition

| Source | Role | Licence | Approx size |
|---|---|---|---|
| Ghostbuster | train (as E1) | CC BY 3.0 | 22,100 records |
| `artem9k/ai-text-detection-pile` | train (decontaminated) | MIT | up to ~1.39M, sample |
| `Ateeqq/AI-and-Human-Generated-Text` | train | MIT | 28,662 |
| HC3 | **evaluation only** | CC BY-SA 4.0 | 8,000 (quarantined) |
| RAID | **evaluation only** | MIT | 2,000 sample (quarantined) |

### Decontamination protocol (blocking)

The pile embeds 58k HC3 human and 27k HC3 ChatGPT rows. Training on them would contaminate the HC3
evaluation column. Because the pile carries no row-level provenance, removal must be content-based:

1. Build a MinHash index over **all** HC3 and RAID canonical records (already implemented in
   `ml/data/dedup.py`).
2. For every candidate pile row, compute 5-word shingle MinHash and test against that index at
   Jaccard >= 0.85.
3. Drop any row above threshold and record the count per class.
4. **Also** dedupe the pile against itself, and against Ghostbuster.
5. Report the removal rate. If > 5% of a class is removed, investigate before proceeding.

**If decontamination cannot be shown to work, E2 does not run.** An improvement on a contaminated
HC3 would be meaningless and would look like success.

### Splits

- New corpora join **train/val only**; they must never enter `test_id`, `test_ood_domain`, HC3 or RAID.
- `group_id` must be per source document, not per row. The pile has no document ids, so group by a
  content hash of the source text where available.
- The existing `COMBINED.yaml` digest changes. E2 therefore gets **its own digest**, and E1 results
  remain tied to `932d20f1...`. They are not comparable by digest, only by evaluation set.

---

## Ablations

| Run | Data | Question |
|---|---|---|
| E1 (done) | Ghostbuster | baseline |
| **E2a** | + pile (decontaminated) | does domain/generator breadth help? |
| **E2b** | + Ateeqq abstracts only | does one clean, narrow corpus help? |
| **E2** | + both | best-case composition |

E2b is cheap and isolates whether the pile's noise is helping or hurting.

---

## Success and falsification

| Outcome | Interpretation |
|---|---|
| RAID AUC >= 0.65 with CI excluding 0.5 | meaningful transfer gain - hypothesis supported |
| RAID AUC 0.55-0.65, CI excluding 0.5 | weak but real gain; report cautiously |
| RAID AUC CI still contains 0.5 | **falsified** - report as a negative result |
| In-domain drops below 0.95 | the mixture diluted the in-domain signal; report the trade-off |

Success is **not** an in-domain improvement. E1 already scores 0.999 there; there is nothing to win.

---

## Risks

| Risk | Mitigation |
|---|---|
| HC3 contamination survives decontamination | MinHash + manual spot check; report removal rate |
| Domain confound: pile's human text is Reddit fiction, AI is instructed prose | per-domain reporting; do not pool silently |
| Class balance shifts | recompute weights; report per-source counts |
| Licence drift on the pile (MIT claim is the uploader's) | recorded; treat as MIT but note provenance is unverified |
| Compute | E2 trains on more rows; expect 1-3 h on a T4 versus <1 h for E1 |
| Digest change breaks comparability | separate digest, explicit mapping in the report |

---

## Gate before training

1. New readers + manifests committed, with licence evidence URLs.
2. Decontamination report produced and reviewed.
3. Audit passes on the new combined corpus (all 12 gates, plus the decontamination check).
4. Split digest recorded.
5. Only then train.
