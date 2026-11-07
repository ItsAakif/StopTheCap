# Baseline results — TF-IDF + logistic regression

**Measured 2026-10-29.** Reproduce with `python -m ml.cli baseline`. Machine-readable output:
`reports/baseline_tfidf.json`. Raw table: `reports/baseline_tfidf.md`.

Every number here was produced by running the code against the committed split manifest
(`split_digest 932d20f1…`). None is estimated.

---

## Headline

**The baseline is near-perfect in-domain and worthless out of domain.** That is the single most
important result of this phase, and it is the number any later model must be compared against.

| Split | Basis | n | negatives | ROC-AUC (word) | ROC-AUC (char) | Human FPR @ frozen threshold (word) |
|---|---|---:|---:|---:|---:|---|
| `test_id` | in-domain, unseen prompts/authors | 1,518 | 223 | **0.996** | **0.998** | 4.5% |
| `quarantine: hc3` | independent, never fitted | 6,575 | 4,157 | **0.652** | 0.767 | 15.7% |
| `quarantine: raid` | independent, never fitted | 1,972 | 161 | **0.480** | 0.500 | **47.2%** |

AUC of 0.48 on RAID is **at or below chance**. The same model that scores 0.996 in-domain cannot
rank RAID's human text above RAID's machine text at all.

## The threshold does not transfer — the product-critical finding

The decision threshold was fixed on `val` at a 5% human false-positive budget and then frozen.

| Split | Human FPR at the frozen threshold |
|---|---|
| `val` (where it was chosen) | 5.4% — by construction |
| `test_id` (in-domain) | 4.5% — holds |
| `hc3` (independent) | **15.7%** (word) / 2.0% (char) |
| `raid` (independent) | **47.2%** (word) / 31.7% (char) |

On RAID, roughly **one in two human-written passages would be flagged as AI**. The 5% budget was
measured on the wrong distribution. This is the concrete reason the product must never present the
score as proof, and why a threshold must be validated out of domain before any user-facing claim.

## What this rules out

- **Not** "the model works and RAID is hard." AUC 0.48 means the ranking is uninformative, not merely
  shifted. A threshold problem would leave AUC intact; this does not.
- **Not** a label error: RAID's human rows are its `model == "human"` records, sampled across
  2.27 M rows.

## Diagnostics that explain the shape

| Diagnostic | Result | Meaning |
|---|---|---|
| Character-only AUC, per source, internal | ghostbuster 0.996 · hc3 0.997 · raid 0.974 | Each corpus is internally separable |
| Dataset fingerprint: ghostbuster vs hc3 | **AUC 1.000** | The corpora are perfectly distinguishable from text alone |
| Dataset fingerprint: ghostbuster vs raid | 0.999 | Same |
| Dataset fingerprint: hc3 vs raid | 0.997 | Same |

Every corpus is separable *within itself* but a model trained on one transfers to another at chance.
The model is learning **corpus identity**, not transferable authorship signal. This is why the
project reports per-source numbers and refuses to present a pooled figure as if the sources were
exchangeable.

## Caveats that must travel with these numbers

- **`raid` has only 161 human negatives.** Its FPR interval is wide (word: 0.396–0.549) and the
  report marks it *indicative, not a measurement*. The direction is unambiguous; the point estimate
  is not precise.
- **`test_id` has 223 negatives** — also below the 1,000-negative rule, so its FPR is likewise marked
  indicative. Its AUC is unaffected by that rule.
- **PR-AUC is inflated on RAID** (0.905–0.938) purely because 92% of that sample is machine text. Do
  not read it as good performance; read the AUC and the FPR.
- **No encoder has been trained.** This is the floor, not a ceiling.

## What happens next, given this result

1. The encoder must be evaluated on **exactly this matrix**, including the RAID/HC3 columns. An
   in-domain-only report would repeat the seed project's mistake in a new form.
2. If the encoder also collapses on RAID, the honest conclusion is that **cross-generator text
   detection does not transfer from 2023-era Ghostbuster data**, and that becomes the project's
   finding rather than a failure to hide.
3. Before any user-facing number, the threshold must be re-validated on a distribution resembling
   real input — not on the training corpus.
