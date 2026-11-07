# E1 calibration, abstention and threshold reliability

**Run:** 2026-11-05 against the E1 checkpoint (`E1/ckpt/last.pt`, `distilbert-base-cased`).
**Artefacts:** `reports/e1/calibration.json` (all numbers), `reports/e1/calibration.md` (generated
tables), `reports/e1/predictions.jsonl` (per-record scores; regenerable, not committed).

**Sample sizes actually scored:** val 1,401 · test_id 1,518 · test_ood_domain 2,000 (strided from
6,976) · hc3 2,000 (strided from 6,575) · raid 1,972. The two capped splits are labelled wherever
they appear.

Everything below is fitted on **`val` only**. No calibration parameter, abstention margin or
threshold was chosen using `test_id`, `test_ood_domain`, HC3 or RAID.

---

## Task 1 - Temperature scaling

One parameter, fitted by minimising NLL on the 1,401 `val` rows (1,199 machine / 202 human):

**T = 1.7605**  → the model is overconfident, and the fit softens it.

Both thresholds, reported separately as required:

| Threshold | Value | How |
|---|---|---|
| **Original frozen threshold (E1)** | **0.0452** | unchanged; recomputed from raw val human scores and matched the run log exactly |
| Calibrated threshold | **0.1492** | same 5% quantile, applied to calibrated val human scores |

### Effect on calibration

| Split | n | ECE raw | ECE calibrated | Brier raw | Brier calibrated |
|---|---:|---:|---:|---:|---:|
| val | 1,401 | 0.007 | **0.005** | 0.0068 | 0.0066 |
| test_id | 1,518 | 0.011 | **0.008** | 0.0106 | 0.0102 |
| test_ood_domain | 2,000 | 0.088 | 0.079 | 0.0841 | 0.0771 |
| hc3 | 2,000 | 0.249 | 0.219 | 0.2579 | 0.2398 |
| **raid** | 1,972 | **0.574** | **0.561** | 0.5676 | 0.5477 |

Human FPR at the calibrated threshold: test_id 3.6% · ood 6.1% · HC3 11.7% · **RAID 49.1%**.

**Temperature scaling does not fix discrimination or domain shift.** It cannot: it is a monotone
transform, so ranking - and therefore AUC - is unchanged (RAID stays at 0.532). It improves ECE where
the model already discriminates (val, test_id) and does almost nothing where it does not
(RAID: 0.574 → 0.561). That is the expected behaviour of a calibration method applied to a model
whose scores carry no signal on that distribution, and it should be stated that way rather than
presented as a partial fix.

---

## Task 2 - Abstention (selective classification)

**Policy:** accept a decision when the calibrated score is at least a margin `m` from the calibrated
threshold; otherwise abstain. `m` is chosen **on `val`** to hit a target coverage, then applied
unchanged to every other split.

### FPR among accepted human samples

| Target coverage (margin) | val | test_id | ood | hc3 | raid |
|---|---:|---:|---:|---:|---:|
| 100% (0.004) | 0.054 | 0.036 | 0.061 | 0.117 | **0.491** |
| 90% (0.142) | 0.136 | 0.096 | **0.500** | **0.481** | **0.876** |
| 75% (0.847) | n/a | n/a | n/a | n/a | n/a |

### Selective risk (error among accepted)

| Target coverage | val | test_id | ood | hc3 | raid |
|---|---:|---:|---:|---:|---:|
| 100% | 0.011 | 0.012 | 0.073 | 0.271 | 0.551 |
| 90% | 0.007 | 0.005 | 0.009 | 0.213 | 0.209 |
| 75% | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

### Coverage actually achieved on external splits

| Target | val | test_id | ood | hc3 | raid |
|---|---:|---:|---:|---:|---:|
| 100% | 1.000 | 1.000 | 1.000 | 0.997 | 0.998 |
| 90% | 0.900 | 0.895 | 0.800 | **0.268** | **0.507** |
| 75% | 0.752 | 0.754 | 0.514 | **0.002** | **0.092** |

### The finding

**In-distribution, abstention works.** On val and test_id, risk falls from ~0.011 to 0.007 at 90%
coverage and to 0.000 at 75% - exactly the intended behaviour.

**Out of distribution, it makes things worse, and then stops working entirely.**

1. At the 90% margin, the model still *answers confidently* on HC3 and RAID, so coverage stays high
   (26.8% and 50.7%) - and **FPR among accepted humans rises to 0.481 and 0.876**, worse than
   answering everything. A confident wrong answer is far from the threshold, so the margin filter
   preferentially keeps it.
2. At the 75% margin, coverage collapses to 0.2% and 9.2% - the system effectively refuses
   everything rather than becoming selectively accurate.

**Conclusion: model confidence is not a usable uncertainty signal under domain shift.** Abstention
as implemented is a valid in-distribution safeguard and must not be presented as protection against
unfamiliar text. Any OOD safety would need an *input-distribution* signal (out-of-distribution
detection on the text itself), not a confidence margin. That is not implemented and is not claimed.

---

## Task 3 - Threshold reliability

The frozen E1 threshold was **not replaced**. This section quantifies how much confidence it deserves.

| Quantity | Value |
|---|---|
| Human validation examples | **202** |
| Flagged at the threshold | **11** (FPR 5.4%) |
| Wilson 95% CI on FPR | **[3.1%, 9.5%]** |
| Flags expected at a 5% budget | 10.1 |
| Rank of the examples that decide the cut | **10 and 11 of 202** |
| Score gap between those two examples | **0.0942** |
| Leave-one-out: max threshold shift | **0.0343** |
| … as a fraction of the threshold | **75.9%** |
| Leave-one-out sd | 0.0082 |
| More human validation needed | **Yes** |

Three things follow:

1. **The 5% budget is really "somewhere between 3% and 9.5%".** 202 negatives cannot resolve a 5%
   target more precisely than that.
2. **The threshold is decided by one rank position.** The 95th percentile sits between the 10th and
   11th largest human scores, and those two scores are 0.094 apart - much larger than the threshold
   itself (0.045). So a single different human example can move the operating point substantially;
   the jackknife confirms up to a 76% relative shift.
3. **Therefore the reported FPRs are indicative, not measurements** - which is exactly how the
   evaluation report already labels them.

**Recommendation:** a 5% budget needs on the order of **1,000+ human validation examples** before the
threshold should be treated as a product guarantee. That is the same rule the research dossier set
for reporting TPR@FPR, now applied to the threshold itself.

---

## What this does and does not establish

**Establishes:** calibration improves in-domain probability estimates; abstention is safe and useful
in distribution; the frozen threshold is statistically fragile because of the 202-example validation
set.

**Does not establish:** any improvement in detection, any transfer to RAID, or any safety mechanism
for unfamiliar text. Temperature scaling cannot change AUC by construction, and confidence-based
abstention demonstrably fails out of distribution.
