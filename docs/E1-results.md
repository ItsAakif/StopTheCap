# E1 results - distilbert encoder

**Run:** 2026-11-05 on a Colab Tesla T4 (15.6 GB), torch 2.11.0+cu130.
**Artefacts:** `reports/e1/E1.json` (run log), `reports/e1/E1_eval.{json,md}`.
**Checkpoint:** `/content/drive/MyDrive/StopTheCap_runs/E1/ckpt/last.pt` (753 MB, not committed).

Configuration: `distilbert-base-cased`, 3 epochs, 2,088 steps, batch 16 x grad-accum 2, 512 tokens,
class weights 3.448 / 0.585, threshold frozen at **0.0452** from 202 human `val` scores at a 5%
false-positive budget. Split digest `932d20f1...` - unchanged from every earlier result.

---

## Results

| Split | n | neg | ROC-AUC [95% CI] | PR-AUC | Human FPR [95% CI] | P | R | F1 | ECE |
|---|---:|---:|---|---:|---|---:|---:|---:|---:|
| val | 1,401 | 202 | 0.999 [0.997, 1.000] | 1.000 | 0.054 [0.031, 0.095] | 0.991 | 0.997 | 0.994 | 0.007 |
| **test_id** | 1,518 | 223 | **0.999 [0.997, 0.999]** | 1.000 | **0.036 [0.018, 0.069]** | 0.994 | 0.992 | 0.993 | 0.011 |
| test_ood_domain | 6,976 | 978 | 0.980 [0.977, 0.983] | 0.997 | 0.055 [0.043, 0.071] | 0.990 | 0.923 | 0.956 | 0.086 |
| **hc3** | 6,575 | 4,157 | **0.817 [0.807, 0.827]** | 0.725 | **0.112 [0.102, 0.122]** | 0.711 | 0.473 | 0.568 | 0.239 |
| **raid** | 1,972 | 161 | **0.532 [0.493, 0.576]** | 0.939 | **0.491 [0.415, 0.567]** | 0.911 | 0.444 | 0.597 | 0.574 |

Confusion matrices at the frozen threshold:

| Split | TN | FP | FN | TP |
|---|---:|---:|---:|---:|
| test_id | 215 | 8 | 10 | 1,285 |
| test_ood_domain | 924 | 54 | 461 | 5,537 |
| hc3 | 3,693 | 464 | 1,274 | 1,144 |
| raid | 82 | 79 | 1,007 | 804 |

Evaluation wall clock: 331.9 s (22.6 s for the threshold pass on `val`).

---

## Against the TF-IDF baseline

| Split | TF-IDF (word) | TF-IDF (char) | E1 encoder | Delta vs word |
|---|---:|---:|---:|---:|
| test_id | 0.996 | 0.998 | **0.999** | +0.002 |
| hc3 | 0.652 | 0.767 | **0.817** | **+0.164** |
| raid | 0.480 | 0.500 | **0.532** | +0.053 |

The encoder is the best model on every split. The gain is large on HC3 and marginal on RAID.

---

## The finding that matters

**On RAID the encoder is statistically indistinguishable from chance.**
AUC 0.532 with a 95% bootstrap CI of **[0.493, 0.576]** - the interval contains 0.5. We cannot claim
it ranks RAID's machine text above its human text.

And the product-relevant number is worse than that summary suggests: at the threshold tuned to a 5%
human-FPR budget, RAID human text is flagged **49.1%** of the time [41.5%, 56.7%]. Half of real
human writing would be labelled AI.

Calibration collapses out of distribution - ECE 0.011 in-domain, **0.574** on RAID. The model is not
merely wrong there, it is confidently wrong.

### What explains it

Consistent with the earlier cross-dataset diagnosis: domain shift inside Ghostbuster costs 0.010 AUC
and generator shift 0.072, but crossing corpora costs ~0.47. The encoder inherits the same
limitation as TF-IDF; a stronger architecture did not fix it, which is itself the useful result.

The likely mechanism is the same one identified earlier: Ghostbuster's human class is student essays
with citations and a specific register; RAID's human class is published prose (books, abstracts,
news) and its machine class spans 11 generators, none of them GPT-3 davinci. The model learned
"Ghostbuster-human vs GPT-3", not "human vs machine".

---

## What this result supports

- A fine-tuned encoder trained on an audited, leakage-free split, evaluated on three held-out sets
  with a frozen threshold and bootstrap intervals.
- A large, reproducible improvement over the classical baseline on HC3 (+0.164 AUC).
- A documented negative: **in-domain and in-corpus results do not transfer across corpora**, and
  the collapse survives a better model.

## What it does not support

- Any claim of general AI-text detection.
- Any user-facing score without re-validating the threshold out of domain - the 5% budget is 49%
  on RAID.
- Any claim that the encoder is "better" in a way users would notice; out of distribution it is not.

---

## Gaps in this run

| Gap | Consequence |
|---|---|
| Training wall-clock not recorded | we know 2,088 steps ran, not how long |
| Run log still reads "evaluation pending" | stale string in `train_encoder.py`; the evaluation did run |
| No temperature scaling | ECE is reported raw; calibration could improve without touching AUC |
| RAID has only 161 negatives | its FPR interval is wide; both should be read as indicative |
| Single corpus for training | no way to test whether more generators would help |

## Next

1. **Calibration + abstention**: temperature scaling on `val`, then a risk-coverage curve so the
   product can refuse rather than guess.
2. **Threshold re-validation on a distribution resembling real input** - the current one was tuned
   on Ghostbuster.
3. **A second training corpus** with different generators/domains is the only experiment likely to
   move the RAID number; nothing else in the plan addresses it.
