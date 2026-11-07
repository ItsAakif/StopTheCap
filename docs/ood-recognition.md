# OOD recognition: methods and evaluation design

**Status:** research only. Nothing here is implemented or claimed as a safeguard.

**Why this exists:** the abstention experiment (docs/calibration-and-abstention.md) showed that
confidence is not an OOD signal. At a 90% coverage target the confidence margin *raised* the
false-positive rate on HC3 (0.481) and RAID (0.876), and at 75% it abstained on >99% of HC3. So the
product needs a different signal for "this text is unlike anything I was trained on", or no
safeguard at all.

---

## What "OOD" means here

Two distinct questions, easily conflated:

1. **Domain/generator shift** - the text is the same kind of object but from a distribution the model
   never saw (RAID's Llama/Mistral output, book prose). This is what broke E1.
2. **Task shift** - the input is not really the thing being classified (a poem, code, a table).

Both matter, and a detector for (1) is not automatically a detector for (2).

---

## Candidate signals

| Signal | Idea | Cost | Known weakness |
|---|---|---|---|
| Max softmax probability | low confidence = OOD | free | **Already falsified here**: it is exactly what abstention used |
| Predictive entropy | diffuse distribution = OOD | free | Same family as above; shares the failure |
| **Energy score** `-logsumexp(logits)` | better-behaved than softmax, no calibration needed | free (reuse logits) | Still derived from the classifier head that is already wrong OOD |
| **Mahalanobis on CLS embeddings** | distance to the training distribution in representation space | one forward pass per doc, no gradients | Assumes a Gaussian per class; sensitive to the embedding layer chosen |
| **kNN distance on CLS embeddings** | mean distance to k nearest training embeddings | one pass + index | Needs the training embeddings stored (~11k x 768 floats = 34 MB) |
| **Perplexity under a reference LM** | text unlike the training domain has high perplexity | needs a small LM (e.g. distilgpt2) | Detects topic, not authorship; correlated with text length |
| **OOV / vocabulary coverage** | fraction of tokens unseen in training | free | Weak alone; we already compute it in the diagnosis (`diagnose.py`) |
| Conformal prediction | calibrated prediction *sets* with a coverage guarantee | moderate; needs a calibration split | Guarantee holds only under exchangeability - exactly what OOD violates |
| Deep ensembles / MC dropout | disagreement across models/passes | expensive | Marginal gain for a 66M single-checkpoint model here |

**Judgement:** the two most promising, given evidence already gathered, are **Mahalanobis / kNN on
CLS embeddings** (they look at the input representation rather than the suspect decision head) and
**perplexity under a small reference LM**. Both are cheap relative to training another model.

---

## Evaluation design

The critical point: an OOD detector must be judged on **whether it prevents false accusations**, not
on how well it separates distributions. A detector with AUROC 0.95 that flags the wrong texts is
useless.

### Stage 1 - separation (necessary, not sufficient)

- **ID set:** Ghostbuster `val` + `test_id` (the training distribution).
- **OOD sets:** HC3, RAID, and separately RAID per domain (books, news, abstracts, poetry).
- Metric: AUROC of "ID vs OOD", plus **FPR at 95% TPR** (how often genuine in-domain text is
  rejected).
- Report per OOD source, never pooled: RAID and HC3 are different shifts.

### Stage 2 - does it actually help?

This is the part the abstention experiment skipped and must not be skipped again.

- Build a policy: **reject** (abstain) when the OOD score crosses a threshold chosen on `val` to keep
  in-domain rejection at a fixed budget (e.g. 5%).
- Apply that policy unchanged to HC3 and RAID.
- Measure, among **accepted** samples only: human FPR, and compare with the no-rejection baseline
  (0.491 on RAID).
- **Success criterion:** accepted-set FPR materially below the baseline *at non-trivial coverage*.
  If coverage collapses below ~25% it is refusal, not selectivity.
- Report a risk-coverage curve where coverage is driven by the OOD score, not by confidence, so it
  can be compared directly with the failed confidence-based curve.

### Stage 3 - failure modes

- Correlation with text length (long documents may look OOD for trivial reasons).
- Correlation with domain rather than generator: if the detector flags "poetry" not "machine text",
  it will misfire on human poetry.
- Cost: one extra forward pass per request.

---

## What would make this a shippable safeguard

1. Stage 1 AUROC >= 0.90 for ID-vs-RAID **and** ID-vs-HC3, with in-domain FPR <= 5%.
2. Stage 2 accepted-set FPR at least 30% lower than baseline at >= 50% coverage.
3. A stated, measured failure mode section.
4. Unit tests pinning the score's behaviour on known-ID and known-OOD fixtures.

Until all four hold, the product should **abstain only on in-distribution uncertainty** and clearly
state that it has no safeguard for unfamiliar text. That is the current position.

---

## Recommendation

Run **Stage 1** first: Mahalanobis and kNN on CLS embeddings, plus perplexity under `distilgpt2`,
all evaluated ID vs HC3 and ID vs RAID-per-domain. It is cheap (embeddings only, no training) and
answers whether a usable signal exists at all before anyone builds a policy on top.

Do **not** wire any of this into the API until Stage 2 passes.
