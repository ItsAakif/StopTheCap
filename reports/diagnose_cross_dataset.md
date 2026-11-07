# Cross-dataset failure diagnosis

Threshold frozen on `val` at a 5% human FPR budget: 0.4391. No hyperparameter, feature or threshold decision used RAID or HC3 labels.

| Evaluation set | n | pos | neg | ROC-AUC | OOV rate (label-free) | mean score |
|---|---:|---:|---:|---:|---:|---:|
| ghostbuster test_id (in-domain) | 1518 | 1295 | 223 | 0.996 | 19.4% | 0.767 |
| ghostbuster test_ood_domain (news_reuters) | 6976 | 5998 | 978 | 0.986 | 18.5% | 0.635 |
| ghostbuster val | 1401 | 1199 | 202 | 0.995 | 19.7% | 0.767 |
| raid (external) | 1972 | 1811 | 161 | 0.480 | 26.7% | 0.436 |
| hc3 (external) | 6575 | 2418 | 4157 | 0.652 | 19.5% | 0.298 |
| ghostbuster: train gpt* -> test claude (generator shift) | 426 | 203 | 223 | 0.928 | 0.0% | 0.273 |

## RAID by domain (reporting only)

| Domain | n | pos | neg | ROC-AUC | OOV rate | mean score |
|---|---:|---:|---:|---:|---:|---:|
| raid:abstracts | 397 | 397 | 0 | n/a | 19.8% | 0.540 |
| raid:books | 797 | 636 | 161 | 0.521 | 33.6% | 0.472 |
| raid:news | 678 | 678 | 0 | n/a | 19.2% | 0.283 |
| raid:poetry | 100 | 100 | 0 | n/a | 28.3% | 0.769 |

## Interpretation

- Generator transfer inside Ghostbuster: training on gpt* and testing on claude gives AUC 0.928. Generator change alone does not explain the external collapse; the difference is corpus/era rather than a single generator.
- Within the licensed corpus, moving to an unseen domain drops AUC by 0.010 (0.996 -> 0.986). Domain shift alone does not explain the external collapse.

## What this rules in and out

- **Domain sensitivity** is measured inside the licensed corpus, so it cannot be blamed on RAID.
- **Covariate shift** is estimated without labels: a high out-of-vocabulary rate means the evaluation text uses vocabulary the model never saw.
- Neither line of evidence licenses a claim of generalisation. They exist to explain a failure, not to tune it away.
