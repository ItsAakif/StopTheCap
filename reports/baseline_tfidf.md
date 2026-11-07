# TF-IDF + logistic-regression baseline

Measured on the committed split manifest (`datasets/manifests/COMBINED.yaml`). No number here is an estimate.

Threshold selected on **val only** at a 5% human false-positive budget, then frozen before the test split was scored.

Class imbalance (machine:human) is handled with `class_weight='balanced'`.

## Configuration: `char`

- vocabulary size: 165,959
- train class counts: {'human': 1617, 'machine': 9533}
- threshold: 0.4066 (quantile(0.95) of 202 human val scores -> FPR budget 5%)

| Split | n | pos | neg | ROC-AUC | PR-AUC | acc | prec | rec | F1 | **human FPR (95% CI)** | Brier | ECE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| val | 1401 | 1199 | 202 | 0.995 | 0.999 | 0.975 | 0.991 | 0.980 | 0.985 | 0.054 [0.031, 0.095] | 0.034 | 0.089 |
| test_id | 1518 | 1295 | 223 | 0.998 | 1.000 | 0.982 | 0.998 | 0.981 | 0.989 | 0.013 [0.005, 0.039] | 0.028 | 0.087 |
| quarantine:hc3 | 6575 | 2418 | 4157 | 0.767 | 0.681 | 0.694 | 0.854 | 0.203 | 0.329 | 0.020 [0.016, 0.025] | 0.225 | 0.189 |
| quarantine:raid | 1972 | 1811 | 161 | 0.500 | 0.938 | 0.404 | 0.931 | 0.379 | 0.538 | 0.317 [0.250, 0.392] | 0.450 | 0.561 |
| ↳ _val_ | | | | | | | | | | only 202 negatives: threshold metrics are indicative, not measurements | | |
| ↳ _test_id_ | | | | | | | | | | only 223 negatives: threshold metrics are indicative, not measurements | | |
| ↳ _raid_ | | | | | | | | | | only 161 negatives: threshold metrics are indicative, not measurements | | |

## Configuration: `word`

- vocabulary size: 200,000
- train class counts: {'human': 1617, 'machine': 9533}
- threshold: 0.4391 (quantile(0.95) of 202 human val scores -> FPR budget 5%)

| Split | n | pos | neg | ROC-AUC | PR-AUC | acc | prec | rec | F1 | **human FPR (95% CI)** | Brier | ECE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| val | 1401 | 1199 | 202 | 0.995 | 0.999 | 0.969 | 0.991 | 0.972 | 0.981 | 0.054 [0.031, 0.095] | 0.041 | 0.110 |
| test_id | 1518 | 1295 | 223 | 0.996 | 0.999 | 0.974 | 0.992 | 0.978 | 0.985 | 0.045 [0.025, 0.081] | 0.036 | 0.107 |
| quarantine:hc3 | 6575 | 2418 | 4157 | 0.652 | 0.496 | 0.640 | 0.519 | 0.291 | 0.373 | 0.157 [0.146, 0.168] | 0.224 | 0.079 |
| quarantine:raid | 1972 | 1811 | 161 | 0.480 | 0.905 | 0.468 | 0.917 | 0.463 | 0.615 | 0.472 [0.396, 0.549] | 0.374 | 0.484 |
| ↳ _val_ | | | | | | | | | | only 202 negatives: threshold metrics are indicative, not measurements | | |
| ↳ _test_id_ | | | | | | | | | | only 223 negatives: threshold metrics are indicative, not measurements | | |
| ↳ _raid_ | | | | | | | | | | only 161 negatives: threshold metrics are indicative, not measurements | | |

## How to read this

- **Human FPR is the primary metric.** A false accusation on human writing is the worst product failure; FPR is reported with its Wilson interval and the negative count.
- **`test_id` is in-domain**: same sources and generators as training, different prompts/authors. It is the optimistic number.
- **`quarantine:*` is independent.** RAID and HC3 were never fitted on.
- Where a split has fewer than 1000 negatives, threshold metrics are marked indicative. Do not cite them as measurements.
- A high in-domain AUC alongside a low independent AUC is the expected shape for this corpus and is the finding, not a bug.
