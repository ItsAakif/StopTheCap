# Encoder evaluation

- model: `distilbert-base-cased`
- checkpoint: `checkpoints\E1-smoke\last.pt`
- device: **cpu**
- max_length: 128
- frozen threshold: **0.1194** (quantile(0.95) of 202 human val scores)
- split digest: `932d20f1dca64c21537bed89739c633ace1b627c1a25f7d78c45a17915cf27a2`

| Split | n | pos | neg | ROC-AUC (95% CI) | PR-AUC | human FPR (95% CI) | P | R | F1 | Brier | ECE |
|---|---:|---:|---:|---|---:|---|---:|---:|---:|---:|---:|
| val | 1401 | 1199 | 202 | 0.580 [0.542, 0.620] | 0.890 | 0.054 [0.031, 0.095] | 0.923 | 0.109 | 0.195 | 0.694 | 0.756 |
| test_id | 1518 | 1295 | 223 | 0.583 [0.543, 0.619] | 0.887 | 0.063 [0.038, 0.103] | 0.904 | 0.102 | 0.183 | 0.691 | 0.753 |
| test_ood_domain | 6976 | 5998 | 978 | 0.592 [0.574, 0.612] | 0.892 | 0.001 [0.000, 0.006] | 0.988 | 0.014 | 0.028 | 0.695 | 0.758 |
| hc3 | 6575 | 2418 | 4157 | 0.634 [0.621, 0.648] | 0.464 | 0.135 [0.125, 0.145] | 0.496 | 0.228 | 0.313 | 0.297 | 0.258 |
| raid | 1972 | 1811 | 161 | 0.521 [0.461, 0.584] | 0.883 | 0.379 [0.308, 0.456] | 0.822 | 0.156 | 0.262 | 0.742 | 0.816 |

## Confusion matrices (at the frozen threshold)

- **val**: TN 191 · FP 11 · FN 1068 · TP 131
- **test_id**: TN 209 · FP 14 · FN 1163 · TP 132
- **test_ood_domain**: TN 977 · FP 1 · FN 5913 · TP 85
- **hc3**: TN 3597 · FP 560 · FN 1866 · TP 552
- **raid**: TN 100 · FP 61 · FN 1529 · TP 282

## Caveats
- only 202 negatives: threshold metrics are indicative
- only 223 negatives: threshold metrics are indicative
- only 978 negatives: threshold metrics are indicative
- only 161 negatives: threshold metrics are indicative

## How to read this

- `test_id` is in-domain and is the optimistic number.
- `test_ood_domain`, `hc3` and `raid` are out-of-distribution. Do not describe a high in-domain AUC as general detection ability.
- The threshold came from `val` only and was frozen before any test split was scored.
