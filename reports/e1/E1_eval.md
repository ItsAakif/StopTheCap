# Encoder evaluation

- model: `distilbert-base-cased`
- checkpoint: `/content/drive/MyDrive/StopTheCap_runs/E1/ckpt/last.pt`
- device: **cuda**
- max_length: 512
- frozen threshold: **0.0452** (quantile(0.95) of 202 human val scores)
- split digest: `932d20f1dca64c21537bed89739c633ace1b627c1a25f7d78c45a17915cf27a2`

| Split | n | pos | neg | ROC-AUC (95% CI) | PR-AUC | human FPR (95% CI) | P | R | F1 | Brier | ECE |
|---|---:|---:|---:|---|---:|---|---:|---:|---:|---:|---:|
| val | 1401 | 1199 | 202 | 0.999 [0.997, 1.000] | 1.000 | 0.054 [0.031, 0.095] | 0.991 | 0.997 | 0.994 | 0.007 | 0.007 |
| test_id | 1518 | 1295 | 223 | 0.999 [0.997, 0.999] | 1.000 | 0.036 [0.018, 0.069] | 0.994 | 0.992 | 0.993 | 0.011 | 0.011 |
| test_ood_domain | 6976 | 5998 | 978 | 0.980 [0.977, 0.983] | 0.997 | 0.055 [0.043, 0.071] | 0.990 | 0.923 | 0.956 | 0.083 | 0.086 |
| hc3 | 6575 | 2418 | 4157 | 0.817 [0.807, 0.827] | 0.725 | 0.112 [0.102, 0.122] | 0.711 | 0.473 | 0.568 | 0.250 | 0.239 |
| raid | 1972 | 1811 | 161 | 0.532 [0.493, 0.576] | 0.939 | 0.491 [0.415, 0.567] | 0.911 | 0.444 | 0.597 | 0.568 | 0.574 |

## Confusion matrices (at the frozen threshold)

- **val**: TN 191 · FP 11 · FN 4 · TP 1195
- **test_id**: TN 215 · FP 8 · FN 10 · TP 1285
- **test_ood_domain**: TN 924 · FP 54 · FN 461 · TP 5537
- **hc3**: TN 3693 · FP 464 · FN 1274 · TP 1144
- **raid**: TN 82 · FP 79 · FN 1007 · TP 804

## Caveats
- only 202 negatives: threshold metrics are indicative
- only 223 negatives: threshold metrics are indicative
- only 978 negatives: threshold metrics are indicative
- only 161 negatives: threshold metrics are indicative

## How to read this

- `test_id` is in-domain and is the optimistic number.
- `test_ood_domain`, `hc3` and `raid` are out-of-distribution. Do not describe a high in-domain AUC as general detection ability.
- The threshold came from `val` only and was frozen before any test split was scored.
