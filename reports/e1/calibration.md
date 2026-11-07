# E1 calibration and abstention

Temperature fitted on **val only** (1401 rows, 202 human / 1199 machine): **T = 1.760459**

- frozen threshold (E1, unchanged): **0.045211683958768845**
- recomputed from raw val scores: 0.045212 (matches run log: True)
- calibrated threshold: **0.149181**
- target human FPR: 5%

## Calibration effect

| Split | n | ECE raw | ECE calibrated | Brier raw | Brier calibrated |
|---|---:|---:|---:|---:|---:|
| val | 1401 | 0.007 | 0.005 | 0.0068 | 0.0066 |
| test_id | 1518 | 0.011 | 0.008 | 0.0106 | 0.0102 |
| test_ood_domain | 2000 | 0.088 | 0.079 | 0.0841 | 0.0771 |
| hc3 | 2000 | 0.249 | 0.219 | 0.2579 | 0.2398 |
| raid | 1972 | 0.574 | 0.561 | 0.5676 | 0.5477 |

## Human FPR at threshold

| Split | negatives | FPR (raw thr) | FPR (calibrated thr) [95% CI] |
|---|---:|---|---|
| val | 202 | 0.054 | 0.054 [0.031, 0.095] |
| test_id | 223 | 0.036 | 0.036 [0.018, 0.069] |
| test_ood_domain | 280 | 0.061 | 0.061 [0.038, 0.095] |
| hc3 | 1264 | 0.117 | 0.117 [0.101, 0.136] |
| raid | 161 | 0.491 | 0.491 [0.415, 0.567] |

## Abstention (policy chosen on val, applied unchanged elsewhere)

### Target coverage 100% (margin 0.0041)

| Split | actual coverage | selective risk | FPR among accepted humans | FNR among accepted AI |
|---|---:|---:|---:|---:|
| val | 100.0% | 0.011 | 0.054 | 0.003 |
| test_id | 100.0% | 0.012 | 0.036 | 0.008 |
| test_ood_domain | 100.0% | 0.073 | 0.061 | 0.075 |
| hc3 | 99.7% | 0.271 | 0.117 | 0.537 |
| raid | 99.8% | 0.551 | 0.491 | 0.556 |

### Target coverage 90% (margin 0.1424)

| Split | actual coverage | selective risk | FPR among accepted humans | FNR among accepted AI |
|---|---:|---:|---:|---:|
| val | 90.0% | 0.007 | 0.136 | 0.000 |
| test_id | 89.5% | 0.005 | 0.096 | 0.000 |
| test_ood_domain | 80.0% | 0.009 | 0.500 | 0.000 |
| hc3 | 26.8% | 0.213 | 0.481 | 0.000 |
| raid | 50.7% | 0.209 | 0.876 | 0.144 |

### Target coverage 75% (margin 0.8473)

| Split | actual coverage | selective risk | FPR among accepted humans | FNR among accepted AI |
|---|---:|---:|---:|---:|
| val | 75.2% | 0.000 | n/a | 0.000 |
| test_id | 75.4% | 0.000 | n/a | 0.000 |
| test_ood_domain | 51.4% | 0.000 | n/a | 0.000 |
| hc3 | 0.2% | 0.000 | n/a | 0.000 |
| raid | 9.2% | 0.000 | n/a | 0.000 |

### Target coverage 50% (margin 0.8479)

| Split | actual coverage | selective risk | FPR among accepted humans | FNR among accepted AI |
|---|---:|---:|---:|---:|
| val | 50.5% | 0.000 | n/a | 0.000 |
| test_id | 50.5% | 0.000 | n/a | 0.000 |
| test_ood_domain | 35.5% | 0.000 | n/a | 0.000 |
| hc3 | 0.0 | n/a | n/a | n/a |
| raid | 3.0% | 0.000 | n/a | 0.000 |

### Target coverage 25% (margin 0.8480)

| Split | actual coverage | selective risk | FPR among accepted humans | FNR among accepted AI |
|---|---:|---:|---:|---:|
| val | 25.2% | 0.000 | n/a | 0.000 |
| test_id | 24.1% | 0.000 | n/a | 0.000 |
| test_ood_domain | 15.0% | 0.000 | n/a | 0.000 |
| hc3 | 0.0 | n/a | n/a | n/a |
| raid | 0.4% | 0.000 | n/a | 0.000 |

## Threshold reliability

- human validation examples: **202**
- flagged at the threshold: **11** (FPR 0.054, Wilson 95% CI [0.031, 0.095])
- at a 5% budget we expect 10.1 flags; the cut is decided by examples ranked 10 and 11 of 202
- score gap between those two examples: 0.09416758641600609
- leave-one-out: threshold moves by at most **0.034323** (75.9% of the threshold), sd 0.008198
- more human validation data needed: **True**
