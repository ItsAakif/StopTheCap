# Dataset audit

**Overall: PASS**

## Gates

| Gate | Result | Detail |
|---|---|---|
| `unique_sample_ids` | PASS | 52100 unique ids |
| `required_provenance` | PASS | all 49592 active records complete |
| `no_empty_text` | PASS | no empty text among active records |
| `labels_valid` | PASS | labels mapped |
| `no_cross_split_duplicates` | PASS | 0 across splits |
| `no_group_split_overlap` | PASS | 31642 groups, none spanning splits |
| `eval_only_not_trained` | PASS | eval-only datasets quarantined |
| `shortcut_single_feature` | PASS | max single-feature AUC=0.672 |
| `shortcut_length_only` | PASS | AUC=0.539 <= 0.60 |
| `split_deterministic` | PASS | recomputed digest matches manifest |
| `video_no_group_overlap` | PASS | 0 video groups span splits |
| `video_media_absent_marked` | PASS | all video records correctly declare media_present=false |

## Notes

- pass unique_sample_ids: 52100 unique ids
- pass required_provenance: all 49592 active records complete
- pass no_empty_text: no empty text among active records
- pass labels_valid: labels mapped
- pass no_cross_split_duplicates: 0 across splits
- pass no_group_split_overlap: 31642 groups, none spanning splits
- pass eval_only_not_trained: eval-only datasets quarantined
- pass shortcut_single_feature: max single-feature AUC=0.672
- pass shortcut_length_only: AUC=0.539 <= 0.60
- diagnostic character_only AUC=0.994 (not a gate; see shortcut.py for why an absolute threshold is inappropriate)
- pass split_deterministic: recomputed digest matches manifest
- diagnostic: At least one dataset pair is separated at AUC >= 0.95 from text alone. A model trained across these sources can identify the corpus rather than the authorship. Report metrics per source and never present a pooled number as if the sources were exchangeable.
- diagnostic: At least one Ghostbuster variant is separable from human text at AUC >= 0.99. Per-variant results must be reported so a strong pooled number cannot hide a variant-specific shortcut.
- pass video_no_group_overlap: 0 video groups span splits
- pass video_media_absent_marked: all video records correctly declare media_present=false

## Records by split

| Split | Records | Groups |
|---|---:|---:|
| excluded | 2508 | 0 |
| quarantine | 28547 | 27599 |
| test_id | 1518 | 421 |
| test_ood_domain | 6976 | 50 |
| train | 11150 | 3164 |
| val | 1401 | 408 |

## Records by dataset (label distribution)

| Dataset | label 0 | label 1 | label 2 |
|---|---:|---:|---:|
| genvidbench | 4724 | 15276 | 0 |
| ghostbuster | 3020 | 18025 | 0 |
| hc3 | 4157 | 2418 | 0 |
| raid | 161 | 1811 | 0 |

## Records by generator

| Generator | Count by label |
|---|---|
| chatgpt | {'1': 2574} |
| claude | {'1': 2993} |
| cogvideo | {'1': 1952} |
| cohere | {'1': 100} |
| cohere-chat | {'1': 100} |
| gpt | {'1': 3040} |
| gpt2 | {'1': 100} |
| gpt3 | {'1': 100} |
| gpt4 | {'1': 143} |
| gpt_prompt1 | {'1': 2998} |
| gpt_prompt2 | {'1': 2997} |
| gpt_semantic | {'1': 2999} |
| gpt_writing | {'1': 2998} |
| human | {'0': 12062} |
| llama-chat | {'1': 100} |
| mistral | {'1': 200} |
| mistral-chat | {'1': 39} |
| mora | {'1': 1951} |
| mpt | {'1': 623} |
| mpt-chat | {'1': 150} |
| ms | {'1': 1902} |
| musev | {'1': 1883} |
| pika | {'1': 1902} |
| svd | {'1': 1883} |
| t2vz | {'1': 1901} |
| vc2 | {'1': 1902} |

## Shortcut probes

| Probe | AUC | Gate | n_test | neg | pos |
|---|---:|---|---:|---:|---:|
| single_feature_artifact | 0.672 | PASS | 1518 | 223 | 1295 |
| length_only | 0.539 | PASS | 1518 | 223 | 1295 |
| character_only | 0.994 | **FAIL** | 1518 | 223 | 1295 |

_Gate: AUC <= 0.60. A high value means the corpus leaks typography or verbosity rather than style._

## Duplicates

- exact pairs: 23
- near pairs (Jaccard >= 0.85): 22
- cross-split pairs: 0

**Paired groups (human+machine of the same source): 2044**

## Video

- inventory records: 20000
- label counts: {'1': 15276, '0': 4724}
- group leaks: 0
- identity leaks: 0

## Diagnostics (non-gating)

### Per source

| Source | Basis | n train | n test | char AUC | length AUC | Note |
|---|---|---:|---:|---:|---:|---|
| ghostbuster | project_split | 11150 | 1518 | 0.996 | 0.539 |  |
| hc3 | internal_diagnostic | 4622 | 1953 | 0.997 | 0.646 |  |
| raid | internal_diagnostic | 1388 | 584 | 0.974 | 0.822 |  |

### Per Ghostbuster variant (human vs variant)

| Variant | n train (machine) | n test (machine) | n test (human) | char AUC |
|---|---:|---:|---:|---:|
| claude | 1578 | 203 | 223 | 0.988 |
| gpt | 1616 | 221 | 223 | 0.997 |
| gpt_prompt1 | 1585 | 218 | 223 | 1.000 |
| gpt_prompt2 | 1584 | 217 | 223 | 0.998 |
| gpt_semantic | 1586 | 218 | 223 | 1.000 |
| gpt_writing | 1584 | 218 | 223 | 1.000 |

### Dataset fingerprints (can text reveal the source?)

| Pair | n A | n B | AUC |
|---|---:|---:|---:|
| ghostbuster vs hc3 | 4000 | 4000 | 1.000 |
| ghostbuster vs raid | 4000 | 1972 | 0.999 |
| hc3 vs raid | 4000 | 1972 | 0.997 |

**Diagnostic notes**

- At least one dataset pair is separated at AUC >= 0.95 from text alone. A model trained across these sources can identify the corpus rather than the authorship. Report metrics per source and never present a pooled number as if the sources were exchangeable.
- At least one Ghostbuster variant is separable from human text at AUC >= 0.99. Per-variant results must be reported so a strong pooled number cannot hide a variant-specific shortcut.

_These diagnostics are reported, not enforced. They exist so a pooled score cannot hide which source or variant produced it._


## Reproduce

```
python -m ml.cli build
python -m ml.cli audit
```
