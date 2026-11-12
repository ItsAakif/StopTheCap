# StopTheCap

Detection of AI-generated content and evidence-based fact-checking, built as a reproducible
research artefact rather than a wrapper around someone else's API.

**Non-commercial research.** The code is Apache-2.0. Third-party text, datasets and model weights
keep their own terms - see [DATA_LICENCES.md](DATA_LICENCES.md). Nothing in this repository is
legal advice.

---

## Status

| Area | State |
|---|---|
| Dataset audit and leakage gates | 12 gates in `python -m ml.cli audit`; E1 split digest frozen |
| **E1 encoder (DistilBERT)** | **trained, evaluated, frozen** - [docs/E1-results.md](docs/E1-results.md), [reports/e1/E1_eval.json](reports/e1/E1_eval.json) |
| **E2-C1 experiment** | **completed, negative result** - in-domain 1.000, HC3 0.447, RAID 0.377; model at [itsaakif/StopTheCap-E2C1-distilbert](https://huggingface.co/itsaakif/StopTheCap-E2C1-distilbert) |
| Video detection | pipeline validated on CPU; detector not selected - [docs/video-alternatives.md](docs/video-alternatives.md) |
| Fact checker | not started |

## Headline results (E1, frozen)

| Evaluation set | ROC-AUC | Note |
|---|---:|---|
| Ghostbuster `test_id` (in-domain) | 0.999 | in-domain separation is easy; not the honest headline |
| `test_ood_domain` (news) | 0.980 | unseen domain, same corpus |
| HC3 | 0.817 | independent |
| RAID | **0.532** | 95% CI includes chance; human FPR 49.1% |

The E1 result is the reason E2-C1 exists: a detector trained on Ghostbuster does not transfer to
RAID's generator families. E1's checkpoint, configuration, threshold, split and evaluation
artefacts are frozen and are not modified by later work.

## E2-C1 experiment (negative result)

Two self-generated corpora were tested against E1's frozen baseline.

| Corpus | In-domain | HC3 | RAID | Human FPR (HC3 / RAID) |
|---|---:|---:|---:|---|
| E1 - Ghostbuster | 0.999 | 0.817 | 0.532 | - |
| E2-C1 short encyclopedic (17,708) | 1.000 | 0.367 | 0.375 | 42% / 44% |
| E2-C1 length-matched essays (2,178) | 1.000 | 0.447 | 0.377 | 88% / 88% |

**What succeeded:** the generation, provenance, split, merge and evaluation pipeline; and a
3-class generator-attribution model (95.1% accuracy, 0.84 macro-F1) that distinguishes
DeepSeek V4.1 Flash, Muse Spark 1.3 and LongCat 2.5.

**What failed:** cross-dataset generalisation. Both corpora produce perfect in-domain separation
and at-or-below-chance ranking on HC3/RAID. The shortcut gate caught the cause: word count first,
then paragraph and digit density. The essay detector is published as a research artefact, not a
usable detector.

## What exists today

```
python -m ml.cli build     # records -> splits -> datasets/manifests/COMBINED.yaml
python -m ml.cli audit     # 12 gates -> reports/audit.{md,json}
python -m pytest -q        # 398 tests
```

E2-C1 tooling:

```
python -m ml.e2c1.build_prompts_wikitext   # rebuild the 5,000-prompt inventory (network)
python -m ml.e2c1.build_splits             # prompt-level 80/10/10 split (offline)
python -m ml.e2c1.generate --help          # generation runner; full run needs --authorise-full-run
```

Requirements: Python 3.12+ with `numpy`, `pydantic`, `PyYAML`, `scikit-learn`, `pytest`. Training and
generation additionally need `torch`/`transformers` and run on Colab or Kaggle, never locally.

## Dataset roles

| Dataset | Licence | Role |
|---|---|---|
| Ghostbuster | CC BY 3.0 | training source (E1) |
| RAID | MIT | evaluation only |
| HC3 | CC BY-SA 4.0 | evaluation only |
| Wikipedia (E2-C1 prompts) | CC BY-SA 4.0 | prompt inventory for generated text |
| GenVidBench | CC BY-NC 4.0 | video inventory (media not downloaded) |
| M4 / M4GT-Bench | none declared | **excluded** |
| DFDC | unverified | **blocked** |

Raw and canonical records of third-party datasets are **not** distributed. Manifests, counts,
digests and evaluation results are. HC3 and RAID can never enter a training path; a test asserts it.

## Design rules that are enforced, not aspirational

- **Original labels are never overwritten.** `original_label` is preserved alongside `label`.
- **Splits are group-based and deterministic.** `sha256(group_id + salt)`, independent of row order;
  the digest is committed and re-verified.
- **Evaluation data is quarantined by construction.** RAID and HC3 are evaluation-only.
- **No licence, no training.** `tests/test_manifests.py` fails if a source lacks licence evidence.
- **Absence of a licence is not permission.** M4/M4GT-Bench is excluded on that basis.
- **The audit must be able to fail.** Negative tests assert it catches a one-character leak.
- **E2-C1 splits are assigned per prompt, before generation.** No prompt or near-duplicate family
  can cross partitions; no E2-C1 sample can reach E1's `test_id`.

## Known limitations

1. Ghostbuster's machine text is GPT-3/Claude-era (2023). A model trained on it need not detect
   2025-26 generators. This is disclosed in any result.
2. The training split is imbalanced 5.9:1 (machine:human). Human false-positive rate is reported
   from the human subset alone.
3. In-domain separation is easy (character-only diagnostic AUC 0.994). Independent RAID numbers are
   the honest headline.
4. **E1 RAID ROC-AUC 0.532 is chance.** No OOD safeguard is claimed; confidence-based abstention was
   tested and failed under domain shift.
5. **E2-C1 is a documented negative result.** Two corpora were generated and trained; both fail
   cross-dataset transfer (HC3 0.367/0.447, RAID 0.375/0.377). The essay detector is published on
   Hugging Face as a research artefact, not a usable detector.
6. The E2-C1 prompt inventory is topically skewed (`other` 44.7%, `medicine_health` 0.5%); see
   [docs/e2/prompt-provenance.md](docs/e2/prompt-provenance.md).

## Repository layout

```
ml/            ingestion, dedup, splits, audit
ml/e2c1/       prompt inventory, families, prompt-level split, generation contract
datasets/      manifests (committed) + raw/canonical/splits (gitignored, not distributed)
tests/         512 tests, including negative tests for every gate
docs/          results, method, licence audit, provenance
reports/       generated audit output and E1 metrics
notebooks/     Colab training and smoke-generation notebooks
scripts/       private ZIP builder for Colab
.github/       CI: tests on 3.12/3.13 + guards against committed data/secrets/weights
```

## Licence

- **Code:** Apache-2.0 - [LICENSE](LICENSE).
- **Wikipedia prompt inventory:** CC BY-SA 4.0, attribution on every record.
- **Third-party datasets, model weights and generated text:** not distributed; their own terms apply.

The layering, and the reason generated text is **not** covered by the weight licences, is recorded in
[DATA_LICENCES.md](DATA_LICENCES.md) and [docs/e2/model-licence-audit.md](docs/e2/model-licence-audit.md).

## Next

E2-C1 is closed as a negative result. The pipeline, corpora, attribution experiment and trained
artefact are published for reproduction.

1. Reuse the pipeline with human text from the target domains and multiple generator families;
   that is the missing ingredient, not more training on the same composition.
2. Video: resolve whether a bounded real-video sample is obtainable under a permissive licence
   before any model work.
