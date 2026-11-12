# StopTheCap

[![ci](https://github.com/ItsAakif/StopTheCap/actions/workflows/ci.yml/badge.svg)](https://github.com/ItsAakif/StopTheCap/actions/workflows/ci.yml)
[![model](https://img.shields.io/badge/model-StopTheCap--E2C1--distilbert-yellow)](https://huggingface.co/itsaakif/StopTheCap-E2C1-distilbert)

Detection of AI-generated content, built as a reproducible research artefact rather than a
wrapper around someone else's API. The fact-checking component is not started.

**Non-commercial research.** The code is Apache-2.0. Third-party text, datasets and model weights
keep their own terms - see [DATA_LICENCES.md](DATA_LICENCES.md). Nothing in this repository is
legal advice.

---

## Summary

- **E1** (DistilBERT on Ghostbuster) reaches **0.999 in-domain** but only **0.532 ROC-AUC on RAID** - chance.
- **E2-C1** tested two self-generated corpora. Both reach **1.000 in-domain** and **0.367-0.447 on HC3** / **0.375-0.377 on RAID**, with human false-positive rates of **42-88%**.
- The corpora were separable by **shortcut features** (word count first, then paragraph and digit density), so the model learned the corpus, not authorship. The failing gate is recorded in the manifests.
- **One positive result:** a 3-class generator-attribution model separates DeepSeek / Muse Spark / LongCat at **95.1% accuracy, 0.84 macro-F1**.
- The trained essay detector is published on Hugging Face as a **research artefact, not a usable detector**.

## Status

| Area | State |
|---|---|
| Dataset audit and leakage gates | 12 gates in `python -m ml.cli audit`; E1 split digest frozen |
| **E1 encoder (DistilBERT)** | **trained, evaluated, frozen** - [docs/E1-results.md](docs/E1-results.md), [reports/e1/E1_eval.json](reports/e1/E1_eval.json) |
| **E2-C1 experiment** | **completed, negative result** - in-domain 1.000, HC3 0.447, RAID 0.377; model at [itsaakif/StopTheCap-E2C1-distilbert](https://huggingface.co/itsaakif/StopTheCap-E2C1-distilbert) |
| Video detection | pipeline validated on CPU; detector not selected - [docs/video-alternatives.md](docs/video-alternatives.md) |
| Fact checker | not started |

## E1 baseline (frozen)

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
| E2-C1 short encyclopedic (23,774) | 1.000 | 0.367 | 0.375 | 42% / 44% |
| E2-C1 length-matched essays (2,178) | 1.000 | 0.447 | 0.377 | 88% / 88% |

**What succeeded:** the generation, provenance, split, merge and evaluation pipeline; and a
3-class generator-attribution model (95.1% accuracy, 0.84 macro-F1) that distinguishes
DeepSeek V4.1 Flash, Muse Spark 1.3 and LongCat 2.5.

**What failed:** cross-dataset generalisation. Both corpora produce perfect in-domain separation
and at-or-below-chance ranking on HC3/RAID. The shortcut gate caught the cause: word count first,
then paragraph and digit density. The essay detector is published as a research artefact, not a
usable detector.

## Quick start

```bash
python -m pip install "numpy>=2.0" "pydantic>=2.7" "PyYAML>=6.0" "scikit-learn>=1.4" "pytest>=8.0"

python -m ml.cli build     # records -> splits -> datasets/manifests/COMBINED.yaml
python -m ml.cli audit     # 12 gates -> reports/audit.{md,json}
python -m pytest -q        # 520 tests
```

Requirements: Python 3.12+ with `numpy`, `pydantic`, `PyYAML`, `scikit-learn`, `pytest` and `Pillow`; video decoding additionally needs `ffmpeg` on `PATH`.
Training, generation and export additionally need `torch`/`transformers` and run on Colab or Kaggle, never locally.

## Reproduce E2-C1

The canonical corpora are not distributed. The builders regenerate them from the committed prompt
inventories, manifests and the third-party records; generation itself calls external APIs and needs
credentials, so it is reproducible only up to the generation step.

```bash
python -m ml.e2c1.build_prompts_wikitext   # 5,000-prompt inventory (network)
python -m ml.e2c1.build_splits             # prompt-level 80/10/10 split
python -m ml.e2c1.generate --help          # generation runner; full run needs --authorise-full-run
python -m ml.e2c1.merge_api_shards --help  # merge API generation shards
python -m ml.e2c1.build_training_corpus    # 23,774-row short-form corpus
python -m ml.e2c1.build_essay_corpus       # 2,178-row length-matched essay corpus
python -m ml.train.train_e2c1 --corpus datasets/canonical/e2c1_essays.jsonl --manifest datasets/manifests/e2_c1_essays.json --device cuda --out reports/E2C1_essays
python -m ml.train.export_checkpoint --checkpoint reports/E2C1_essays/checkpoints/last.pt --out reports/E2C1_essays/export
```

The published essay detector is [itsaakif/StopTheCap-E2C1-distilbert](https://huggingface.co/itsaakif/StopTheCap-E2C1-distilbert).
Generator attribution is `ml/train/train_attribution.py` with `notebooks/colab_e2c1_attribution.ipynb`.

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
ml/e2c1/       prompt inventory, families, prompt-level split, generation, corpus builders
ml/train/      trainers, calibration, export
datasets/      manifests (committed) + raw/canonical/splits (gitignored, not distributed)
tests/         520 tests, including negative tests for every gate
docs/          results, method, licence audit, provenance
reports/       generated audit output and E1 metrics
notebooks/     Colab generation, training and attribution notebooks
scripts/       private ZIP builder for Colab
.github/       CI: tests on 3.12/3.13 + guards against committed data/secrets/weights
```

## Citation

If you use this artefact, cite the negative result:

```bibtex
@misc{stopthecap2026e2c1,
  title  = {StopTheCap E2-C1: length-matched API-generated essays do not transfer to HC3 or RAID},
  author = {ItsAakif},
  year   = {2026},
  note   = {Negative result; model at huggingface.co/itsaakif/StopTheCap-E2C1-distilbert}
}
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
