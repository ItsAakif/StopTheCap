# DATASET_REGISTRY.md

**Canonical dataset inventory for StopTheCap.**
**Last verified:** 2026-11-02 · **Verified against:** primary repository files and official licence
documents, not against this project's earlier notes.

This document supersedes the dataset tables in other docs. Other documents should link here rather
than restate licences, because duplicated licence text drifts and then contradicts itself.

---

## 0. How to read this

**Evidence labels**

| Label | Meaning |
|---|---|
| **VERIFIED** | I opened the licence file or official terms page and read it |
| **REPORTED** | Quoted from a secondary source or a search-layer rendering of an official page |
| **UNVERIFIED** | No licence document found; absence of a licence is **not** permission |
| **BLOCKED** | Terms exist but are behind authentication / a document I could not read |

**Integration status**

| Status | Meaning |
|---|---|
| **Integrated** | Actually flowing through the repository pipeline, producing canonical records |
| **Downloaded** | Data present locally; not necessarily integrated |
| **Manifest only** | Metadata/labels present; media deliberately not downloaded |
| **Evaluation only** | Held out from training by design and enforced in code |
| **Research candidate** | Investigated; not integrated, not downloaded |
| **Blocked** | Access or terms currently prevent use |
| **Excluded** | Deliberately not used, with a recorded reason |

**Permission matrix columns** (all assessed separately — a licence that allows one rarely allows all):

1. Download for research · 2. Frame/text extraction & preprocessing · 3. Create derived datasets ·
4. Train models · 5. Retain features/weights · 6. Publish results · 7. Redistribute original or weights

✅ permitted · ⚠️ unverified or conditional · ⛔ not permitted · — not applicable

---

## 0.1 Policy: permissions are documented, not negotiated

**No experiment waits on a dataset maintainer.** We do not send permission requests, do not wait for
replies, and do not treat developer contact as a prerequisite for any work.

The operating rule is:

> Use a dataset for the operations its **documented** terms clearly permit. Where the terms are
> silent or ambiguous for a specific operation, record that limitation and **route around it** with a
> dataset whose terms do permit the operation.

Consequences:

- Academic-use-only datasets are **not** excluded from the inventory or from research use. Vript,
  Vript-RR, HD-VG-130M, GenVidBench and DFDC all remain research inventory entries.
- Where an operation is unclear — notably *publishing derived weights* from Vript-derived media, or
  *frame extraction* under Vript's "no modification" clause — we do not perform that operation on
  that data. We use a source that clearly allows it instead.
- Attribution alone never grants permission. A licence that is silent is treated as **not permitted
  for that operation**, not as permitted.
- Nothing in this repository claims a permission that was not read in a primary document.

**Currently unblocked path:** Pexels (real, permissive) + Wan 2.1 outputs (Apache-2.0). Neither
requires any contact, and both clearly permit research use, processing and weight publication.

| Dataset | Licence | Evidence | Status | Records in repo |
|---|---|---|---|---|
| **Ghostbuster** | **CC BY 3.0** | VERIFIED — `ghostbuster-data/LICENSE` | **Integrated, downloaded** | 22,100 |
| **RAID** | **MIT** | VERIFIED — HF dataset card | **Integrated, evaluation only** | 2,000 sample |
| **HC3** | **CC BY-SA 4.0** | VERIFIED — HF dataset card | **Integrated, evaluation only** | 8,000 of 85,431 |
| **M4** | **None declared** | VERIFIED absence — repo tree has no `LICENSE` | Research candidate → **Excluded** | 0 |
| **M4GT-Bench** | **None declared** | VERIFIED absence — repo tree has no `LICENSE` | Research candidate → **Excluded** | 0 |
| **SemEval-2024 Task 8** | **Unverified** | BLOCKED — official task site returned 404; no canonical repo located | Research candidate | 0 |
| **ai-text-detection-pile** | **MIT** (uploader claim) | VERIFIED card; provenance unverified | **Audited: usable only after decontamination** - reproduces HC3 verbatim (sharded by class) | 0 |
| **AI-and-Human-Generated-Text** | **MIT** (upstream corroborated) | VERIFIED - upstream `AI-GA/LICENSE` | **Audited: clean overlap, but human text is CORD-19 and AI text is templated** | 0 |
| **ArguGPT** | **`cc` - variant unspecified** | Ambiguous licence = not permitted by policy | Excluded (E2) | 0 |

### 1.1 Ghostbuster — primary training corpus

- URL: https://github.com/vivek3141/ghostbuster-data
- Licence: **CC BY 3.0** (`LICENSE` in the data repo, 19,467 bytes; verified twice, 2026-10-29 and 2026-11-02)
- Pinned version: `86ebd72590556a81622986fab736ab9227a948af`
- Local: `datasets/raw/ghostbuster` — **53,340 files, 158.7 MB** (sparse clone excluding `logprobs`)
- Applies to: the text corpus. Attribution required.
- Notes: machine text is GPT-3/Claude-era (2023). Disclosed in every model card.

### 1.2 RAID — independent evaluation

- URL: https://huggingface.co/datasets/liamdugan/raid
- Licence: **MIT** (HF card tag)
- Local: `datasets/raw/raid` — **2 files, 4.5 MB**, a 2,000-row systematic random sample
- **Evaluation only.** Forced to `QUARANTINE`; `tests/test_manifests.py` and the audit enforce it.
- The official test split has **2 columns and no text** — it is submission-only.

### 1.3 HC3 — independent evaluation

- URL: https://huggingface.co/datasets/Hello-SimpleAI/HC3
- Licence: **CC BY-SA 4.0** — ShareAlike applies to any redistributed derivative
- Local: `datasets/raw/hc3` — **5 files, 69.7 MB** (5 subset JSONL; `all.jsonl` omitted as a duplicate union)
- **Evaluation only**, quarantined. Single generator (ChatGPT, 2023).

### 1.4 M4 / M4GT-Bench / SemEval-2024 Task 8

These are **one lineage, not three datasets**. M4's README states the corpus is extended by SemEval
2024 Task 8, and M4GT-Bench *is* that benchmark. Counting them separately triple-counts samples.

- M4/M4GT-Bench: **no `LICENSE` file** and no terms in the README (verified by tree listing).
  Absence of a licence is not permission → **Excluded**, `datasets/manifests/m4gt-bench.yaml`
  (`enabled: false`).
- SemEval-2024 Task 8: the official task page returned **404** and no canonical repository was
  located under the names tried. Mirror copies on the Hub exist but a mirror's licence tag is the
  uploader's claim, not the task's. **Unverified.**
- Consequence: `text_generator_attribution` and `text_mixed_boundary` tasks are **not implemented**.

---

## 2. AI-generated video datasets

| Dataset | Licence | Evidence | Status |
|---|---|---|---|
| **GenVidBench** | **CC BY-NC 4.0** (data); repo `LICENSE` is Apache-2.0 **for code only** | VERIFIED — official site `## License`; HF card has **no** licence tag | **Manifest only** |
| **Vript** | **Academic research only; no distribution; no modification** | VERIFIED — HF card `## License` | **Manifest only** (captions held; media blocked) |
| **Vript-RR** | Same Vript terms | VERIFIED — HF card README | Research candidate |
| **Vript-Hard** | — | **NOT FOUND** on Hugging Face under that name | Unverified / does not exist as named |
| **Vript-ERO** | **None declared** | VERIFIED absence — README is 156 bytes, front-matter only | **Excluded** |
| **HD-VG-130M** | **Custom "HDVG LICENSE AGREEMENT.pdf"** — not an OSI licence | BLOCKED — PDF not read | **Blocked** |
| **VidProM** | **Mixed**: Pika & ModelScope CC BY-NC 4.0 · **VideoCraft2 Apache-2.0** · T2V-Zero CreativeML Open RAIL-M | VERIFIED — HF card README | Research candidate |
| **VideoCraft2** | Code: **NOASSERTION**; released videos stated Apache-2.0 via VidProM | VERIFIED (both) | Research candidate — verify scope |
| **DFDC** | Terms require authenticated Kaggle acceptance | BLOCKED | **Blocked** |
| **OpenAI Sora ZIP** | Inside GenVidBench ⇒ wrapper **CC BY-NC 4.0**; upstream Sora output terms **not documented anywhere in the release** | Partially VERIFIED | **Not used** — terms unclear for our operations |
| **Pexels Videos** | **Pexels custom licence** — free use, **modification allowed**, no attribution; cannot redistribute as stock media | VERIFIED — licence page read in full | Approved source, **not downloaded** |
| **Wan 2.1 T2V 1.3B** | **Apache-2.0** | VERIFIED — `LICENSE.txt` in repo, 11,357 bytes | Generator, **not downloaded** |

### 2.1 The licence trap this registry exists to prevent

Three distinct licences attach to the GenVidBench stack, and they are **not interchangeable**:

| Layer | Licence | Consequence |
|---|---|---|
| GitHub repository **code** | **Apache-2.0** (Open-MMLab copyright) | code reusable |
| **Benchmark data** | **CC BY-NC 4.0** (site only; HF untagged) | non-commercial research |
| **Underlying real media** (Vript) | **academic-only, no redistribution, no modification** | stricter than the wrapper |

A benchmark's wrapper licence does **not** upgrade the terms of the media inside it. Treating
CC BY-NC 4.0 as if it covered Vript clips would be wrong.

### 2.2 Vript — the current blocker

Verbatim from the card:
> "Any content from Vript/Vript-Bench dataset and Vriptor model is available for academic research
> purposes only. You agree not to reproduce, duplicate, copy, trade, or exploit for any commercial
> purposes."
> "Without the permission of the copyright owner, you are not allowed to perform any form of
> broadcasting, **modification** or any other similar behavior to the data set content."

- **"No modification"** plausibly covers frame extraction and re-encoding → preprocessing unverified.
- **Derived weights** are not addressed → treat as **not permitted** until clarified.
- Permission request drafted: `docs/video-permission-request.md`. **Not yet sent.**

### 2.3 Vript-Hard / Vript-RR

- `Mutonix/Vript-Hard` — **not found**. Do not cite it as an available dataset.
- `Mutonix/Vript-RR` — exists (3.5 GB), and its README carries the **same academic-only terms** as
  Vript. It is not a separate licence.

### 2.4 HD-VG-130M

Repo `daooshee/HD-VG-130M` contains **`HDVG LICENSE AGREEMENT.pdf`** — a bespoke agreement, not an
OSI/SPDX licence (GitHub reports no SPDX id). It is the basis Vript's licence says it was "modified
from". **Not read. Pair2's real class stays blocked.**

### 2.5 VidProM / VideoCraft2

The VidProM card distinguishes licences **per generator**, which is unusual and useful:

| Subset | Licence |
|---|---|
| Pika prompts & videos | CC BY-NC 4.0 |
| ModelScope videos | CC BY-NC 4.0 |
| **VideoCraft2 videos** | **Apache-2.0** |
| Text2Video-Zero | CreativeML Open RAIL-M |

The **VideoCrafter code repository reports `NOASSERTION`** — so the Apache-2.0 claim applies to the
*released video material as distributed via VidProM*, not to the code. Do not generalise it.

Access problem: **30 tar shards of ~34 GB per generator**. Licence ideal, access impractical.

### 2.6 DFDC

Requires accepting competition rules inside an authenticated Kaggle session. I could not read the
terms. **The `video_manipulated` task has no licensed source.** Do not substitute generated video.

---

## 3. Permission matrix

Assessed per activity, because "academic use permitted" does not imply "weights publishable".

| Dataset | 1 Download | 2 Extract | 3 Derived data | 4 Train | 5 Retain weights | 6 Publish results | 7 Redistribute |
|---|---|---|---|---|---|---|---|
| Ghostbuster | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ (attrib.) |
| RAID | ✅ | ✅ | ⚠️ eval only | ⛔ by policy | ✅ | ✅ | ✅ (attrib.) |
| HC3 | ✅ | ✅ | ⚠️ ShareAlike | ⛔ by policy | ✅ | ✅ | ⚠️ ShareAlike |
| M4 / M4GT-Bench | ⚠️ | ⚠️ | ⛔ | ⛔ | ⛔ | ⚠️ | ⛔ |
| SemEval-2024 T8 | ⚠️ | ⚠️ | ⚠️ | ⚠️ | ⚠️ | ⚠️ | ⛔ |
| GenVidBench (data) | ✅ | ⚠️ | ⚠️ | ✅ research | ⚠️ unstated | ✅ | ⚠️ NC only |
| Vript | ✅ academic | ⚠️ "no modification" | ⛔ | ✅ academic | ⛔ unstated | ✅ | ⛔ |
| Vript-RR | ✅ academic | ⚠️ | ⛔ | ✅ academic | ⛔ | ✅ | ⛔ |
| Vript-ERO | ⛔ | ⛔ | ⛔ | ⛔ | ⛔ | ⛔ | ⛔ |
| HD-VG-130M | ⚠️ | ⚠️ | ⚠️ | ⚠️ | ⚠️ | ⚠️ | ⛔ |
| VidProM (Pika/MS) | ✅ | ✅ | ⚠️ | ✅ research | ⚠️ | ✅ | ⚠️ NC |
| VidProM (**VideoCraft2**) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| VideoCraft2 code | ✅ | — | — | — | — | ✅ | ⚠️ NOASSERTION |
| DFDC | ⚠️ | ⚠️ | ⚠️ | ⚠️ | ⚠️ | ⚠️ | ⛔ |
| Sora ZIP | ✅ | ⚠️ | ⚠️ | ⚠️ | ⚠️ | ⚠️ | ⚠️ NC |
| Pexels | ✅ | ✅ | ⛔ republish imagery; ✅ features | ✅ | ✅ | ✅ | ⛔ as stock |
| Wan 2.1 (outputs) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

**Reading the table:** a dataset can be perfectly usable for research (columns 1–4) and still not
publishable (columns 5, 7). That distinction is the reason the video weights are currently withheld.

---

## 4. What is actually downloaded (verified on disk, 2026-11-02)

| Path | Files | Size | Contents | Media? |
|---|---:|---:|---|---|
| `datasets/raw/ghostbuster` | 53,340 | 158.7 MB | text corpus, sparse clone | n/a (text) |
| `datasets/raw/hc3` | 5 | 69.7 MB | 5 subset JSONL | n/a |
| `datasets/raw/genvidbench` | 4 | 16.4 MB | `Pair1_labels.txt`, `Pair2_labels.txt`, `Vript_20k_classes.txt`, verify list | **no media** |
| `datasets/raw/raid` | 2 | 4.5 MB | cached API sample + meta | **no media** |
| `datasets/raw/vript` | 2 | 529.6 MB | long + short **captions** (annotations only) | **no media** |

**No video, image or archive media is present anywhere under `datasets/raw`.** Verified by extension
scan. A single 11.4 MB Vript probe clip used during the access investigation was deleted on
2026-11-02 to comply with the media block.

Cloud measurement: `ml/cloud/probe.py` plus `notebooks/colab_measure.ipynb` and `notebooks/kaggle_measure.ipynb`; output goes to `reports/cloud/`. Nothing has been measured on a cloud GPU yet.

Generated artefacts (gitignored): `datasets/canonical/*.jsonl` (137 MB),
`datasets/splits/assignments.jsonl` (7.7 MB), `datasets/manifests/COMBINED.yaml`.

**Split manifest:** `split_digest 932d20f1dca64c21537bed89739c633ace1b627c1a25f7d78c45a17915cf27a2`,
salt `stopthecap-2026-10-29`. Records: train 11,150 · val 1,401 · test_id 1,518 · test_ood_domain
6,976 · quarantine 28,547 · excluded 2,508.

---

## 5. Model inventory

**One model has been trained: the E1 encoder (2026-11-05).** Its weights live outside the repository (753 MB, on Drive); only its evaluation artefacts are committed.

| # | Model | Type | Source / licence | Task | Trained? | Config / code | Eval status |
|---|---|---|---|---|---|---|---|
| 1 | **TF-IDF + Logistic Regression (word 1–2 gram)** | classical baseline | scikit-learn, BSD-3 | text: human vs machine | **Fitted** (in-process, not persisted) | `ml/baselines/tfidf.py` | **Measured** — threshold 0.4391, vocab 200,000 |
| 2 | **TF-IDF + LR (char 3–5 gram)** | classical baseline | scikit-learn, BSD-3 | same | **Fitted** | same | **Measured** — threshold 0.4066, vocab 165,959 |
| 3 | **distilbert-base-cased** | encoder | HF; **Apache-2.0** (verified) | text classification | **Trained (E1, 2026-11-05)** | `ml/train/encoder_config.yaml`, `ml/train/train_encoder.py` | measured - see `docs/E1-results.md` |
| 4 | **Wan2.1-T2V-1.3B** | **generator, not a detector** | HF; **Apache-2.0** (verified) | generate synthetic video | not run | — | none |
| 5 | **CLIP ViT-B/32** (frozen backbone) | feature extractor, candidate | HF `openai/clip-vit-base-patch32`; **MIT** per `openai/CLIP` repo (HF card has **no** licence tag) | embed frames for the video baseline | **not run** | `ml/video/pipeline.py` (`FeatureFn` slot) | none |
| 6 | **Video detector** | - | - | - | - | - | **No architecture implemented**; recommendation is frozen CLIP + logistic regression |

### 5.1 Baseline numbers (measured, preserved)

| Split | ROC-AUC | Human FPR @ frozen threshold |
|---|---:|---|
| `test_id` (in-domain) | 0.996 | 4.5% |
| `hc3` (independent) | 0.652 | 15.7% |
| `raid` (independent) | **0.480** | **47.2%** |

Caveats that must travel with these: `raid` has only 161 negatives (interval 0.396–0.549, marked
indicative); `test_id` has 223; PR-AUC on RAID is inflated by a 92% positive base rate. **Do not
describe this baseline as production-ready, and do not call 0.996 general detection performance.**

### 5.2 Encoder readiness

`ml/train/train_encoder.py` **executes end to end** — verified by a CPU smoke run on 2026-11-03
(`reports/runs/E1_smoke.json`): 30 optimiser steps at 128 tokens on a 2,000-row stratified subset,
then threshold selection and a `test_id` evaluation pass.

- Class weights computed from the train split: human **3.448**, machine **0.585**.
- Smoke evaluation: `test_id` ROC-AUC 0.802, human FPR 7.0% on 200 negatives.
- **These numbers must not be cited as model performance.** The run used 128-token truncation, a
  subset, and 30 steps. The log's own status field reads "SMOKE — NOT a trained model".
- Full E1 (512 tokens, 11,150 rows, 3 epochs) still requires a GPU session.
- Evaluation harness `ml/train/evaluate_encoder.py` was verified by scoring the smoke checkpoint across all five splits (`reports/E1_eval_smoke.*`). It reproduced the smoke run's subset AUC exactly (0.8023 vs 0.8023), so the harness agrees with the training code. On the full imbalanced splits that barely-trained checkpoint scores 0.583 - near chance, as 30 steps should.
- **E1 has since been trained and evaluated.** Real results: `test_id` AUC 0.999, HC3 0.817, RAID 0.532 (CI includes chance). Full breakdown in `docs/E1-results.md`. The smoke numbers above remain smoke numbers.

### 5.3 Video detector — none selected

Stated plainly because it is easy to misread the inventory: **there is no video detection model in
this repository.** Wan 2.1 is a generator used to *create* the synthetic class, not to detect it.

**Recommended first baseline (recommendation only — not implemented):**

> Frozen **CLIP ViT-B/32** image encoder → mean-pool 8–16 uniformly sampled frames → logistic
> regression on the pooled embedding.
>
> Why: no video training required, runs in minutes on a T4, ~350 MB of weights, and gives a
> reproducible reference point. If a frozen probe cannot separate the classes, fine-tuning will not
> rescue it — and the three mandatory content-control probes will show whether any signal is
> content rather than synthesis.

Alternatives if CLIP is unavailable: a frozen ImageNet ResNet-50 with the same pooling, or a
VideoMAE-style video transformer (heavier; only justified after the frozen baseline shows signal).

---

## 6. Unresolved items — none of these block development

These are **documented limitations**, not dependencies. Each is routed around by using a different
source; none is waiting on a reply.

| # | Limitation | Effect | How we route around it |
|---|---|---|---|
| 1 | Vript: weights, frame extraction, results publication unstated | No Vript-derived model is published, and frame extraction from Vript media is not performed | use Pexels + Wan 2.1 for anything that must be publishable |
| 2 | GenVidBench: CC BY-NC does not state whether it covers trained weights | No GenVidBench-trained weights are published | same |
| 3 | HD-VG-130M: `HDVG LICENSE AGREEMENT.pdf` unread | Pair2 real class unused | use Pexels for the real class |
| 4 | DFDC terms unread (authenticated) | `video_manipulated` has no licensed source | feature stays unshipped; not substituted with generated video |
| 5 | SemEval-2024 T8 licence + canonical URL not located | mixed-boundary / attribution tasks not implemented | out of scope until located |
| 6 | VideoCraft2 Apache-2.0 scope (videos vs code) | not used yet | use Wan outputs, which are unambiguous |
| 7 | **Sora ZIP: upstream output terms undocumented** | **not used** | **Wan 2.1 outputs (Apache-2.0) instead** |
| 8 | distilbert-base-cased licence not read this pass | encoder release pending | read the model card before publishing weights |
| 9 | Vript-Hard not found under that name | cannot be cited | omit; do not invent terms |

**Explicitly not a blocker:** maintainer contact. See §0.1.

---

## 7. Maintenance rules

1. **Evidence URL and date are mandatory** for any licence claim. `tests/test_manifests.py` fails the
   build if an enabled manifest lacks one.
2. **Never** infer data permission from a code licence.
3. **Never** infer media permission from a benchmark wrapper.
4. **Never** mark a dataset `Integrated` because it appears here — integration means canonical records
   exist and the audit passes.
5. **Unverified stays unverified.** A dataset available on Hugging Face is not thereby licensed.
6. Status changes must update both this file and the matching YAML in `datasets/manifests/`.
