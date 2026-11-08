# E2-C1 model licence audit

**Audit date:** 2026-11-06
**Scope:** the two proposed generators for the E2-C1 self-generated corpus.
**Machine-readable registry:** `datasets/manifests/e2_c1_licences.json`
**Rule applied:** a licence is recorded only where its actual text establishes it. Uploader tags,
model-card marketing copy and recollection of a related release are not evidence.

---

## 0. Correction of the earlier claim

An earlier proposal described **both** generators as Apache-2.0. That was wrong for Llama.

| Claim | Status |
|---|---|
| `mistralai/Mistral-7B-Instruct-v0.2` is Apache-2.0 | **Correct** (publisher model-card metadata; see 2) |
| `meta-llama/Llama-3.2-3B-Instruct` is Apache-2.0 | **Wrong.** It is governed by the **Llama 3.2 Community License Agreement** (see 3) |

The two licences are materially different: one is an OSI-approved permissive licence with no
field-of-use restriction, the other is a bespoke community licence with an acceptable-use policy,
attribution obligations and a conditional naming requirement that can reach a detector trained on
its outputs.

---

## 1. Method

For each model:

1. Resolved the exact Hugging Face model ID and the **immutable revision SHA** from the HF API
   (`/api/models/{id}` -> `sha`), not from the moving `main` ref.
2. Listed every file at that revision to see whether a licence file actually exists.
3. Attempted to fetch the licence file **at the pinned SHA**. Where access was gated, retrieved the
   identical text from Meta's official `llama-models` repository, pinned to the commit that last
   touched that file, and confirmed byte-identical hashes.
4. Read the licence text in full and recorded the clauses that bear on generation, outputs,
   attribution and redistribution.
5. Stored the evidence locally under `research/_evidence/licences/` with SHA-256 digests.

Retrieval was over the public internet on 2026-11-06. No account credentials were used.

### Evidence inventory

| File | SHA-256 | Bytes |
|---|---|---:|
| `mistral__README.md` (model card @ `63a8b081`) | `eb9690e6121bd255499c0aae038bbdba40efa16cff20cca1329a0c3ee4fbe613` | 5,683 |
| `llama32__pinned__LICENSE` | `8cc15535a8a34b41888f644b339a1a9eb428af793a4f5e24df58a3e5b1487d74` | 7,669 |
| `llama32__pinned__USE_POLICY.md` | `40e2777d7faa6beaf98400654170f414d8ab29b921b5163ad4ea0a1d39894201` | 6,021 |
| `llama32__llama.com-3_2-license.html` (rendered page, corroboration) | `31cb865ac1977034eef2a51982d09c5b762a9c035b0772ce698831bfd7e88cf6` | 182,270 |

The pinned Llama source is `meta-llama/llama-models` at commit
`8d29d93fa5700a60532e0061a02ffa89d0acd3fc` (2024-09-25, "Support for Llama3.2 series of models").

---

## 2. `mistralai/Mistral-7B-Instruct-v0.2`

| Property | Value |
|---|---|
| Model ID | `mistralai/Mistral-7B-Instruct-v0.2` |
| Revision | `63a8b081895390a26e140280378bc85ec8bce07a` |
| Revision last modified | 2025-07-24 |
| Gated | **No** - public download at the pinned revision |
| Licence | **Apache-2.0** |
| Licence evidence | Model-card front-matter `license: apache-2.0` at the pinned revision |
| Licence file in repository | **Absent.** All 16 files at the pinned revision were listed; none is `LICENSE`, `NOTICE`, `LICENSE.md` or `LICENSE.txt` (HTTP 404 for each) |
| Parameters | 7.24B |

### Findings

- The licence is established by the **publisher's own model-card metadata** on the distribution
  channel, not by a third-party uploader tag. This is materially stronger evidence than a dataset
  card's licence field, but it is weaker than a checked-in licence file: the Apache-2.0 **text** is
  incorporated by reference and is not shipped with the weights. Recorded as a documentation gap,
  not a permission gap.
- Apache-2.0 section 2 grants a perpetual, worldwide, non-exclusive, royalty-free, irrevocable
  copyright licence to reproduce, prepare derivative works of, publicly display, sublicense and
  distribute the Work, **with no field-of-use restriction**.
- Section 3 adds a patent grant that terminates on patent litigation against contributors.
- Section 4 imposes conditions only on **redistribution** of the Work or derivative works (ship the
  licence, mark modified files, retain NOTICE if one exists - there is none).
- Section 6 grants **no trademark licence**.
- The model card explicitly states the model **has no moderation mechanisms**. That is a known
  characteristic, not a licence restriction, and it is why the generation pipeline must screen its
  own outputs.

### Output terms

Apache-2.0 is a software licence. It says nothing about model outputs: it neither grants rights in
them nor restricts them. **The generated text is therefore not Apache-2.0**, and this audit does not
label it as such.

### Verdict

> **Permitted - high confidence.** Generation, storage and use of the outputs for private,
> non-commercial academic research is within the rights Apache-2.0 grants. No attribution obligation
> attaches to the outputs. No redistribution of the weights is planned; if that changed, section 4
> would apply.

---

## 3. `meta-llama/Llama-3.2-3B-Instruct`

| Property | Value |
|---|---|
| Model ID | `meta-llama/Llama-3.2-3B-Instruct` |
| Revision | `0cb88a4f764b7a12671c53f0838cd831a0843b95` |
| Revision last modified | 2024-10-24 |
| Gated | **Yes - `manual` approval required** |
| Licence | **Llama 3.2 Community License Agreement**, version date 2024-09-25 |
| Licence file | `LICENSE.txt` **exists** in the repository, but fetching it returns **HTTP 401** without an authenticated account that has accepted the terms |
| Adjunct | `USE_POLICY.md` (Acceptable Use Policy), incorporated by reference |
| Parameters | 3.21B |

The licence text used for this audit was retrieved from Meta's official `llama-models` repository at
the pinned commit and hash-matched against the second retrieval path (`llama.com`). It is **not**
Apache-2.0 and must never be described as such.

### Clause-by-clause findings

| Clause | What it actually says | Consequence for E2-C1 |
|---|---|---|
| 1.a Grant of Rights | Non-exclusive, worldwide, non-transferable, royalty-free **limited** licence to use, reproduce, distribute, copy, create derivative works of, and modify the **Llama Materials** | Generating text is "use". The act is licensed |
| 1.b.i Redistribution | Distributing Llama Materials (or a product/service containing them) requires a copy of the Agreement **and** "Built with Llama" displayed prominently | Does not trigger: we distribute neither weights nor a product containing them |
| 1.b.i Naming | If Llama Materials **or their outputs/results** are used to create, train, fine-tune or otherwise improve an AI model **that is distributed or made available**, that model's name **must begin with "Llama"** | **Bites later, not now.** A distributed StopTheCap detector trained on these outputs would have to be named `Llama...` |
| 1.b.iii Attribution | Distributed copies of Llama Materials must carry the Meta attribution notice | Does not trigger (no distribution) |
| 1.b.iv Acceptable Use | Use must comply with the Llama 3.2 Acceptable Use Policy | Applies at generation time; private detection research is not a prohibited use |
| 2 Commercial threshold | >700M monthly active users requires a separate Meta licence | Not applicable |
| 3 Warranty | "The Llama Materials **and any output and results therefrom** are provided on an 'AS IS' basis... **You are solely responsible** for determining the appropriateness of using or redistributing... and assume any risks associated with... any output and results" | Outputs carry no warranty and no grant; responsibility sits with us |
| 5.a Trademark | No trademark licence except the limited "Llama" mark use required by 1.b.i | Cannot brand the dataset as Llama-endorsed |
| 5.c Litigation | Licences terminate if the Licensee sues Meta alleging the materials **or outputs** infringe | Not applicable |
| 6 Termination | Meta may terminate on breach; sections 3, 4, 7 survive | Applies |

**Notably absent:** the pinned 3.2 text contains **no clause prohibiting use of outputs to train
another language model**. The only output-linked obligation is the naming clause in 1.b.i. This is
recorded as an observation about the pinned text, not as an inference about any other Llama release.

### Output terms

The Agreement grants rights in **Llama Materials** only. Outputs are mentioned solely in section 3
(warranty disclaimer and allocation of responsibility) and 5.c (litigation termination). It
therefore:

- does **not** grant a licence in the generated text;
- does **not** state that the generated text is unrestricted;
- does **not** assert Meta ownership of the generated text.

The correct record is: **generated text is not licensed by Meta, and its standalone copyright status
is uncertain and jurisdiction-dependent.** This audit makes no assertion of copyright over it.

### Verdict

> **Permitted for the intended use, conditional - medium confidence.**

The intended use is: generate 10,000 samples, store them in a **private** research dataset, train a
detector, and publish **aggregate metrics only**. Section 1.a licenses the act of generation, and the
Acceptable Use Policy does not prohibit this research. What the licence does **not** establish is a
grant over the *outputs* - so the workflow is designed so that no such grant is needed:

1. Generated text is **never redistributed** and never published verbatim.
2. The dataset, its manifest and the detector weights remain private.
3. If the detector is ever distributed or served publicly, it is renamed with **`Llama` as the first
   token**, per 1.b.i.
4. Access requires an authenticated Hugging Face account that has accepted the Llama 3.2 terms; the
   generation job **fails closed** without one.

This is the boundary of what the text supports. It is not a claim that Llama outputs are
"unrestricted", and it is not a claim that the Apache-2.0 conclusion for Mistral transfers here.
---

## 4. The four layers, kept separate

The single most common error in this area is collapsing these into one "dataset licence". They are
not the same, and E2-C1 records them separately.

| Layer | What it is | Governing terms | May we redistribute it? |
|---|---|---|---|
| **1. Source prompt** | Verbatim Wikipedia paragraphs | **CC BY-SA 4.0** (per-page); requires attribution, share-alike on adaptations | Only with attribution + share-alike; the E2-C1 dataset stays private, so this is not exercised |
| **2. Model weights** | Mistral-7B-Instruct-v0.2 / Llama-3.2-3B-Instruct | Apache-2.0 / Llama 3.2 Community License | We do not redistribute weights |
| **3. Generated output** | Text produced by those weights | **No licence granted by either weight licence**; copyright status uncertain | **No.** Private research use only |
| **4. Derived record** | `sample_id`, hashes, `partition`, `topic_category`, annotations | Our own record-keeping; no third-party licence attaches | Internal only |

Consequences that are enforced in code:

- `licence_ref` on each generated sample points to the **model** entry that produced it, and states
  `outputs_licensed: false`. It must not be rendered as "Apache-2.0 text".
- The prompt manifest carries its **own** `source_licence` (`CC-BY-SA-4.0`) plus `attribution` and
  `revision_id`. It is not inherited from the generator.
- No file in the repository claims a single blanket licence over the generated corpus.

---

## 5. Decision

| Generator | Status | Can it enter E2-C1? |
|---|---|---|
| `mistralai/Mistral-7B-Instruct-v0.2` | `approved_private_research` | **Yes** |
| `meta-llama/Llama-3.2-3B-Instruct` | `approved_private_research_conditional` | **Yes, subject to the four conditions in section 3** |

Both generators are admitted. The Llama entry is admitted at **medium confidence** and carries
enforceable conditions, which is why the registry distinguishes the two statuses rather than
flattening them to "Apache-2.0".

### What would reverse this decision

- Evidence that Meta's Acceptable Use Policy prohibits AI-text-detection research (it does not, on
  the pinned text).
- A requirement to publish the generated corpus or the detector weights. Publishing outputs would
  require a rights basis the Llama licence does not supply; publishing a detector trained on Llama
  outputs would trigger the 1.b.i naming obligation.
- Any change of revision. The audit is bound to the two SHAs in sections 2 and 3; a new revision
  requires a new audit. **Licences are not inferred across revisions or families.**

---

## 6. Residual uncertainty

| Item | Status |
|---|---|
| Mistral ships no licence file with the weights | Recorded. Licence text incorporated by reference to Apache-2.0 |
| Llama licence file unreadable without gated access | Mitigated: identical text from Meta's official repo, hash-matched across two paths |
| Copyright status of machine-generated text | **Unresolved and unresolvable here.** No copyright is asserted over outputs; they are used privately |
| Output ownership as between us and the model provider | Not established by either licence. Not relied upon: outputs are not redistributed |
| Llama 1.b.i naming reach | Uncertain in scope (does "train... an AI model" cover a classifier trained on generated text?). Treated **conservatively as applying**. Cost of compliance is a rename |
| Acceptable Use Policy interpretation | Read at the pinned revision; research use is not enumerated as prohibited |

---

## 7. Reproduction

```powershell
# revision + gating
curl.exe -s "https://huggingface.co/api/models/mistralai/Mistral-7B-Instruct-v0.2"
curl.exe -s "https://huggingface.co/api/models/meta-llama/Llama-3.2-3B-Instruct"

# Mistral licence declaration (no LICENSE file exists at this revision)
curl.exe -sL "https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2/raw/63a8b081895390a26e140280378bc85ec8bce07a/README.md"

# Llama licence text (HF returns 401 when gated; official mirror pinned to the commit)
curl.exe -sL "https://raw.githubusercontent.com/meta-llama/llama-models/8d29d93fa5700a60532e0061a02ffa89d0acd3fc/models/llama3_2/LICENSE"
```

Verify with `Get-FileHash <file> -Algorithm SHA256` against section 1.

---

## 8. Amendment, 2026-11-08: Llama withdrawn, Qwen3.5-4B admitted

This section supersedes section 5 for the **active** generators. The Llama 3.2 analysis in section 3
remains as evidence for a future run if access is granted; it is no longer the E2-C1 generator.

### 8.1 Why the change

The Hugging Face access request for `meta-llama/Llama-3.2-3B-Instruct` is **pending review**
(2026-11-08). The model is gated (`manual`) and cannot be downloaded until Meta approves, so the
generation run cannot proceed with it. A substitute was required.

The constraint is the runtime: a free Colab T4 has **15.64 GB** of VRAM. The official Qwen3.6 and
Qwen3.8 lineups do not contain a model that fits and remains practical for 20,000 samples:

| Model | Size (bf16) | Licence | Verdict |
|---|---:|---|---|
| `Qwen/Qwen3.8-27B` | 55.6 GB | Apache-2.0 | 4-bit is ~14 GB; borderline OOM on T4, ~20-30 h on an L4 |
| `Qwen/Qwen3.8-Flash-Next` | 360 GB | Qwen Community Licence 1.0 | Not runnable on a single GPU |
| `Qwen/Qwen3.8-2.4T-A95B` | ~2.4 T | Qwen Community Licence 1.0 | Not runnable on a single GPU |
| `Qwen/Qwen3.6-27B` | 55.6 GB | Apache-2.0 | Same problem as 3.8-27B |
| `Qwen/Qwen3.6-35B-A3B` | 71.9 GB | Apache-2.0 | Too large |
| **`Qwen/Qwen3.5-4B`** | **9.3 GB** | **Apache-2.0** | **Fits a T4 at fp16** |

### 8.2 The admitted model

| Property | Value |
|---|---|
| Model ID | `Qwen/Qwen3.5-4B` |
| Revision | `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` |
| Revision last modified | 2026-03-02 |
| Gated | **No** |
| Licence | **Apache-2.0** |
| Licence file | **Present**: `LICENSE`, 11,544 bytes, sha256 `bbedc3fda3305820b977265f01b8619d87570a6739de3a5582c3464840f1e57a` |
| Model card | `README.md`, 77,661 bytes, sha256 `1406be1b6b8fd8a6545870da516912804756593628a1d0fb0a7965211e82a7bb` |
| Architecture | `Qwen3_5ForConditionalGeneration` (`pipeline_tag: image-text-to-text`) |
| Size | 9.3 GB bf16; loaded as fp16, no quantisation |

**Licence basis.** Apache-2.0 section 2 grants use, reproduction, derivative works and distribution
with no field-of-use restriction. Unlike Mistral, Qwen ships the licence text with the weights, so
this entry does not rely on model-card metadata alone.

**Output terms.** Apache-2.0 is silent on model outputs. Generated text is therefore recorded as
`outputs_licensed: false`, exactly as for Mistral. There is **no** attribution or naming obligation
attached to the outputs, and no acceptable-use policy beyond applicable law.

### 8.3 What this costs

1. **RAID family parity is reduced.** RAID's labels include `llama-chat`; Qwen is not among the
   generators RAID tests. E2-C1 can no longer claim to target that specific column. It still targets
   `mistral` and `mistral-chat` through Mistral-7B-Instruct-v0.2.
2. **The model is 4B, not 27B.** Qwen3.5-4B is the newest *runnable* member of the family, not the
   strongest. Results must not be described as representative of Qwen3.8.
3. **The architecture is multimodal.** Text-only inference goes through `AutoProcessor` and
   `AutoModelForImageTextToText`, not `AutoModelForCausalLM`. That path must be proven by the Colab
   smoke test before any full run.

### 8.4 Updated decision

| Generator | Status | Used in E2-C1? |
|---|---|---|
| `mistralai/Mistral-7B-Instruct-v0.2` | `approved_private_research` | **Yes** |
| `Qwen/Qwen3.5-4B` | `approved_private_research` | **Yes** |
| `meta-llama/Llama-3.2-3B-Instruct` | `not_admitted_access_pending` | **No** - retained in `withdrawn_models` for a future run |

The machine-readable registry reflects this: active entries live in `models`, the Llama entry lives
in `withdrawn_models` and is not resolvable by the generation config.
