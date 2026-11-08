# Data and licence layering

The Apache-2.0 licence in `LICENSE` covers the **source code** in this repository. It does not, and
cannot, relicense third-party text, datasets or model weights. This file records what applies to
what. Read it before reusing anything under `datasets/` or `reports/`.

## Scope of the code licence

| Path | Licence | Notes |
|---|---|---|
| `ml/`, `scripts/`, `tests/`, `notebooks/` | **Apache-2.0** | The code grant in `LICENSE` |
| `docs/`, `README.md`, `DATASET_REGISTRY.md` | **Apache-2.0** | Our own prose |
| `reports/` (audit output, E1 metrics, diagnosis) | **Apache-2.0** | Our own measurements and derived numbers |
| `datasets/manifests/*.yaml`, `*.json` (except the prompt manifest) | **Apache-2.0** | Metadata, split assignments, digests, licence registry |
| `datasets/manifests/e2_c1_prompts.jsonl` | **CC BY-SA 4.0** | Wikipedia text; see below |

The presence of a file in this repository does not change the licence of the material inside it.

## Wikipedia prompt inventory

`datasets/manifests/e2_c1_prompts.jsonl` contains 5,000 paragraphs from English Wikipedia.

- **Licence:** CC BY-SA 4.0 (`https://creativecommons.org/licenses/by-sa/4.0/`).
- **Attribution:** every record carries an `attribution` string, `source_title`, `revision_id` and a
  permanent `source_url` of the form `https://en.wikipedia.org/w/index.php?oldid=<revision_id>`.
- **Share-alike:** applies to adaptations of the text. If you modify these paragraphs and distribute
  the result, the adaptation must be CC BY-SA 4.0.
- **What we did:** the stored text is a deterministic plain-text rendering of wikitext (markup,
  templates, references and tables removed). It is a derivation, not a byte copy of rendered HTML.
  `prompt_hash` pins exactly what is stored; `revision_id` pins the source.
- **Provenance:** `docs/e2/prompt-provenance.md` records the full selection funnel, the split and the
  limitations (including topic skew and best-effort safety filtering).

Because attribution is carried per record, redistributing the file as-is is compatible with
CC BY-SA 4.0. Removing the attribution fields is not.

## Third-party datasets: not distributed

E1 was trained on **Ghostbuster (CC BY 3.0)** and evaluated on **HC3 (CC BY-SA 4.0)** and **RAID
(MIT)**. The raw and canonical records are **not** in this repository. What is committed is
metadata: manifests, counts, digests, evaluation results and audit reports.

If you rebuild those corpora locally, their licences apply to you. In particular, HC3 and RAID are
evaluation-only in this project by policy, enforced in `ml/audit/audit.py` and tested in
`tests/test_manifests.py`. Do not train on them.

## Model weights

| Model | Licence | Role |
|---|---|---|
| `distilbert-base-cased` | Apache-2.0 | detector (E1, and the E2-C1 architecture) |
| `mistralai/Mistral-7B-Instruct-v0.2` | Apache-2.0 | generator for E2-C1 |
| `Qwen/Qwen3.5-4B` | Apache-2.0 | generator for E2-C1 (active) |
| `meta-llama/Llama-3.2-3B-Instruct` | Llama 3.2 Community License | **not used**: HF access pending review (2026-11-08) |

No weights are distributed here. See `docs/e2/model-licence-audit.md` for the clause-level audit.
That document also records two corrections: Llama 3.2 is **not** Apache-2.0, and it is no longer an
active generator because its access request is pending (amendment dated 2026-11-08).

## Generated text

No generated corpus is distributed. This matters because **neither weight licence grants rights in
model outputs**:

- Apache-2.0 licenses the Work (software and weights) and is silent on outputs.
- The Llama 3.2 Community License (retained as historical evidence only) addresses outputs only to
  disclaim warranties and place responsibility on the licensee.

Generated text is therefore recorded in the pipeline as `outputs_licensed: false`. Its standalone
copyright status is uncertain and jurisdiction-dependent. If you generate a corpus with this code,
you are responsible for how you use and distribute it.

## What this repository does not grant

- No patent or trademark rights beyond what Apache-2.0 grants for the code.
- No rights in Wikipedia text beyond CC BY-SA 4.0.
- No rights in Ghostbuster, HC3, RAID or any other third-party dataset.
- No rights in model weights.
- No rights in generated text.

Nothing here is legal advice.
