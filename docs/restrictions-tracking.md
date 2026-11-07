# Restrictions tracking — media, derived data and weights

> **Canonical dataset inventory: [`../DATASET_REGISTRY.md`](../DATASET_REGISTRY.md).**
> This file records *policy* derived from licences; the registry records *facts about licences*.
> If they disagree, the registry wins and this file is wrong.

Decision 6 requires restrictions to be tracked **separately** for each layer, because they differ.
A single "licence" column hides the exact thing that decides what can ship.

Tracked layers:

| Layer | What it is |
|---|---|
| **L1 Code** | application, data-processing, evaluation and training code |
| **L2 Source media** | original datasets / videos as downloaded |
| **L3 Derived data** | normalised records, frame caches, splits, manifests |
| **L4 Architecture** | model design, configs, training recipes |
| **L5 Pretrained weights** | third-party checkpoints we start from |
| **L6 Derived weights** | checkpoints we produce |
| **L7 Results** | metrics, reports, evaluation outputs |

---

## Current status

| Asset | L1 Code | L2 Media | L3 Derived data | L5 Pretrained | L6 Derived weights | L7 Results |
|---|---|---|---|---|---|---|
| **Text — Ghostbuster** | ✅ publish | ✅ CC BY 3.0 | ✅ publish | n/a | ✅ publish | ✅ publish |
| **Text — RAID** | ✅ publish | ✅ MIT (attribute) | ⚠️ eval-only; publish derived metrics, not a derived corpus | n/a | ✅ publish (never trained on) | ✅ publish |
| **Text — HC3** | ✅ publish | ✅ CC BY-SA 4.0 | ⚠️ ShareAlike applies | n/a | ✅ publish | ✅ publish |
| **Video — GenVidBench** | ✅ publish | ⚠️ CC BY-NC 4.0, media not held | ✅ manifests/audit only | ⚠️ backbone TBD | ⛔ **withheld** | ✅ publish |
| **Video — Vript** | ✅ publish | ⛔ **do not redistribute** | ⛔ **not redistributable** (also "no modification") | — | ⛔ **withheld pending clarification** | ✅ publish |
| **Video — Vript-ERO** | ✅ publish | ⛔ **no licence declared** | ⛔ | — | ⛔ | ⛔ do not use |
| **Video — Pexels** (real) | ✅ publish | ⚠️ permissive, but **no redistribution as stock media** | ⛔ don't republish frames; ✅ features | — | ✅ publish | ✅ publish |
| **Video — Wan2.1-T2V-1.3B outputs** (we generate) | ✅ publish | ✅ Apache-2.0 model, no output restriction | ✅ publish | ✅ Apache-2.0 | ✅ publish | ✅ publish |
| **Video — HD-VG-130M** | — | **BLOCKED** | **BLOCKED** | — | **BLOCKED** | **BLOCKED** |
| **Video — DFDC** | — | **BLOCKED** | **BLOCKED** | — | **BLOCKED** | **BLOCKED** |

Legend: ✅ permitted · ⚠️ conditional or unresolved · ⛔ not permitted · BLOCKED = terms unread

---

## Rules that follow

1. **Code is always open** (L1, L4, L7). Nothing in this project changes that.
2. **Original media is never redistributed** unless its own licence says so. RAID and Ghostbuster
   text may be redistributed (with attribution); HC3 only under ShareAlike; no video media may be.
3. **Derived data inherits the strictest input.** A frame cache from Vript is a derived dataset and
   is not redistributable, even though the *code that made it* is.
4. **Weight publication is judged per dataset**, not per project. A text model trained on Ghostbuster
   may be published. A video model trained on Vript may not, yet.
5. **Do not describe the video model as "open-source"** while L6 is withheld. Correct phrasing:
   *"research prototype; training and evaluation code open, weights withheld pending licence
   clarification."*
6. **Record the licence evidence URL and date** in the manifest for every source — enforced by
   `tests/test_manifests.py`, which fails the build if an enabled source lacks an evidence URL.

---

## Open items

| Item | Blocks | Owner |
|---|---|---|
| Vript: are derived weights publishable? | Video model release | Maintainer request (`video-permission-request.md`) |
| Vript: is frame extraction a permitted use? | Video preprocessing | Maintainer request |
| Vript: may results and code be published? | Portfolio artefact | Maintainer request |
| GenVidBench: does CC BY-NC cover trained weights? | Video model release | Maintainer request |
| HD-VG-130M terms | Pair2 real class | Read the licence |
| DFDC terms | `video_manipulated` task | Kaggle acceptance |
| M4GT-Bench licence | mixed-text / attribution tasks | Maintainer request |

---

## Feature vs frame distinction (added 2026-10-30)

The cloud workflow publishes **embeddings, not imagery**. That distinction is what makes the
Pexels route publishable at all:

| Layer | Pexels | Implications |
|---|---|---|
| Original clips | ⛔ do not redistribute as stock media | keep locally / on Hub **private** only |
| Extracted frames | ⛔ treat as redistribution; do not publish | derive embeddings instead |
| **Embeddings / features** | ✅ publish | they are not the imagery |
| Model weights | ✅ publish | training on permissively licensed media |
| Code, manifests, metrics | ✅ publish | — |

If a reviewer ever asks "did you redistribute the data?", the answer must be **no** — only features,
manifests and weights.
