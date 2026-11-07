# Video pilot — file list, matching methodology and licence evidence

> **Canonical dataset inventory: [`../DATASET_REGISTRY.md`](../DATASET_REGISTRY.md).**
> Licence summaries here are for convenience; the registry is authoritative.

**Status: ⛔ NOT DOWNLOADED.** Permitted use is not yet documented to the standard required
(decision 8), for the reasons in §4. This document is the pre-download record.

**Date:** 2026-10-29 · Labels: VERIFIED (primary page read) / INFERRED / BLOCKED.

---

## 1. What the investigation established

| Question | Answer | Evidence |
|---|---|---|
| Do GenVidBench's real clips come from Vript? | **Yes — all 20,131** Pair1 `vript` scenes appear in Vript's long-video captions | VERIFIED: normalised name match = 20,131/20,131 |
| Is there a shared ID between real and generated clips? | **No.** Generated clips use UUIDs (`ms-0015c714-…`); real clips use `<videoId>-Scene-NNN.mp4` | VERIFIED |
| Is there a published generated→real mapping? | **No.** No file in the HF repo or GitHub repo maps a generated UUID to a source scene or prompt | VERIFIED by exhaustive file listing |
| How is "Pair1" pairing defined? | At **benchmark-set / prompt-distribution level**, not per clip. The paper states the cross-source design "reduces the interference of video content" — i.e. content is *deliberately decorrelated* | VERIFIED (project site) + INFERRED |
| Does a Vript shard manifest exist? | **No.** Shards are named `video_NNNN_of_1095.zip` with no id mapping | VERIFIED |
| Can a shard's contents be predicted? | **No.** Probed the smallest shard (11.4 MB): `video_1095_of_1095.zip` contains exactly one video, `-0kUuGnnZwU`, which has **0** clips in Pair1. Two candidate orderings (captions order, sorted order) put it at positions 2318 and 47 — neither is 1095 | VERIFIED |
| Is Vript-ERO usable? | **No — no licence declared.** Its README is 156 bytes of front-matter only, with no terms. Absence of a licence is not permission | VERIFIED |

**Consequence:** the approved "matched ~2.9 GB route" **cannot be assembled as a matched pilot**
from public data. Pairing is impossible to reconstruct without either the 46.9 GB `vript.rar`, a
shard manifest from the maintainers, or brute-force downloading all 1,095 shards (607 GB).

---

## 2. Exact files — the three executable bundles

### Bundle A — truly matched (rejected: 16× over budget)

| File | Size | Class |
|---|---|---|
| `GenVidBench/Pair1/vript.rar` | **46,912 MB** | real (all 20,131 Pair1 scenes) |
| `GenVidBench/Pair1/ms.rar` | 2,623 MB | generated (13,501 clips) |
| **Total** | **≈ 48.5 GB** | |

Rejected: exceeds the approved budget by ~16× and cannot be partially extracted (RAR).

### Bundle B — closest to the approved ~2.9 GB (executable, **unmatched**)

| File | Size | Class | URL |
|---|---|---|---|
| `GenVidBench/Pair1/ms.rar` | **2,623 MB** | generated | `huggingface.co/datasets/jian-0/GenVidBench/resolve/main/GenVidBench/Pair1/ms.rar` |
| `vript_long_videos/video_1095_of_1095.zip` | 11.4 MB | real | `huggingface.co/datasets/Mutonix/Vript/resolve/main/vript_long_videos/video_1095_of_1095.zip` |
| `vript_long_videos/video_807_of_1095.zip` | 166.9 MB | real | `…/video_807_of_1095.zip` |
| `vript_long_videos/video_7_of_1095.zip` | 224.3 MB | real | `…/video_7_of_1095.zip` |
| **Total** | **3,025.6 MB (≈3.0 GB)** | | |

The three shards are chosen **by size only** (smallest first), because shard contents cannot be
predicted. Expected yield: ~1 video each, ~38 clips per video on average, so **~40–120 real clips**
against 13,501 generated. That imbalance is acceptable for a feasibility probe but must be
subsampled on the generated side and reported.

### Bundle C — cheapest (≈0.87 GB), if Bundle B's real-class yield proves too thin

| File | Size | Class |
|---|---|---|
| `GenVidBench/6.7m/OpenAI_Sora.zip` | 468 MB | generated |
| the same three Vript shards | 402.6 MB | real |
| **Total** | **≈ 871 MB** | |

---

## 3. How real and generated clips are matched

Because no per-clip mapping exists, matching is defined explicitly at three levels. **Only the
weakest level is achievable with public data.**

| Level | Definition | Achievable? |
|---|---|---|
| **L1 — instance pairing** (clip ↔ the exact clip generated from its prompt) | requires a generated→scene map | ❌ **not published** |
| **L2 — set pairing** (both classes from the same benchmark release, Pair1) | generated from Pair1 + real from Pair1's 20,131 scenes | ❌ needs `vript.rar` |
| **L3 — corpus pairing** (both classes descend from Vript: real Vript scenes ↔ videos generated from Vript-derived prompts) | real from any Vript shard + generated from Pair1 | ✅ **the best available** |

**Bundle B is L3.** It is honest to describe it as "same source corpus, content decorrelated", not
as a matched pair. The paper's own design intends this decorrelation, so L3 is arguably closer to
the benchmark's intent than a naive content match would be — but it must be described accurately.

**Mandatory control experiment.** L3 leaves a content confound: real shards may show arbitrary
scenes while generated clips follow prompts. Before any claim, run the same probe on:

1. **real vs real** — two disjoint Vript shards;
2. **generated vs generated** — two generator archives (e.g. `ms` vs `cogvideo`);
3. **content-stratified** — restrict both classes to overlapping semantic classes using
   `Vript_20k_classes.txt` (which maps every Pair1 real clip to a 3-part semantic class such as
   `6_1_1` = Nature / Action / Location).

If (1) or (2) separate as strongly as real-vs-generated, the probe is detecting content or
generator artefacts, not synthesis.

---

## 4. Why the download is withheld

| Gate | Status |
|---|---|
| Exact file list | ✅ §2 |
| Expected disk usage | ✅ §2 (3.0 GB, plus ~2× for frames) |
| Benchmark licence | ✅ CC BY-NC 4.0 (genvidbench.github.io) |
| **Underlying media licence** | ⚠️ **Vript: "ACADEMIC USE ONLY" + "NO DISTRIBUTION"** |
| **"No modification" clause** | ⚠️ Vript forbids "any form of broadcasting, **modification** or any other similar behavior". Frame extraction and re-encoding are plausibly "modification". **Unresolved.** |
| **Derived model weights** | ⛔ Vript is silent. Decision 5 forbids publishing weights trained on Vript-derived media until terms clearly permit it. |
| Matching feasibility | ⛔ L1/L2 impossible; only L3 available |

**Vript licence, verbatim:** *"Any content from Vript/Vript-Bench dataset and Vriptor model is
available for academic research purposes only. You agree not to reproduce, duplicate, copy, trade,
or exploit for any commercial purposes."* and *"Without the permission of the copyright owner, you
are not allowed to perform any form of broadcasting, modification or any other similar behavior to
the data set content."*

**This is stricter than the benchmark wrapper.** GenVidBench's CC BY-NC 4.0 does not upgrade the
underlying media terms. Permissions are therefore **not** inferred from the wrapper.

---

## 5. Licence evidence per source

| Source | Licence | Evidence URL | Permits research? | Permits derived weights? | Permits redistribution? |
|---|---|---|---|---|---|
| GenVidBench (release) | CC BY-NC 4.0 | `genvidbench.github.io` (§License) | ✅ non-commercial | ⚠️ not stated | ⚠️ NC only |
| **Vript** (real class) | Academic research only; no distribution; no modification | `huggingface.co/datasets/Mutonix/Vript` → README → `## License` | ✅ academic | **⛔ silent — treat as not permitted** | ⛔ |
| Vript-ERO | **none declared** | `…/Mutonix/Vript-ERO/README.md` (156 bytes, no terms) | ⛔ unknown | ⛔ | ⛔ |
| HD-VG-130M (Pair2 real) | Not read | — | **BLOCKED** | **BLOCKED** | **BLOCKED** |
| VidProM | CC BY-NC 4.0 / **Apache-2.0 (VideoCraft2)** / RAIL-M (T2V-Zero) | `…/WenhaoWang/VidProM` README | ✅ / ✅ | mixed | mixed |
| DFDC | Not read (Kaggle auth) | — | **BLOCKED** | **BLOCKED** | **BLOCKED** |

---

## 6. Group preservation in splits

Already implemented for the inventory and enforced by `video_no_group_overlap`:

- `group_id = genvidbench:{pair}:{generator_dir}:{stem}` — one clip per group for generated data.
- `identity_group = generator_dir` — prevents a generator straddling splits.
- For downloaded media the splitter must additionally key on **source video id**, so all clips cut
  from one long video stay together. Concretely: `group_id = vript:{video_id}` for real clips and
  `group_id = genvidbench:{pair}:{generator}:{uuid}` for generated, with the union-find merging any
  future pairing keys.

**Never** split by clip. Two scenes from the same source video are near-duplicates in lighting,
subject and camera, and separating them would leak.

---

## 7. Decision required

1. **Accept L3 corpus pairing** (Bundle B, ~3.0 GB) with the mandatory content controls — this is
   what is executable today and what the budget was sized for.
2. **Or** authorise a maintainer request for (a) the shard manifest or (b) the Pair1 real subset,
   and defer the download until answered.
3. **Or** pivot to a licence-clean alternative dataset (see `video-alternatives.md`).

**Recommendation:** do (2) and (1) in parallel — send the maintainer request, and hold the download
until the "no modification" question is answered, because a 3 GB download whose output cannot be
published is a poor use of the budget.
