# Video access and licensing feasibility (V0)

> **Partly superseded.** The licence facts live in [`../DATASET_REGISTRY.md`](../DATASET_REGISTRY.md);
> the current pilot plan is [`video-pilot-plan.md`](video-pilot-plan.md). Retained for the original
> access analysis and the bounded-download arithmetic.

**Date:** 2026-10-29 · Evidence labels: VERIFIED (primary page read) / INFERRED / BLOCKED.

**Question this answers:** can a bounded, legally usable real-video sample be obtained for the
mandatory v1 video workstream — without downloading 201 GB and without breaching terms?

**Answer: yes, for non-commercial academic research only.** A pilot of roughly **0.7–2.9 GB** is
legally obtainable. The catch is that the real-video licence is **stricter than the benchmark's**,
which changes what may be published.

---

## 1. The licence chain — the important finding

| Layer | Licence | What it actually permits |
|---|---|---|
| **GenVidBench** (benchmark release) | **CC BY-NC 4.0** [VERIFIED, genvidbench.github.io] | Use and adaptation for non-commercial purposes, with attribution |
| **Vript** (source of GenVidBench's *real* class) | **"Academic research purposes only… you agree not to reproduce, duplicate, copy, trade, or exploit for any commercial purposes."** [VERIFIED, HF dataset card README] | Research use only. **No redistribution. No copying.** |
| **HD-VG-130M** (source of Pair2's real class) | Vript's licence is stated as "modified from the HD-VG-100M license" [VERIFIED] — HD-VG terms themselves **not read this session** | **BLOCKED** — must be checked before using Pair2 real clips |
| **VidProM** (generated source) | CC BY-NC 4.0 for Pika/ModelScope; **Apache-2.0 for VideoCraft2**; CreativeML Open RAIL-M for Text2Video-Zero [VERIFIED, HF card] | Mixed; the Apache-2.0 subset is the most permissive |

### Why this matters

**A benchmark's CC BY-NC licence does not automatically cover every underlying video.** GenVidBench
is CC BY-NC 4.0, but its real class comes from Vript, whose terms are *stricter*: research-only and
explicitly non-redistributable. The binding constraint is therefore **the strictest input**, not the
benchmark wrapper. This is exactly the risk the brief warned about, now confirmed against primary
sources.

**Practical consequence:** the project may **train on** and **report results from** Vript-derived
media for non-commercial research, must **never redistribute** those video files, and should treat
any derived artefacts conservatively until the terms are clarified in writing.

---

## 2. Access paths and sizes (VERIFIED from HF APIs)

| Source | Structure | Size of one bounded unit | Verdict |
|---|---|---|---|
| `GenVidBench/Pair1/vript.rar` | one archive, 20,131 real clips | **46,912 MB** | ⛔ too large for a pilot |
| `GenVidBench/Pair2/hd_vg_130m.7z.001–004` | multi-part 7z | **76,313 MB** | ⛔ too large; multi-part cannot be partially extracted |
| `GenVidBench/6.7m/OpenAI_Sora.zip` | zip | **468 MB** | ✅ smallest **generated** slice |
| `GenVidBench/Pair1/ms.rar` | rar | **2,623 MB** | ✅ generated, Pair1-matched |
| **`Mutonix/Vript` short videos** | **42 zip shards** | **166–261 MB per shard** | ✅ **bounded real-video access** |
| **`Mutonix/Vript-ERO`** | **402 individual `.mp4`** | **0.81 GB total**, 0.6–4.4 MB each | ✅ **per-file real access — best for a pilot** |
| `WenhaoWang/VidProM` | 30 tar shards per generator | **~34 GB per tar** | ⛔ too large |

**Key operational fact:** Vript is not one monolithic download. It ships as 42 short-video zip
shards (~200 MB each) plus per-file `.mp4` access in the ERO subset. That is what makes a bounded
pilot possible at all.

---

## 3. Recommended pilot (bounded)

| Component | Source | Download | Licence posture |
|---|---|---|---|
| **Real clips** | `Mutonix/Vript-ERO` — ~150–200 individual `.mp4` | **~0.3–0.4 GB** | research only; do not redistribute |
| **Generated clips** | `GenVidBench/6.7m/OpenAI_Sora.zip` | **468 MB** | CC BY-NC 4.0 |
| **Total** | | **≈ 0.8–0.9 GB** | |

Matched-content variant, if storage allows (~3 GB): `GenVidBench/Pair1/ms.rar` (2.6 GB generated)
paired with the corresponding `vript` clips, which GenVidBench designed as a content-matched pair.

**Known confound to measure, not ignore:** in the unmatched variant, real and generated clips come
from different corpora, so *content* differs as well as provenance. A frozen-backbone probe scoring
well there could be detecting "scenery vs prompt-generated content" rather than synthesis artifacts.
The pilot must therefore include a **content-control**: the same probe on real-vs-real (two disjoint
real subsets) and on generated-vs-generated (two generators). If those also separate strongly, the
signal is content, not generation.

---

## 4. Go / no-go

**GO**, with these conditions:

1. Use only non-commercial research purposes.
2. Never commit, redistribute or bundle Vript-derived media or derived frame caches.
3. Record the licence chain in the data card, naming Vript explicitly.
4. Treat HD-VG-130M (Pair2 real) as **blocked** until its licence is read.
5. Keep DFDC (manipulated video) blocked until its Kaggle terms are read.
6. Do **not** claim the model detects "deepfakes" from this data — it detects fully generated video.

## 5. Fallbacks if Vript access proves impractical

| Fallback | Trade-off |
|---|---|
| Generated-only + real clips from a permissively licensed source (e.g. VideoCraft2 clips under Apache-2.0 from VidProM) | Cleaner licensing; content still unmatched; VidProM access is 34 GB tar shards, so this is weak in practice |
| Single-generator only (Sora 468 MB) with no real class | Feasible but cannot support a binary real-vs-generated claim; only "can a probe recognise Sora?" — a much weaker result |
| Public benchmark leaderboard numbers instead of own training | Honest but not a portfolio contribution |
| Report infeasibility | Legitimate outcome; must be reported, not worked around |

---

## 6. What remains unverified

| Item | Status | Why it matters |
|---|---|---|
| HD-VG-130M terms | **BLOCKED** | Gates Pair2 real clips |
| DFDC competition terms | **BLOCKED** | Gates the entire manipulated-video task |
| Whether Vript terms extend to **derived model weights** | **Unclear** | Determines whether a trained video model may be published at all |
| Whether GenVidBench's `vript.rar` is byte-identical to `Mutonix/Vript` clips | Unknown | Affects whether Pair1 match-ups can be reconstructed |
| Actual video codecs/frame rates in the shards | Unknown | Affects decode cost estimates |

**Open question requiring the owner's judgement:** given that Vript is research-only and silent on
derived weights, should the video model be published at all, or only its *evaluation results*? The
conservative reading is to publish code + results + an evaluation harness, and withhold weights
until the licensor clarifies.
