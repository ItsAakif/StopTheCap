# Video alternatives and the smallest legally publishable path

> **Canonical dataset inventory: [`../DATASET_REGISTRY.md`](../DATASET_REGISTRY.md).**
> This file argues *which route to take*; the registry records *what each licence permits*.

**Date:** 2026-10-30 · VERIFIED = primary page read · ESTIMATE = arithmetic, not measured ·
BLOCKED = could not read.

Requirements any candidate must meet (unchanged):

| # | Requirement |
|---|---|
| R1 | Bounded access (< ~5 GB per pilot) |
| R2 | Licence permits non-commercial research |
| R3 | Licence permits **publishing derived model weights** |
| R4 | Licence permits open-source code and evaluation artefacts |
| R5 | Media redistribution permitted, or cleanly avoidable |
| R6 | A usable negative (real) class, or a lawful way to construct one |
| R7 | Actually contains AI-generated **video**, not just prompts |

---

## 1. Candidates, with the evidence

| Candidate | Licence (evidence) | R1 | R2 | R3 | R6 | Verdict |
|---|---|---|---|---|---|---|
| **GenVidBench** generated archives | CC BY-NC 4.0 ([site](https://genvidbench.github.io)) | ✅ 468 MB–3 GB | ✅ | ⚠️ not stated | ⛔ generated only | **Prototype only**; weights unproven |
| **Vript** (real class) | Academic only, no distribution, no modification ([card](https://huggingface.co/datasets/Mutonix/Vript)) | ✅ 12–1,100 MB/shard | ✅ | ⛔ silent | ✅ | **Blocked** pending reply |
| **Vript-ERO** | **none declared** (156-byte README) | ✅ 0.8 GB | ⛔ | ⛔ | ✅ | **Unusable** |
| **VidProM** — VideoCraft2 subset | **Apache-2.0** for vc2 clips ([card](https://huggingface.co/datasets/WenhaoWang/VidProM)) | ⛔ **30 × ~34 GB tars** | ✅ | ✅ | ⛔ generated only | Licence ideal, access impossible |
| **Wan2.1-T2V-1.3B** (generate our own) | **Apache-2.0** ([card](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B)) | ✅ model 43 GB repo, subset needed | ✅ | ✅ | ⛔ generated only | **Best generated-side option** |
| **Pexels** (real class) | Free to use, **modification allowed**, no attribution ([licence](https://www.pexels.com/license/)) | ✅ per-clip API | ✅ | ✅ | ✅ **real** | **Best real-side option** |
| **Kinetics-400/700** | `other` / no tag; YouTube-sourced | ⛔ | ⚠️ | ⛔ | ✅ | **Avoid** — same provenance problem as Vript |
| **DFDC / Celeb-DF / FaceForensics++** | Not read; research-only | ⛔ | **BLOCKED** | **BLOCKED** | ✅ manipulated | **Blocked** |

### The two verified enablers

**Pexels licence (read in full, 2026-10-30).** Allowed: free download and use of photos and videos;
**"You can modify the photos and videos from Pexels. Be creative and edit them as you like."**;
attribution not required. Not allowed: selling unaltered copies; implying endorsement; **"Don't
redistribute or sell the photos and videos on other stock photo or wallpaper platforms"**; using
them as a trade mark.

That last clause is the only one that touches us. It prohibits republishing the clips **as stock
media**. It does not prohibit training on them, nor publishing weights or embeddings. Still, the
conservative reading is: **do not republish Pexels frames as a dataset**; publish features instead.

**Wan2.1-T2V-1.3B (Apache-2.0).** Model card states it "requires only **8.19 GB VRAM**" and
generates a 5-second 480P clip in "about **4 minutes on an RTX 4090**" without optimisation.
Apache-2.0 has no output restriction, so clips we generate are ours to publish.

---

## 2. The smallest legally publishable path

> **Real class:** Pexels video clips, sampled via the official API.
> **Generated class:** clips we generate ourselves with Wan2.1-T2V-1.3B (Apache-2.0).
> **Published:** code, manifests, features/embeddings, metrics, model weights.
> **Not published:** Pexels frames as a redistributable dataset.

This is the only combination found so far satisfying R1–R5 **simultaneously**.

### Honest weaknesses

| Weakness | Consequence |
|---|---|
| One generator only (Wan 1.3B) | The detector recognises *that* generator. Never claim general video detection. |
| Real and generated content are unrelated | Strong content confound → the control probes are mandatory, not optional |
| Generation is compute-bound | 480P/5s at ~4 min on a 4090 ⇒ **ESTIMATE 8–16 min/clip on a T4** |
| Kaggle allows ~30 GPU-h/week [REPORTED, verify in-account] | ~120–220 clips/week ⇒ 150 clips ≈ 1–1.5 weeks of allowance |
| Pexels is stock footage, not "everyday internet video" | Distribution differs from real deployment |

### Compute estimate (ESTIMATE — must be measured)

| Target | Clips | GPU-h on T4 (8–16 min/clip) | Kaggle weeks at 30 h |
|---|---:|---:|---:|
| Minimum viable pilot | 100 | 13–27 | ~1 |
| Useful pilot | 300 | 40–80 | 1.5–3 |
| Not recommended yet | 1,000 | 133–267 | 4.5–9 |

**First action is not generation — it is a 2-clip timing run.** Measure seconds/clip and peak VRAM,
then substitute into the table above. Do not trust these numbers.

### Cheaper variant if generation is too slow

Shorter clips (2–3 s) and/or a lower step count cut cost roughly proportionally. A 2-second 480P
clip at reduced inference steps may land near 3–6 min on a T4. **Trade-off:** shorter clips are
harder to classify, so this must be validated rather than assumed.

---

## 3. How this interacts with the existing plan

| Route | Status | Weights publishable? |
|---|---|---|
| **A. GenVidBench + Vript** (Bundle B) | **not pursued** — terms silent on frame modification and weights | ⛔ no |
| **B. GenVidBench generated only** | prototype only; no real class; Sora ZIP terms undocumented | ⚠️ unresolved |
| **C. Pexels + Wan2.1 (own generations)** | **selected** — no contact needed, no restricted media | ✅ **yes** |
| **D. VideoCraft2 (Apache-2.0)** | licence ideal, 34 GB tar shards | ✅ yes, if access solved |

**Recommendation:** run **C**. Vript and GenVidBench stay in the research inventory with their
restrictions recorded, but nothing waits on them. If per-clip access to **D** ever becomes
available, it is the better generated-side source because Apache-2.0 is unambiguous.

---

## 4. What must be measured before committing

1. Seconds per generated clip and peak VRAM on the actual cloud GPU.
2. Pexels API: rate limits and whether bulk metadata fetch is permitted under its API terms.
3. Frame-extraction throughput (frames/s) for 480P clips.
4. Whether a frozen-backbone probe separates Pexels-real from Wan-generated **after** the content
   controls — if the controls also separate strongly, the whole approach is measuring content.
5. Disk per 1,000 clips at 8–16 frames, 224², float16 or uint8.

Only (4) decides whether a detector claim is possible at all.
