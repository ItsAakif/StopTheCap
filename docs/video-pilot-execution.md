# Video pilot execution guide (Pexels + Wan 2.1)

**Status: prepared, not executed.** No Pexels clip has been downloaded and no Wan generation has
been run. This document is the exact procedure and the acceptance criteria, so the run is
reproducible and its licence basis is on record.

Canonical dataset inventory: [`../DATASET_REGISTRY.md`](../DATASET_REGISTRY.md).

---

## 1. Why these two sources

| Side | Source | Licence | Why chosen |
|---|---|---|---|
| Real | **Pexels** | Custom licence: free use, **modification explicitly allowed**, no attribution required; may not be redistributed as stock media | Only real-video source found whose terms clearly permit processing *and* weight publication |
| Generated | **Wan 2.1 T2V 1.3B** outputs | **Apache-2.0** (`LICENSE.txt` verified, 11,357 bytes) | No contact needed; no output restriction; we own the clips |

Neither requires an email, a reply, or a signed agreement. Vript and GenVidBench remain in the
inventory but are **not** used for this pilot because their terms are silent on frame modification
and on derived weights.

---

## 2. Pexels: getting 100 real clips

### Option A - manual download (works today, no key)

1. Search https://www.pexels.com/videos/ for a category (see §4).
2. Open a clip, choose a resolution near **720p or smaller**, download the MP4.
3. Save as `data/pilot/real/<category>/<pexels-id>.mp4`.
4. Record the page URL in the manifest so attribution and provenance exist even though attribution
   is not required.

This is 100 downloads of a few MB each - roughly 300-800 MB total. It is manual, lawful, and needs
no API access.

### Option B - Pexels API (needs a free key from pexels.com/api)

```bash
export PEXELS_API_KEY=...        # never commit this
python - <<'PY'
import os, json, urllib.request
key = os.environ["PEXELS_API_KEY"]
q = "?query=nature&per_page=20&orientation=landscape"
req = urllib.request.Request("https://api.pexels.com/videos/search" + q,
                             headers={"Authorization": key})
data = json.loads(urllib.request.urlopen(req, timeout=30).read())
for v in data.get("videos", [])[:5]:
    print(v["id"], v["duration"], [(f["width"], f["height"], f["link"]) for f in v["video_files"][:2]])
PY
```

Respect the API rate limits and its terms. Do **not** scrape the website in bulk; the API is the
sanctioned path, and the manual route above is the fallback.

---

## 3. Wan 2.1: generating 100 clips

| Property | Value | Source |
|---|---|---|
| Model | `Wan-AI/Wan2.1-T2V-1.3B` | HF card |
| Licence | **Apache-2.0** | `LICENSE.txt`, verified |
| VRAM required | **8.19 GB** | model card |
| Reference speed | ~4 min per 5 s 480P clip on an RTX 4090 | model card |
| Repo size | 43.32 GB (includes multiple variants/VAEs) | HF API |

### Do not clone the whole repo

```python
from huggingface_hub import snapshot_download
snapshot_download("Wan-AI/Wan2.1-T2V-1.3B",
                  allow_patterns=["*.json", "*.pth", "*.safetensors", "LICENSE.txt"],
                  local_dir="/content/wan")
```

Use `dry_run=True` first and record the file list and total size in the run log.

### Benchmark before scaling (mandatory)

Generate **two** clips and record seconds per clip and peak VRAM. Only then decide the batch size.

```bash
python generate.py --task t2v-1.3B --size 480*832 --ckpt_dir /content/wan \
  --prompt "a red fox walking through snow" --offload_model True --t5_cpu
```

On a T4 the 4090 figure will not hold. **Treat any T4 estimate as unknown until measured.**

### Output metadata to record per clip

prompt, seed, resolution, duration, fps, generator name and version, generation date, and the
command line. All of it goes into the clip manifest - a generated clip without its prompt is not
reproducible.

---

## 4. Composition: 100 clips per class, category-matched

`CATEGORIES` in `ml/video/pilot.py`: nature, city, people, animals, food, transport, indoor,
abstract.

Aim for **12-13 clips per category per class**. Content matching matters because real and generated
clips come from different processes; if the categories diverge, a probe can separate them on
content alone.

`match_categories()` pairs clips one-to-one within a category and reports how many were dropped per
category. Use the report - an unnoticed 5:1 imbalance in one category is a confound.

---

## 5. Manifest schema

`ml/video/pilot.py::ClipRecord` enforces: `clip_id`, `label`, `source`, `group_id`, `category`,
`path`, `licence_ref`, and for generated clips `generator`; optionally `prompt`, `resolution`,
`duration_s`, `fps`, `sha256`, `split`.

Every record carries a `licence_ref` naming the exact licence and URL. A record without one fails
validation (`validate_all`), so the manifest cannot silently contain unlicensed media.

Splits come from `assign_splits()`, which buckets by **`group_id`** - so all clips cut from one
source video stay together. `leakage_report()` must return `leaking_groups == 0` before any
training.

---

## 6. The three controls

Run all three **before** reporting any detection number.

### Control 1 - source classification probe
Can a classifier tell the clip came from Pexels versus Wan, ignoring the label? Train on metadata
only (resolution, duration, fps, file size, codec). High accuracy means the classes are separable
by *acquisition* artefacts, not synthesis.

### Control 2 - real vs real
Split the Pexels clips into two disjoint halves and run the same detector between them. Any AUC
materially above 0.5 here is a pipeline artefact.

### Control 3 - compression and resizing robustness
Re-encode every clip as JPEG q=40 and downscale to 112 px, then re-extract features:

```python
from ml.video.pilot import compression_variant, resize_variant
```

Report the AUC drop. A detector that only works on pristine frames will not survive a real upload.

**Decision rule:** if controls 1 or 2 separate as strongly as real-vs-generated, stop - the probe
is detecting content or acquisition, not synthesis, and no detection claim is available.

---

## 7. Video detector baseline (P4)

**Architecture:** frozen CLIP ViT-B/32 -> uniformly sampled frames -> mean-pooled features ->
logistic regression. CLIP is **not** fine-tuned.

| Item | Value |
|---|---|
| Checkpoint | `openai/clip-vit-base-patch32` (HF) |
| Licence | **MIT** per `openai/CLIP` repository (verified). The HF model card carries **no** licence tag - record both facts |
| Params / size | ViT-B/32, image tower ~88 M params; ~600 MB fp32 |
| Input | 8-16 frames per clip, 224x224, ImageNet normalisation |
| Feature dim | 512 |
| Head | logistic regression, class-weighted if needed |

**Status: not implemented as a runnable stage yet.** The plumbing that will host it
(`run_frames_stage` -> `run_features_stage`) exists and is tested; the CLIP feature function plugs
into the same `FeatureFn` signature.

### What to measure when it runs
- feature extraction throughput (clips/s) and peak VRAM
- storage for features (should be tiny: 512 floats per clip)
- ROC-AUC with bootstrap CI, PR-AUC, human FPR
- **source leakage**: can a probe predict category or source from the features?
- **group leakage**: `leaking_groups == 0`

### What must not be claimed
Nothing about "AI video detection" from 100 clips of one generator. The honest claim ceiling is:
*"on this pilot, a frozen CLIP probe separates Pexels footage from Wan 1.3B output at AUC X; the
source and category controls suggest the signal is Y."*

---

## 8. Acceptance criteria for the pilot

1. 100 real + 100 generated clips, category-matched, all with `licence_ref` and `sha256`.
2. `validate_all().ok == True`.
3. `leakage_report().leaking_groups == 0`.
4. All three controls executed and reported.
5. Feature extraction throughput, VRAM and storage measured.
6. Every clip reproducible: prompt + seed + command for generated; URL for real.
7. No Pexels frames redistributed; only features, manifests and (Wan) clips published.
8. Report explicitly separates "pipeline validation" from "detector performance".

---

## 9. Deepfake track

Unchanged and separate. `video_manipulated` has no licensed source (DFDC terms unread), so it stays
unshipped. Generated-video data must never be relabelled as deepfake data.
