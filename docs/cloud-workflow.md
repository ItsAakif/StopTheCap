# Cloud-first dataset and training workflow

> **Canonical dataset inventory: [`../DATASET_REGISTRY.md`](../DATASET_REGISTRY.md).**
> Dataset licences are referenced there; this file covers infrastructure only.

**Date:** 2026-10-30 · VERIFIED / ESTIMATE / BLOCKED as marked.
**Governing principle:** *compute is disposable, the Hub is durable.* Every stage must be able to
die mid-run and resume without redoing work, because both Colab and Kaggle terminate sessions.

---

## 0. What I could and could not verify

| Claim | Status |
|---|---|
| HF selective download via `snapshot_download(allow_patterns=…, ignore_patterns=…, dry_run=True)` | **VERIFIED** (HF docs, read) |
| `hf download --include … --dry-run` lists files and sizes without downloading | **VERIFIED** (same page) |
| Kaggle ≈30 GPU-h/week, 12 h CPU/GPU sessions, 9 h TPU, 20 GB `/kaggle/working` | **REPORTED** — Kaggle's docs page is client-rendered; I could not read it directly. Confirm in-account |
| Colab compute units, per-hour rates, GPU class | **BLOCKED** — behind sign-in. Never assume |
| Wan2.1-T2V-1.3B: 8.19 GB VRAM, ~4 min per 5 s 480P clip on a 4090 | **VERIFIED** (model card), but a 4090 is not a T4 — the T4 time is an ESTIMATE |
| Pexels licence permits modification, forbids stock redistribution | **VERIFIED** (licence page, read) |

**Nothing in this document assumes a fixed free-tier quota.** Every capacity number is either
labelled REPORTED or left for you to measure.

---

## 1. Architecture

```text
        ┌──────────────── DURABLE (survives everything) ────────────────┐
        │  Hugging Face Hub  · private dataset repo                    │
        │    manifests/  plan.json  chunk markers (sha256)             │
        │    features/   frame embeddings (parquet)                    │
        │    ckpts/      last.pt + run logs                            │
        │  GitHub                                                        │
        │    code, configs, split manifests, reports (no media)        │
        └──────────────────────────────────────────────────────────────┘
                     ▲                    ▲                    ▲
     ┌───────────────┴──────┐  ┌──────────┴────────┐  ┌────────┴─────────┐
     │ Colab (AI Pro)       │  │ Kaggle Notebooks  │  │ Local (4060)     │
     │ ephemeral disk       │  │ ephemeral + /kaggle│  │ dev + unit tests │
     │ Drive for ckpts      │  │ /working 20 GB     │  │ no big data      │
     └───────────┬──────────┘  └──────────┬────────┘  └──────────────────┘
                 │                        │
     ┌───────────▼────────────────────────▼───────────────┐
     │ Pipeline stages (each chunked, resumable, hashed)  │
     │  S0 plan → S1 acquire/generate → S2 frames         │
     │         → S3 features → S4 train → S5 evaluate     │
     └────────────────────────────────────────────────────┘
```

**Stage outputs are the only interface between stages.** Every stage reads a manifest and writes
(a) one artefact per chunk and (b) a `.done.json` marker containing the artefact's SHA-256. That
makes the whole pipeline restartable from any environment.

Implemented and tested: `ml/video/chunks.py` (planning, markers, resume, frame sampling) with 15
tests, including "a truncated artefact is not done".

---

## 2. Colab vs Kaggle, per experiment

| Experiment | Needs | Better fit | Why |
|---|---|---|---|
| Text baseline (TF-IDF) | CPU, minutes | **Either** — local is fine | Runs in ~90 s locally |
| Text encoder E1 (distilbert/bert, 11,150 rows, 3 ep) | ~1–5 GPU-h | **Kaggle**, secondarily Colab | Fits comfortably in 30 h/week; 12 h session covers it |
| Text architecture sweep | 10–30 GPU-h | **Kaggle** | Predictable weekly allowance beats unmetered-but-opaque units |
| **Video generation (Wan 1.3B)** | 8.19 GB VRAM, long runs | **Kaggle** | 12 h sessions; T4 16 GB fits the 8.19 GB requirement; generation is hours, and Kaggle's session cap is the binding constraint |
| Video frame extraction | CPU, I/O | **Kaggle CPU sessions** | 12 h CPU sessions; no GPU needed |
| Frozen-backbone feature extraction | GPU, minutes per 100 clips | **Either** | Small; run where the frames already are |
| Full video fine-tune | Not planned yet | **Blocked** on data licensing | — |

**Practical rule:** Kaggle for anything long or GPU-bound; Colab for short interactive work and as
overflow when Kaggle's weekly GPU allowance is exhausted. The local 4060 is for development and
unit tests only — its PyTorch build is CPU-only (`research/_notes/environment.md`).

**Storage reality check:** Kaggle's 20 GB `/kaggle/working` cannot hold a video corpus. Frames for
1,000 clips at 12 frames × 224² × 3 channels ≈ **1.8 GB** as `uint8` — fine.
**Measured 2026-11-03:** 1.204 MB per clip at 8 frames/224² ⇒ **1.81 MB per clip at 12 frames**
(`reports/video_poc.json`). Raw 480P clips are not fine and must never enter a training path.

---

## 3. Getting data in: selective, never wholesale

Verified pattern (HF docs):

```python
from huggingface_hub import snapshot_download

# 1. See what WOULD be downloaded, with sizes. Costs nothing.
snapshot_download("jian-0/GenVidBench", repo_type="dataset",
                  allow_patterns="GenVidBench/6.7m/OpenAI_Sora.zip", dry_run=True)

# 2. Fetch exactly one archive.
snapshot_download("jian-0/GenVidBench", repo_type="dataset",
                  allow_patterns="GenVidBench/6.7m/OpenAI_Sora.zip",
                  local_dir="/content/data")

# 3. Never take the whole repo (201 GB).
```

```bash
hf download jian-0/GenVidBench --repo-type dataset \
  --include "GenVidBench/6.7m/OpenAI_Sora.zip" --dry-run
```

**Streaming instead of downloading** (for text, and for any parquet-based corpus):

```python
from datasets import load_dataset
ds = load_dataset("liamdugan/raid", split="train", streaming=True)
for i, row in enumerate(ds):
    if i >= 2000: break
```

That is how the existing RAID sample was taken without the 800 MB–11.8 GB file.

**Rules**

1. `dry_run=True` (or `--dry-run`) before every download; paste the file list into the run log.
2. `allow_patterns` pinned to exact paths, never `*`.
3. Pin `revision=` to a commit SHA in the manifest — `main` moves.
4. Record the resolved commit + file SHA-256 in the chunk marker.
5. Never `snapshot_download` a repo >2 GB without a written justification in the manifest.

---

## 4. Persistent storage across disconnects

| Tier | What goes there | Notes |
|---|---|---|
| **HF Hub private dataset repo** | manifests, chunk markers, features, checkpoints | The only durable store that both Colab and Kaggle can read/write cheaply. Versioned; resumable |
| **Kaggle Dataset** | processed features, as an immutable input | Mount at `/kaggle/input/<name>`; read-only; perfect for training notebooks |
| **Google Drive** | Colab checkpoints, run logs | 15 GB free tier; unreliable for many small files — zip or use HF instead |
| **`/kaggle/working`** | scratch only | 20 GB, wiped between sessions |
| **`/content`** | scratch only | wiped on disconnect |

**Checkpoint cadence:** every chunk (not every N minutes) — chunks are already sized to complete in
minutes. A chunk that finishes writes its artefact then its marker; a crash between them loses at
most one chunk.

**Anti-pattern to avoid:** writing thousands of small frame files to Drive. It is slow and hits
quota. Write one `.npy`/parquet per chunk.

---

## 5. Resumable, chunked video pipeline

```text
S0  plan      item manifest ──► plan_chunks(stage, size=100)  ──► plan.json
S1  acquire   per chunk: download archive slice OR generate N clips
              └─► writes clips/<chunk>.tar + marker(sha256)
S2  frames    per chunk: decode → 8–16 frames @224² → uint8 .npy
              └─► frames/<chunk>.npy + marker
S3  features  per chunk: frozen backbone → (N, D) float16 parquet
              └─► features/<chunk>.parquet + marker
S4  train     reads all features, group-split, train LR head (minutes, CPU)
S5  evaluate  frozen threshold, per-source and per-generator tables
```

Resume is: `store.pending(chunks, artefact_for=lambda c: path_for(c))`. Already implemented and
tested — including the case where an artefact exists but its hash does not match (interrupted
write), which is treated as *not done*.

**Idempotency:** re-running a completed chunk is a no-op. Re-running with a changed plan produces
new chunk keys (`stage-index-planhash`), so stale markers are ignored rather than silently reused.

---

## 6. Notebook layout

Notebooks are thin wrappers; logic lives in `ml/`. A notebook that contains the actual pipeline is
not reproducible.

```text
notebooks/
  nb00_env.ipynb        environment probe: GPU name, VRAM, disk, package versions
  nb01_plan.ipynb       build the item manifest and chunk plan (CPU, seconds)
  nb02_acquire.ipynb    run S1 for pending chunks (GPU if generating, CPU if downloading)
  nb03_frames.ipynb     run S2 (CPU, I/O bound)
  nb04_features.ipynb   run S3 (GPU)
  nb05_train.ipynb      run S4 (CPU)
  nb06_eval.ipynb       run S5 (CPU)
```

Each notebook starts with the same three lines — mount storage, restore state, print `pending` —
so it is obvious on entry whether there is work to do. If `pending == 0`, the notebook exits.

---

## 7. Checkpointing and experiment tracking

One JSON file per run, written to the Hub repo under `runs/`:

```json
{
  "run_id": "S3-2026-10-30T14:02Z",
  "stage": "features",
  "config": {"backbone": "clip-vit-b32", "frames": 12, "size": 224, "dtype": "float16"},
  "env": {"gpu": "Tesla T4", "vram_gb": 16.0, "torch": "...", "python": "3.11"},
  "input_manifest_sha256": "...",
  "chunks_total": 12, "chunks_done": 12,
  "wall_clock_s": 1840.2,
  "throughput": {"clips_per_s": 0.65, "frames_per_s": 7.8},
  "peak_vram_gb": 3.1,
  "status": "complete"
}
```

Rule: a metric may only be reported alongside the `run_id` and the `input_manifest_sha256` that
produced it. That is the same discipline already applied to text (`COMBINED.yaml` digest).

---

## 8. Sharing processed data without redistributing restricted media

Publish **features and manifests**, not frames, unless the source licence clearly allows it.

| Artefact | Pexels + Wan path | Vript path |
|---|---|---|
| Code, configs, notebooks | ✅ publish | ✅ publish |
| Item manifest (ids, hashes, counts) | ✅ publish | ✅ publish |
| **Frames** | ⛔ don't republish as stock-derived data | ⛔ |
| **Embeddings / features** | ✅ publish (not the imagery) | ⚠️ ask |
| Model weights | ✅ publish (both sources permit) | ⛔ withheld |
| Metrics, model card, eval set definitions | ✅ publish | ✅ publish |

This is what makes the Pexels + Wan route "legally publishable" while the Vript route stays
blocked: the former needs no restricted media at all.

---

## 9. Measurements to collect (the pilot's whole purpose)

Record these before any scaling decision. Each is currently unknown.

| # | Measurement | Status |
|---|---|---|
| 1 | Colab plan, units, units/hour per GPU class, expiry | ⬜ **needs your account** |
| 2 | Kaggle weekly GPU hours remaining | ⬜ **needs your account** |
| 3 | Actual GPU + VRAM on each platform | ⬜ needs a cloud session |
| 4 | HF download throughput (MB/s) | ⬜ not measured |
| 5 | **Seconds per Wan clip** (2-clip timing run) | ⬜ needs GPU |
| 6 | Peak VRAM during generation | ⬜ needs GPU |
| 7 | Frames/s extraction | ✅ **48.3 frames/s, 6.04 clips/s — CPU, 224², 8 frames** |
| 8 | **Disk per clip at 224²** | ✅ **1.204 MB at 8 frames ⇒ 1.81 MB at 12** |
| 9 | Feature extraction clips/s | ✅ **17.8 clips/s (placeholder extractor, CPU)** |
| 10 | Session survival / resume | ✅ **PASS — 0 chunks redone after rerun** |
| 11 | Kaggle `/kaggle/working` usable size | ⬜ needs Kaggle |
| 12 | Colab Drive write speed | ⬜ needs Colab |

Measured values come from `reports/video_poc.json` (24 synthetic clips, CPU-only machine).
They are real measurements of the *pipeline*, not of any detector.

### The probe is automated

`ml/cloud/probe.py` collects everything the runtime can expose (GPU model and VRAM, disk, RAM, throughput, peak VRAM, checkpoint writability) and writes JSON. Two notebooks wrap it: `notebooks/colab_measure.ipynb` and `notebooks/kaggle_measure.ipynb`. Each writes `reports/cloud/<host>_measure.json`.

Quota and session timeout are **not** exposed to code - the notebooks leave them null and say where to read them. A missing value is never guessed.

**Acceptance for the engineering prototype (decision 2)** — all must hold:

1. Frame extraction is deterministic (same clip → identical frames, byte-for-byte).
2. Resume after a forced disconnect re-does **zero** completed chunks.
3. A truncated artefact is detected and redone (tested in `tests/test_chunks.py`).
4. Splits group by **source video**, with 0 group overlaps (extends the existing video gate).
5. Frozen-backbone features are reproducible within float tolerance across two runs.
6. Measurements 3–10 are recorded in a run log.
7. The report is labelled **"pipeline validation — not a real-vs-generated detector"**.
8. No restricted media is published; no Vript media is downloaded.

**Explicitly not accepted:** any claim of detection accuracy from the generated-only prototype.

---

## 10. Smallest proof-of-concept book before any large download

| Item | Size | Purpose |
|---|---|---|
| `OpenAI_Sora.zip` | 468 MB | generated class, CC BY-NC, no Vript media |
| ~40 Pexels clips via API | ~100–200 MB | real class, permissive licence |
| Frames (52 clips × 12 × 224² × uint8) | ~30 MB | storage measurement |
| Features (52 × 512 float16) | ~50 KB | stage plumbing |

Total **< 700 MB**, roughly **1–2 GPU-hours**, and it exercises every stage end to end: acquire →
frames → features → split → train → evaluate → resume test. That is the correct thing to run next —
not Bundle B, and not 3 TB.

**What it can and cannot show:** it can prove the pipeline works and produce the measurements above.
It cannot show that AI-video detection is learnable, because one generator is not a distribution.

---

## 11. What I need from you

1. **Colab:** the 12 fields in `docs/colab-entitlement.md` (plan, units, per-class unit rates,
   assigned GPU, VRAM, session limits, throughput).
2. **Kaggle:** confirm the current weekly GPU allowance **in your account** — the public docs page
   is client-rendered and I could only obtain a quoted figure, not a verified one. Also confirm
   whether phone verification is required for GPU access.
3. **Pexels API key** (or a handful of manually downloaded clips) if you want the real-video side
   included in the next pilot run.

**Not needed:** any contact with dataset authors. Where terms are unclear, the pipeline uses a
clearly-permitted source instead.
