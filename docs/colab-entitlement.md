# Colab entitlement measurement

**Why this file exists:** no compute-unit figure appears anywhere in this project's
planning documents. Google Colab's pricing page is behind a sign-in wall, so the
account's actual entitlement, the per-hour unit cost of each accelerator class, and
the assigned GPU cannot be read without you. Anything written down without this
measurement would be fabricated.

**Complete this before approving any training run.** It takes ~30 minutes and converts
every later estimate from a guess into arithmetic.

Record the answers in the table at the bottom and commit this file.

---

## The 12 numbers to collect (paste these back)

Fill this exact table. Nothing in the project can quote a compute budget without it.

| # | Field | Where to find it | Format |
|---|---|---|---|
| 1 | Plan name | Colab purchase panel | text (e.g. "Pro") |
| 2 | Compute units available | purchase panel | integer |
| 3 | Unit expiry date | purchase panel | ISO date |
| 4 | T4 units/hour | purchase panel / FAQ | float |
| 5 | L4 units/hour | purchase panel / FAQ | float |
| 6 | A100 units/hour | purchase panel / FAQ | float |
| 7 | Assigned GPU name | `!nvidia-smi` | text |
| 8 | VRAM (MiB) | `!nvidia-smi` | integer |
| 9 | Idle disconnect timeout | runtime menu / docs | minutes |
| 10 | Max session length | runtime menu / docs | hours |
| 11 | `examples_per_sec` | benchmark below | float |
| 12 | Peak training VRAM (GB) | benchmark below | float |

From (9)–(12) the whole plan is computable:

```
hours_per_epoch = n_train_examples / examples_per_sec / 3600
total_hours     = hours_per_epoch × epochs
units_needed    = total_hours × units_per_hour(gpu_class)
```

With the current split (11,150 train rows, 3 epochs) and a `distilbert-base-cased` run, the
projected cost is `33,450 / examples_per_sec / 3600 × 3` hours. **That formula is the deliverable —
not a guessed number.**

Also record whether a **free-tier T4** is available in addition to the paid allowance; it changes
the fallback plan.

---

## 1. Entitlement

1. Open <https://colab.research.google.com> and sign in with the account holding
   Google AI Pro.
2. Open the **purchase / compute-units** panel. Paths change; look for
   *Help → View compute units* or the resource pane in the top-right.
   Direct link that usually works: <https://colab.research.google.com/settings>
3. Record:
   - plan name shown (e.g. "Pro", "Pro+")
   - compute units currently available
   - expiry date of those units
   - **the per-hour unit cost displayed for each accelerator class** (T4, L4, A100, …)
   - whether a free-tier T4 is additionally available

> If the panel shows only "units", record the unit cost table separately — it is
> usually visible in the same dialog or at
> <https://research.google.com/colaboratory/faq.html#resource-limits> (limits page).

## 2. Assigned hardware

```python
!nvidia-smi
```

Record: GPU name, total VRAM, driver/CUDA version. **Do not assume L4.** Colab assigns
whatever is free; a plan does not guarantee a specific accelerator.

## 3. Session limits

Record the idle timeout and maximum session length shown in the runtime menu, and
whether the UI warns before disconnection.

## 4. Throughput benchmark (the number that matters)

```python
# Run once per encoder you intend to train.
import time, torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

name = "distilbert-base-cased"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=2).cuda().half()
opt = torch.optim.AdamW(model.parameters(), lr=5e-5)

B, L = 16, 512
ids = torch.randint(0, tok.vocab_size, (B, L), device="cuda")
mask = torch.ones_like(ids)
y = torch.randint(0, 2, (B,), device="cuda")

def step():
    opt.zero_grad()
    out = model(input_ids=ids, attention_mask=mask, labels=y)
    out.loss.backward()
    opt.step()

for _ in range(5): step()          # warm-up, discard
torch.cuda.synchronize()
t0 = time.time()
N = 50
for _ in range(N): step()
torch.cuda.synchronize()
dt = time.time() - t0

print(f"steps/sec={N/dt:.3f}  examples/sec={N*B/dt:.2f}  "
      f"peak_GB={torch.cuda.max_memory_allocated()/1e9:.2f}")
print(f"GPU={torch.cuda.get_device_name(0)}")
```

Record `examples/sec` and `peak_GB`. Then estimate any run as:

```
wall_clock_hours = (n_examples × epochs) / examples_per_sec / 3600
```

## 5. Persistence and resume

1. Mount Drive: `from google.colab import drive; drive.mount('/content/drive')`
2. Write a small file to Drive, then **Runtime → Disconnect and delete runtime**.
3. Reconnect and confirm the file is still present.
4. Confirm your training script can resume from a checkpoint on Drive.

If step 3 fails, training cannot be trusted to survive a session and the run strategy
must change before any long job.

## 6. Environment capture

```python
!pip freeze > /content/drive/MyDrive/stopthecap/pip-freeze-$(date +%F).txt
```

Commit nothing from Colab directly; copy the freeze file into `docs/colab/` if a run
informs a reported number.

---

## Recorded results

Measured 2026-11-04 on the owner's Google AI Pro account. Source file:
reports/cloud/colab_measure.json (committed).

| Field | Value |
|---|---|
| Plan | Google AI Pro (account-level; not visible to code) |
| Host | Colab, release-colab-external-images_20261102-060107_RC00 |
| Assigned GPU | **Tesla T4**, 15.64 GB VRAM, sm_75, 40 SMs |
| Driver / CUDA | 580.82.07 / CUDA 13.0 |
| PyTorch | **2.11.0+cu130** (CUDA build, cuda_available: true) |
| Python | 3.13.16 |
| CPU | 2 vCPU, torch.get_num_threads() = **1** |
| RAM | 13.29 GB total, 9.59 GB available |
| Disk | 120.5 GB free at /content; 114.5 GB free on Drive |
| Throughput (distilbert, B=8, L=128) | **10.46 steps/s, 83.65 examples/s** |
| Peak VRAM | **1.356 GB** |
| Checkpoint dir writable | yes - /content/drive/MyDrive/StopTheCap_reports/ckpt_probe |
| Drive persistence across disconnect | NOT YET TESTED |
| Remaining GPU quota | **still null** - read it in the Colab purchase panel |
| Session timeout | **4 h 40 min** - Colab message: "At your current usage level, this runtime may last up to 4 hours 40 minutes" (2026-11-04). Usage-dependent, not a guarantee |

## Derived E1 budget (ESTIMATE, from the measured rate)

Measured throughput is **model compute only**: the benchmark used synthetic tensors, so it excludes
tokenisation and data loading and overstates real training speed.

| Assumption | Result |
|---|---|
| Scale L=128 to 512 (x3.5-4) | ~21-24 examples/s |
| 11,150 rows x 3 epochs = 33,450 passes | **~23-27 min of pure GPU compute** |
| Add tokenisation on 1 thread plus validation | expect **35-60 min wall clock** |
| VRAM at B=16, L=512 (~8x the measured activation footprint) | **~6-9 GB** of 15.64 GB - fits |

**Recommendation:** keep batch_size 16, grad_accum 2, fp16 true. Do **not** use bf16 - T4 is sm_75
and has no native bf16. Raise torch.set_num_threads(2) and consider pre-tokenising once, because
2 vCPUs with 1 torch thread will gate the run before the GPU does.

## Colab vs Kaggle - what the 4 h 40 min figure changes

The 4 h 40 min limit is an *estimate at current usage level*, not a contract. Two consequences:

- **E1 (~1 h) is unaffected.** No reason to move it.
- **Video generation is affected.** 100 Wan clips at an estimated 8-16 min each is 13-27 GPU-hours,
  which is 3-6 Colab sessions. Kaggle's 12 h session cap and stated weekly allowance fit that
  shape far better, so Kaggle is the better host for the video workstream and a quota backstop if
  Colab compute units run out.

`notebooks/kaggle_measure.ipynb` is prepared for the same probe using a **private Kaggle Dataset**
(`/kaggle/input/<slug>/StopTheCap.zip`) instead of Drive. Nothing has been run there yet.

## TPU note

Colab also offers a **v5e-1 TPU** runtime. Not recommended for E1:

- PyTorch on TPU needs torch_xla; our evaluator uses **dynamic padding**, so every distinct batch
  shape triggers an XLA recompilation - pathological for a short run.
- The measured T4 already runs E1 in well under an hour, so there is no compute pressure to relieve.
- T4 is fp16; TPU is bf16, so results would not be directly comparable without re-running baselines.

Revisit TPU only if we later train something much larger or need much longer sequences.
