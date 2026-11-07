# E2 training readiness - decision

**Date:** 2026-11-06. Supersedes the composition proposal in `E2-dataset-audit-and-plan.md` §4.
E1 untouched; evaluation sets untouched.

---

## 1. Can the Pile's components be separated? Yes.

The `id` column is **contiguous and monotonic** (0 to 1,418,655), shards are ordered by id, and all
human rows precede all AI rows. Measuring the exact-match rate against the full HC3 hash set per
50,000-row id band shows HC3 concentrated in **contiguous blocks**:

| id band | HC3-match rate | Content sample |
|---|---:|---|
| 0 - 349,999 | 0.0% | essay-mill titles ("Absolut Vodka Report", "7FE Project Framework") |
| 350,000 - 399,999 | 18.7% | mixed |
| **400,000 - 449,999** | **91.9%** | short forum answers |
| 450,000 - 1,049,999 | 0.0% | Reddit WritingPrompts fiction ("Harambe is dead", sonnets) |
| 1,050,000 - 1,349,999 | 0.0% | news-style prose ("The Canadian Press", "city will pay $18.25 million") |
| **1,350,000 - 1,399,999** | **19.9%** | how-to text |
| **1,400,000 - 1,449,999** | **60.1%** | short advice answers |

So the components are separable by id range, and the HC3 blocks are identifiable by content.

**But separation does not create permission.** The task was to find components with *documented*
training rights, and that is where the Pile fails.

## 2. Which components have documented permission for academic model training?

| Component (id range, inferred) | Underlying rights holder | Documented permission for training? |
|---|---|---|
| Essay-mill essays (0 - ~350k) | ivypanda-type commercial essay sites | **None.** Scraped commercial content |
| HC3 human (~350k - 450k) | HC3 (CC BY-SA 4.0) | Yes - **but this is one of our evaluation sets. Excluded by design** |
| Reddit WritingPrompts (~450k - 1.05M) | Reddit users (fanfiction) | **None.** User-generated text, no training grant |
| WebText-like news (~1.05M - 1.35M) | Scraped web publishers | **None.** OpenAI's own WebText release carries a *takedown* contact, not a licence |
| HC3 ChatGPT (1.35M - 1.42M) | HC3 (CC BY-SA 4.0) | Yes - **evaluation set. Excluded** |
| GPT-2 / GPT-3 / GPT-J / ChatGPT outputs | respective providers | **Not verified.** No per-row provenance; API output terms vary and no record exists |

**Conclusion: no defensible subset of the Pile exists.** Every component is either unlicensed for
training, or is one of the two corpora reserved for independent evaluation. Separation solves a
technical problem, not the rights problem - and the task is explicit that rights must not be inferred
from the uploader's MIT tag.

**The Pile is dropped from E2.**

## 3. Alternatives, compared

| Candidate | Generators | Human diversity | Size | Licence for training | Verdict |
|---|---|---|---|---|---|
| **Ghostbuster** (incumbent) | GPT-3 davinci, Claude | Wikipedia-derived essays, news, Reuters | 22,100 | **CC BY 3.0 - clear** | keep |
| HC3 | ChatGPT | 5 domains | 85,449 | CC BY-SA 4.0 - clear | **reserved: evaluation** |
| RAID | 11 models | 8 domains | 4.98M | MIT - clear | **reserved: evaluation** |
| M4 / M4GT-Bench | 6+ | 7 domains | ~1M | **none declared** | excluded |
| ArguGPT | 7 GPT-family | TOEFL essays | ~4.7k machine | **`cc`, variant unstated** | excluded |
| AI-GA | GPT-3 only | CORD-19 abstracts | 28,662 | MIT wrapper, **CORD-19 underneath** | excluded |
| **openai/gpt-2-output-dataset** | **GPT-2 (4 sizes)** | **WebText test, 250k** | 250k paired | repo **MIT**, but **data licence not stated** - research release with takedown contact | **interim option** |
| Self-generated (Mistral, Llama) | **Mistral, Llama** - the families RAID tests | any permissively licensed prompts | we choose | **clear**: our outputs from Apache-2.0 models | **recommended** |

## 4. Decision

> **E2 is NOT ready to train.**

**The blocker is precise:** there is no downloadable corpus with documented permission covering both
classes for model training, other than the two corpora we have reserved for evaluation. Every
candidate is excluded for a specific, recorded reason - no candidate is excluded because it is
"academic-only".

### Recommended E2: constructed, not downloaded

**E2-C1 = Ghostbuster + 20,000 generations from Apache-2.0 models on CC BY-SA prompts.**

| Element | Choice | Why |
|---|---|---|
| Human prompts | Wikipedia paragraphs (CC BY-SA 4.0, attributed) | documented licence; already the domain of Ghostbuster's human side, so the *generator* is the only variable |
| Generators | **Mistral-7B-Instruct** and **Llama-3.2-3B-Instruct** (both Apache-2.0) | these are the families RAID actually tests - the gap that explains 0.532 |
| Volume | 10,000 per model, 2 decoding settings | comparable to E1's scale; ~2 h on a T4 |
| Licence | our outputs, from Apache-2.0 weights, on CC BY-SA prompts | fully documented end to end |
| Contamination | prompts are Ghostbuster's own domain, so dedup against HC3/RAID is still required and cheap | |

This is the only composition that is both **defensible** and **targeted at the measured failure**. It
is generation work rather than a download, which is why it was previously scoped as E3 - but the
licence audit has now made it the *only* viable E2.

### Interim option if you want to train sooner

`openai/gpt-2-output-dataset` gives 250k WebText documents paired with GPT-2 outputs, released by
OpenAI for research. It is better provenance than the Pile, but **the README does not license the
data files** - only the repository code is MIT. Under our own rule that is *unverified*, so it would
have to be recorded as a research-use assumption, not a permission. It also adds only GPT-2, which
is unlikely to move RAID.

**Recommendation: do not use the interim option.** It trades a documented blocker for an undocumented
assumption to save two hours of generation.

## 5. What exists now, and what does not

| Artefact | Status |
|---|---|
| Pile component map (id ranges) | produced, documented above |
| Pile decontamination run | complete, retained as evidence (`fd012688...`) - but **not used** |
| E2 training manifest | **does not exist** |
| E2 training digest | **does not exist** |
| E2 training run | **not started** |

The decontamination work is not wasted: the same pipeline and thresholds are what the constructed
corpus will run through before training.

## 6. Unblocking steps, in order

1. Generate 20,000 completions on Colab (Mistral-7B-Instruct, Llama-3.2-3B-Instruct; Apache-2.0).
2. Run the existing decontamination against HC3, RAID and Ghostbuster.
3. Build the manifest, produce the digest, run the 12-gate audit.
4. Only then train E2-C1 with E1's exact configuration.

Step 1 is the only thing that needs a GPU session and cannot be done locally.
