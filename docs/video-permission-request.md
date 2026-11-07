# Permission questions — ARCHIVED, NOT SENT, NOT A DEPENDENCY

> **Status: not sent. Not scheduled. Not a prerequisite for any work.**
>
> Project policy is that no experiment waits on a dataset maintainer
> (`DATASET_REGISTRY.md` §0.1). This file is retained only as a record of *which questions would
> matter* if the situation ever changed, and to justify why certain media are unused.

---

## Why this file still exists

Three video datasets carry terms that are silent or ambiguous for operations we care about:

| Dataset | Ambiguity | What we do instead |
|---|---|---|
| Vript / Vript-RR | "no modification" may cover frame extraction; weights unaddressed | do not extract frames from Vript; use Pexels |
| GenVidBench | CC BY-NC does not say whether trained weights are covered | do not publish GenVidBench-trained weights |
| HD-VG-130M | bespoke PDF agreement, unread | do not use Pair2 real clips |

Rather than ask, we **route around** each. The questions below are therefore documentary, not
pending.

---

## The questions (for the record only)

**Vript / Vript-RR**
1. Is decoding and frame extraction permitted, given the "no modification" clause?
2. May evaluation code, results and manifests be published?
3. May model weights trained on the data be published for non-commercial research?

**GenVidBench**
1. Does the CC BY-NC 4.0 licence extend to trained model weights?
2. Is there a published mapping from generated clip UUIDs to source scenes or prompts?

**HD-VG-130M**
No question — the licence document simply needs reading. It is a PDF agreement, not an OSI licence.

---

## If you ever do send these

Nothing in the repository depends on the answers, so treat any reply as a bonus that may *expand*
what is publishable. Record the reply verbatim in `docs/licences/` with a date, and update
`DATASET_REGISTRY.md` §3 accordingly.

A non-answer changes nothing: the current routing already assumes "not permitted".
