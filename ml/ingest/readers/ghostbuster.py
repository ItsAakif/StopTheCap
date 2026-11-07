"""Ghostbuster data reader (CC BY 3.0).

Layout verified 2026-10-29 by direct inspection of the cloned repository:

    essay/<variant>/<id>.txt          student essays        domain=student_essay
    wp/<variant>/<id>.txt             news                  domain=news
    reuter/<variant>/<author>/<id>.txt news, author-conditioned  domain=news_reuters
    other/gptzero/<variant>/<id>.txt  news (50 pairs)       domain=news
    other/{bawe,ets,lang8,pelic,toefl91,undetectable}/     EXCLUDED
    perturb/<kind>/<level>/<id>.txt                        EXCLUDED

variants (label 1 unless stated): gpt, claude, gpt_prompt1, gpt_prompt2,
gpt_semantic, gpt_writing;  human (label 0).

Why some directories are excluded
---------------------------------
The authors' label rule in ``train.py`` is a *path substring match*:

    lambda file: 1 if any([m in file for m in ["gpt", "claude"]]) else 0

That labels ``other/undetectable/`` (a detector-evasion tool's output -- i.e. machine
text) as **human**, and says nothing about ``bawe``/``ets``/``lang8``/``pelic``/
``toefl91``. Rather than inherit an unverifiable label we exclude those directories
and count them. ``perturb/`` holds perturbation variants of text whose base label is
undocumented in the repository, so it is excluded from v1 as well; it is noted as the
natural home for a future robustness suite.

Grouping (the leakage control)
------------------------------
* ``essay``/``wp``: the **file stem** pairs human and machine versions of the same
  prompt. Verified: ``essay/human/1.txt`` and ``essay/gpt/1.txt`` are both essays
  about ``12 Years a Slave``.
* ``reuter``: grouped by **author**, which is the more conservative key -- it keeps
  every article by one author in one split, preventing author-style leakage as well
  as prompt pairing.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterator

from ..schema import Modality, Task
from .base import RawSample

_HUMAN_VARIANT = "human"

# (top-level dir, domain, grouping mode)
_DOMAINS: dict[str, tuple[str, str]] = {
    "essay": ("student_essay", "stem"),
    "wp": ("news", "stem"),
    "reuter": ("news_reuters", "author"),
    "other/gptzero": ("news", "stem"),
}

EXCLUDED_DIRS: dict[str, str] = {
    "other/bawe": "human corpus; label and pairing unverifiable from repository",
    "other/ets": "human corpus; label and pairing unverifiable from repository",
    "other/lang8": "human corpus; label and pairing unverifiable from repository",
    "other/pelic": "human corpus; label and pairing unverifiable from repository",
    "other/toefl91": "human corpus; label and pairing unverifiable from repository",
    "other/undetectable": "machine (evasion tool) but authors' path rule labels it human",
    "perturb": "perturbation suite; base-label provenance undocumented",
    "perturb_old": "perturbation suite (legacy); base-label provenance undocumented",
    "essay/prompts": "prompt text, not a sample",
    "wp/prompts": "prompt text, not a sample",
}


class GhostbusterReader:
    name = "ghostbuster"

    def read(self, root: Path) -> Iterator[RawSample]:
        # Phase 1: enumerate deterministically (sorted) so output order is stable.
        plan: list[tuple[Path, str, str, str, str, tuple[str, ...]]] = []
        for top, (domain, mode) in _DOMAINS.items():
            base = root / top
            if not base.is_dir():
                continue
            for f in sorted(base.rglob("*.txt")):
                rel = f.relative_to(root).as_posix()
                if any(rel.startswith(ex) for ex in EXCLUDED_DIRS):
                    continue
                parts = f.relative_to(base).parts
                if len(parts) < 2:
                    continue  # e.g. a stray file at the domain root
                if mode == "author" and len(parts) < 3:
                    continue
                plan.append((f, rel, parts[0], domain, mode, parts))

        # Phase 2: read concurrently -- this is 22k small files on Windows, where
        # per-file I/O latency dominates. Order is preserved by map().
        def _read(p: Path) -> str:
            return p.read_text(encoding="utf-8", errors="replace")

        with ThreadPoolExecutor(max_workers=16) as pool:
            texts = list(pool.map(_read, [entry[0] for entry in plan]))

        for (f, rel, variant, domain, mode, parts), text in zip(plan, texts):
            is_human = variant == _HUMAN_VARIANT
            label = 0 if is_human else 1
            generator = None if is_human else variant

            if mode == "author":
                author = parts[1]
                stem = f.stem
                group_id = f"ghostbuster:{domain}:{author}"
                extra = {"author": author, "file_stem": stem}
            elif variant == "claude":
                # `claude/` is NOT index-aligned with `human/`: verified 2026-10-29
                # that claude/5 discusses British air strategy while human/5 and
                # gpt/5 both discuss the medical-office prompt. Grouping it by stem
                # would falsely merge unrelated documents, so it gets its own key.
                stem = f.stem
                group_id = f"ghostbuster:{domain}:claude:{stem}"
                extra = {"file_stem": stem, "pairing": "unpaired_different_prompts"}
            else:
                stem = f.stem
                group_id = f"ghostbuster:{domain}:{stem}"
                extra = {"file_stem": stem}

            yield RawSample(
                source_sample_id=rel,
                original_label=variant,
                label=label,
                task=Task.TEXT_BINARY,
                modality=Modality.TEXT,
                text=text,
                domain=domain,
                generator=generator,
                group_id=group_id,
                source_uri=("https://github.com/vivek3141/ghostbuster-data/"
                            f"blob/master/{rel}"),
                metadata={"ghostbuster_dir": rel, "variant": variant, **extra},
            )
