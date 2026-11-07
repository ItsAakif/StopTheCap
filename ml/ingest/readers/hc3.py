"""HC3 reader (CC BY-SA 4.0) -- EVALUATION ONLY.

HC3 is designated evaluation-only by the research decision (research/13 s3): it has a
single generator (ChatGPT, 2023) and must never enter a training path. The pipeline
enforces this via the manifest (``permitted_uses.train: false``) and by assigning
every HC3 record to ``Split.QUARANTINE``.

Record format (verified 2026-10-29) is one JSON object per line:

    {"question": ..., "human_answers": [...], "chatgpt_answers": [...], "source": ...}

Each line yields one record per answer. All answers to one question share a
``group_id`` so a question cannot straddle a split.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from ..schema import Modality, Task
from .base import RawSample

# subset file -> domain label
_SUBSETS = {
    "finance.jsonl": "finance",
    "medicine.jsonl": "medicine",
    "open_qa.jsonl": "open_qa",
    "reddit_eli5.jsonl": "reddit",
    "wiki_csai.jsonl": "wiki",
}


class HC3Reader:
    name = "hc3"

    def read(self, root: Path) -> Iterator[RawSample]:
        for fname, domain in _SUBSETS.items():
            path = root / fname
            if not path.is_file():
                continue
            with path.open(encoding="utf-8") as fh:
                for lineno, line in enumerate(fh):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    question = (obj.get("question") or "").strip()
                    group = f"hc3:{domain}:{lineno}"

                    for i, ans in enumerate(obj.get("human_answers") or []):
                        if not isinstance(ans, str) or not ans.strip():
                            continue
                        yield RawSample(
                            source_sample_id=f"{domain}:{lineno}:human:{i}",
                            original_label="human",
                            label=0,
                            task=Task.TEXT_BINARY,
                            modality=Modality.TEXT,
                            text=ans,
                            domain=domain,
                            generator=None,
                            group_id=group,
                            source_uri=(f"https://huggingface.co/datasets/Hello-SimpleAI/HC3/"
                                        f"blob/main/{fname}#L{lineno + 1}"),
                            metadata={"subset": fname, "question": question, "answer_index": i},
                        )

                    for i, ans in enumerate(obj.get("chatgpt_answers") or []):
                        if not isinstance(ans, str) or not ans.strip():
                            continue
                        yield RawSample(
                            source_sample_id=f"{domain}:{lineno}:chatgpt:{i}",
                            original_label="chatgpt",
                            label=1,
                            task=Task.TEXT_BINARY,
                            modality=Modality.TEXT,
                            text=ans,
                            domain=domain,
                            generator="chatgpt",
                            group_id=group,
                            source_uri=(f"https://huggingface.co/datasets/Hello-SimpleAI/HC3/"
                                        f"blob/main/{fname}#L{lineno + 1}"),
                            metadata={"subset": fname, "question": question, "answer_index": i},
                        )
