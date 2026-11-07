"""Identity helpers shared by the E2-C1 split and generation modules.

``make_sample_id`` is defined here (not in ``generation.py``) so that the
pre-generation split module can compute the planned sample ids without importing
the generation module (which is torch-adjacent). ``ml.e2c1.generation`` should
import this function rather than re-implement it.
"""

from __future__ import annotations

import hashlib


def make_sample_id(
    prompt_id: str,
    generator_id: str,
    model_revision: str,
    decoding_config_id: str,
) -> str:
    """Stable sample id (§D.5).

    The four components are separated explicitly so concatenation is unambiguous.
    ``model_revision`` is part of the id so a model update cannot collide with an
    older run; quantisation deliberately is not (``config_digest`` disambiguates
    runs instead).
    """
    digest = hashlib.sha256(
        f"{prompt_id}|{generator_id}|{model_revision}|{decoding_config_id}".encode("utf-8")
    ).hexdigest()
    return "gen:" + digest[:32]
