"""Build the E2-C1 prompt split manifest (§C.1).

Live/no-network. Loads the prompt manifest, validates it, computes its SHA-256,
assigns family-atomic partitions, validates the result, and only then writes the
split manifest atomically. Any validation failure refuses to write.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import prompts
from .constants import SPLIT_SALT
from .families import build_families
from .split import (
    assign_partitions,
    build_split_manifest,
    validate_split_manifest,
)

DEFAULT_PROMPTS = "datasets/manifests/e2_c1_prompts.jsonl"
DEFAULT_OUT = "datasets/manifests/e2_c1_splits.json"
DEFAULT_CONFIG = "ml/e2c1/generation_config.json"


class SplitValidationError(RuntimeError):
    """Raised when a split manifest violates its invariants; nothing is written."""

    def __init__(self, problems: Sequence[str]) -> None:
        self.problems = list(problems)
        super().__init__("; ".join(self.problems))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _atomic_write_json(path: Path, obj: Mapping[str, Any]) -> str:
    payload = (
        json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    descriptor, temp_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise
    return hashlib.sha256(payload).hexdigest()


def write_split(
    out_path: Path, split: Mapping[str, Any], prompt_rows: Sequence[Mapping[str, Any]]
) -> str:
    """Validate then atomically write. Returns the split digest; never partial."""
    problems = validate_split_manifest(split, prompt_rows)
    if problems:
        raise SplitValidationError(problems)
    _atomic_write_json(out_path, split)
    return str(split["split_digest"])


def _families_from_rows(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    families: dict[str, str] = {}
    missing = False
    for row in rows:
        prompt_id = str(row["prompt_id"])
        family_id = row.get("family_id")
        if family_id:
            families[prompt_id] = str(family_id)
        else:
            missing = True
    if missing:
        return build_families(rows)
    return families


def build(
    prompts_path: Path,
    config_path: Path,
    *,
    salt: str = SPLIT_SALT,
) -> dict[str, Any]:
    """Build (but do not write) a split manifest, raising on bad input.

    ``SplitValidationError`` is used both for an invalid prompt manifest and an
    invalid split manifest so the CLI and tests share one refusal path.
    """
    rows = prompts.read_manifest(Path(prompts_path))
    config = load_config(config_path)

    expected = int(config.get("expected_prompts", len(rows)))
    problems = prompts.validate_manifest(rows, expected=expected)
    if problems:
        raise SplitValidationError(problems)

    families = _families_from_rows(rows)
    assignments = assign_partitions(rows, families, salt=salt)
    return build_split_manifest(
        rows,
        assignments,
        config,
        _sha256_file(Path(prompts_path)),
        salt=salt,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the E2-C1 prompt split manifest (offline)."
    )
    parser.add_argument("--prompts", default=DEFAULT_PROMPTS, help="prompt manifest JSONL")
    parser.add_argument("--out", default=DEFAULT_OUT, help="split manifest JSON output")
    parser.add_argument(
        "--config", default=DEFAULT_CONFIG, help="generation config JSON"
    )
    parser.add_argument("--salt", default=SPLIT_SALT, help="split salt")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)

    try:
        split = build(Path(args.prompts), Path(args.config), salt=args.salt)
    except FileNotFoundError as exc:
        print(f"input not found: {exc}", file=sys.stderr)
        return 2
    except (json.JSONDecodeError, ValueError) as exc:
        print(f"invalid input: {exc}", file=sys.stderr)
        return 2
    except SplitValidationError as exc:
        for problem in exc.problems:
            print(f"refused: {problem}", file=sys.stderr)
        return 2

    try:
        digest = write_split(Path(args.out), split, prompts.read_manifest(Path(args.prompts)))
    except SplitValidationError as exc:
        for problem in exc.problems:
            print(f"refused: {problem}", file=sys.stderr)
        return 2

    print(f"wrote {args.out}")
    print(f"split_digest={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
