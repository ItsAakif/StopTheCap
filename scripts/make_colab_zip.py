"""Build the private Colab ZIP.

Excludes everything that is large, restricted or machine-local. The Colab measurement notebook
needs only code plus a few manifests, so the archive stays small and contains no media.

Usage:
    python scripts/make_colab_zip.py --out StopTheCap.zip
    python scripts/make_colab_zip.py --out StopTheCap.zip --with-e2c1
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path
from typing import Iterable

REPO = Path(__file__).resolve().parents[1]

EXCLUDE_DIRS = {
    ".git", ".venv", "venv", "__pycache__", ".pytest_cache", ".mypy_cache",
    ".ipynb_checkpoints", "node_modules", "checkpoints",
    "datasets/raw", "datasets/canonical", "datasets/splits",
}

# Canonical records the training notebook needs. Text only - no video inventory, no media.
DATA_FILES = (
    "datasets/canonical/ghostbuster.jsonl",
    "datasets/canonical/hc3.jsonl",
    "datasets/canonical/raid.jsonl",
)

# Pre-generation artefacts the generation notebook needs. They are not training data, so
# they stay out of DATA_FILES (which is gated behind --with-data).
E2C1_FILES = (
    "datasets/manifests/e2_c1_prompts.jsonl",
    "datasets/manifests/e2_c1_prompt_audit.json",
    "datasets/manifests/e2_c1_splits.json",
    "datasets/manifests/e2_c1_licences.json",
    "ml/e2c1/generation_config.json",
)

EXCLUDE_SUFFIXES = {".pt", ".ckpt", ".onnx", ".safetensors", ".pyc", ".zip", ".mp4", ".rar", ".7z"}
# Restricted or bulky evidence data, re-downloadable via the scripts beside them.
EXCLUDE_NAME_PATTERNS = ("_evidence/*.csv", "_evidence/*.jsonl")

# A --with-e2c1 archive must never carry raw/canonical data or model/archive binaries.
FORBIDDEN_PREFIXES = ("StopTheCap/datasets/raw/", "StopTheCap/datasets/canonical/")
FORBIDDEN_SUFFIXES = (".pt", ".safetensors", ".ckpt", ".zip")


def _rel(p: Path, root: Path = REPO) -> str:
    return p.relative_to(root).as_posix()


def should_skip(path: Path, include_data: bool = False, root: Path = REPO) -> bool:
    rel = _rel(path, root)
    if include_data and rel in DATA_FILES:
        return False
    # Keep the explicit E2-C1 allowlist ahead of EXCLUDE_DIRS so a future change to the
    # dataset exclusions can never silently drop a required generation input.
    if rel in E2C1_FILES:
        return False
    for d in EXCLUDE_DIRS:
        if rel == d or rel.startswith(d + "/"):
            return True
    if path.suffix.lower() in EXCLUDE_SUFFIXES:
        return True
    for pat in EXCLUDE_NAME_PATTERNS:
        if Path(rel).match(pat):
            return True
    return path.name in {".env"} or path.name.endswith(".local")


def iter_files(root: Path = REPO, include_data: bool = False) -> Iterable[Path]:
    """Yield the files that belong in the archive, in the same order as before."""
    root = Path(root)
    for p in sorted(root.rglob("*")):
        if p.is_dir() or should_skip(p, include_data, root=root):
            continue
        yield p


def build_zip(
    out: Path,
    root: Path = REPO,
    with_data: bool = False,
    with_e2c1: bool = False,
) -> tuple[Path, int, int]:
    """Write the archive and return ``(out, file_count, byte_size)``.

    ``with_e2c1`` is accepted for symmetry with the CLI; the allowlist in
    :func:`should_skip` always includes :data:`E2C1_FILES`, and the CLI performs the
    existence and membership verification.
    """
    del with_e2c1  # inclusion is unconditional; the flag only gates CLI validation
    out = Path(out)
    root = Path(root)
    out.parent.mkdir(parents=True, exist_ok=True)
    files = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for p in iter_files(root, include_data=with_data):
            zf.write(p, f"StopTheCap/{_rel(p, root)}")
            files += 1
    return out, files, out.stat().st_size


def missing_e2c1_files(root: Path = REPO) -> list[str]:
    """Return the E2-C1 paths that are absent under ``root`` (empty means complete)."""
    root = Path(root)
    return [rel for rel in E2C1_FILES if not (root / rel).is_file()]


def verify_zip_members(
    names: Iterable[str],
    required: Iterable[str] = E2C1_FILES,
) -> list[str]:
    """Return membership problems; an empty list means the archive is acceptable.

    ``names`` are the archive's full member names (``StopTheCap/...``). ``required``
    are repo-relative paths that must be present.
    """
    names = [str(name) for name in names]
    problems: list[str] = []
    for rel in required:
        member = f"StopTheCap/{rel}"
        if member not in names:
            problems.append(f"missing required member: {member}")
    for name in names:
        for prefix in FORBIDDEN_PREFIXES:
            if name.startswith(prefix):
                problems.append(f"forbidden path in zip: {name}")
        lowered = name.lower()
        if lowered.endswith(FORBIDDEN_SUFFIXES):
            problems.append(f"forbidden file type in zip: {name}")
    return problems


def verify_zip(out: Path, required: Iterable[str] = E2C1_FILES) -> list[str]:
    """Verify an existing archive against :func:`verify_zip_members`."""
    with zipfile.ZipFile(out) as zf:
        return verify_zip_members(zf.namelist(), required)


def main(argv=None, root: Path = REPO) -> int:
    root = Path(root)
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(root / "StopTheCap.zip"))
    ap.add_argument("--with-data", action="store_true",
                    help="include datasets/canonical text records (needed for training/evaluation)")
    ap.add_argument("--with-e2c1", action="store_true",
                    help="require and verify the E2-C1 pre-generation artefacts")
    args = ap.parse_args(argv)

    if args.with_e2c1:
        missing = missing_e2c1_files(root)
        if missing:
            print("missing required E2-C1 files (build them first):", file=sys.stderr)
            for rel in missing:
                print(f"  {rel}", file=sys.stderr)
            return 2

    out, files, size = build_zip(
        Path(args.out),
        root=root,
        with_data=args.with_data,
        with_e2c1=args.with_e2c1,
    )

    if args.with_e2c1:
        problems = verify_zip(out)
        if problems:
            print("ZIP verification failed:", file=sys.stderr)
            for problem in problems:
                print(f"  {problem}", file=sys.stderr)
            return 2

    print(f"wrote {out}")
    print(f"files: {files}   size: {size / 1e6:.2f} MB")
    print("no datasets/raw, no checkpoints, no media." + ("" if not args.with_data else
          " Includes canonical TEXT records for Colab training."))
    if args.with_e2c1:
        print("E2-C1 artefacts verified: 5 required members present, no forbidden paths.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
