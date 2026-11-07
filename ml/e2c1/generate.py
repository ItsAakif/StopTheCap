"""CLI for the E2-C1 generation contract (§D).

Argument parsing happens **before** any model import. ``--help``,
``--dry-run`` and ``--plan-only`` therefore work on a CPU-only machine with
torch absent; only a real generation run imports torch/transformers.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from pydantic import ValidationError

from .generation import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_OUT_DIR,
    REPO_ROOT,
    AuthorisationError,
    GenerationError,
    LicenceValidationError,
    ResumeCorruptionError,
    ResumeMismatch,
    SmokeOutputError,
    load_config,
    load_registry,
    run_generation,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ml.e2c1.generate",
        description=(
            "Generate the E2-C1 sample matrix (2 generators x 2 decodings x "
            "prompts). Dry-run/plan-only never import torch; a full run requires "
            "--authorise-full-run."
        ),
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="generation config JSON (default: ml/e2c1/generation_config.json)",
    )
    parser.add_argument(
        "--prompts",
        default=None,
        help="prompt manifest JSONL (default: config.prompt_manifest)",
    )
    parser.add_argument(
        "--splits",
        default=None,
        help="split manifest JSON (default: config.split_manifest)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help=f"output directory (full-run default: {DEFAULT_OUT_DIR})",
    )
    parser.add_argument(
        "--smoke",
        type=int,
        default=0,
        metavar="N",
        help="select N prompts per generator; ephemeral, never under datasets/",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the planned stream matrix and sample-id count only",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="alias of --dry-run (prints the plan and writes no records)",
    )
    parser.add_argument(
        "--generator",
        action="append",
        default=None,
        help="restrict to a generator id (repeatable)",
    )
    parser.add_argument(
        "--decoding",
        action="append",
        default=None,
        help="restrict to a decoding config id (repeatable)",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="re-run ids present only in failures-*.jsonl",
    )
    parser.add_argument(
        "--report",
        default=None,
        help="write the run/plan report JSON here",
    )
    parser.add_argument(
        "--max-prompts",
        type=int,
        default=None,
        metavar="N",
        help="limit the prompt set (debug/planning aid)",
    )
    parser.add_argument(
        "--authorise-full-run",
        action="store_true",
        help="required for a full run; prevents an accidental 20,000-sample run",
    )
    return parser


def _print_plan(report: dict[str, Any]) -> None:
    print(f"mode={report['mode']} run_dir={report['run_dir']}")
    print(
        f"prompts={report['n_prompts']} generators={report['generators']} "
        f"decodings={report['decodings']}"
    )
    print("planned streams:")
    for entry in report["streams"]:
        print(f"  {entry['kind']:<8} {entry['stream_id']:<44} {entry['path']}")
    print(
        f"expected_sample_ids={report['sample_id_count']} "
        f"unique={report['unique_sample_ids']} "
        f"stream_files={report['stream_count']}"
    )
    if report.get("mode") == "smoke":
        print(f"smoke_prompt_ids={report.get('smoke', {}).get('prompt_ids', [])}")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)

    try:
        config = load_config(Path(args.config))
    except FileNotFoundError as exc:
        print(f"config not found: {exc}", file=sys.stderr)
        return 2
    except (json.JSONDecodeError, ValidationError, ValueError) as exc:
        print(f"invalid config: {exc}", file=sys.stderr)
        return 2

    prompts_path = Path(args.prompts) if args.prompts else REPO_ROOT / config.prompt_manifest
    splits_path = Path(args.splits) if args.splits else REPO_ROOT / config.split_manifest
    out_dir = Path(args.out) if args.out else None

    if (
        not args.smoke
        and not args.dry_run
        and not args.plan_only
        and not args.authorise_full_run
    ):
        print(
            "refusing a full run without --authorise-full-run: this would plan/"
            "write 20,000 samples. The flag exists so the notebook cannot start "
            "a corpus by accident.",
            file=sys.stderr,
        )
        return 2

    registry: dict[str, Any] | None = None
    registry_path = REPO_ROOT / config.licence_registry
    if registry_path.exists():
        try:
            registry = load_registry(registry_path)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"invalid licence registry: {exc}", file=sys.stderr)
            return 2

    try:
        report = run_generation(
            config,
            prompts_path=prompts_path,
            splits_path=splits_path,
            out_dir=out_dir,
            generator_ids=args.generator,
            decoding_ids=args.decoding,
            smoke=args.smoke,
            dry_run=args.dry_run,
            plan_only=args.plan_only,
            retry_failed=args.retry_failed,
            max_prompts=args.max_prompts,
            authorise_full_run=args.authorise_full_run,
            report_path=Path(args.report) if args.report else None,
            registry=registry,
        )
    except SmokeOutputError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    except AuthorisationError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    except LicenceValidationError as exc:
        for problem in exc.problems:
            print(f"refused: {problem}", file=sys.stderr)
        return 2
    except (ResumeMismatch, ResumeCorruptionError) as exc:
        print(f"resume refused: {exc}", file=sys.stderr)
        return 3
    except (GenerationError, FileNotFoundError, json.JSONDecodeError, ValueError) as exc:
        print(f"failed: {exc}", file=sys.stderr)
        return 2

    if args.dry_run or args.plan_only:
        _print_plan(report)
        return 0

    print(
        f"mode={report['mode']} run_dir={report['run_dir']} "
        f"n_records={report['n_records']} n_failures={report['n_failures']} "
        f"attempts={report['attempts']}"
    )
    if report.get("mode") == "smoke":
        print(f"smoke_prompt_ids={report['smoke']['prompt_ids']}")
    if args.report:
        print(f"report={args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
