"""CLI for the wikitext-revision prompt build.

This is the fast path: ``prop=revisions&rvprop=content`` returns up to 50
articles per request, with the revision id and categories in the same response.
The manifest and audit schemas match :mod:`ml.e2c1.build_prompts_api`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from .wikitext_prompts import (
    DEFAULT_AUDIT,
    DEFAULT_DEADLINE_SECONDS,
    DEFAULT_MAX_PER_PAGE,
    DEFAULT_MAX_REQUESTS,
    DEFAULT_MIN_INTERVAL_S,
    DEFAULT_OUT,
    DEFAULT_PER_PAGE,
    DEFAULT_REQUEST_TIMEOUT_S,
    DEFAULT_SECTIONS,
    DEFAULT_WORKERS,
    run_wikitext_build,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build the E2-C1 prompt manifest from wikitext revisions "
            "(50 articles per request)."
        )
    )
    parser.add_argument("--out", default=DEFAULT_OUT, help="prompt manifest output")
    parser.add_argument("--audit", default=DEFAULT_AUDIT, help="audit JSON output")
    parser.add_argument(
        "--n", type=int, default=5000, help="number of prompts to select"
    )
    parser.add_argument(
        "--per-page",
        type=int,
        default=DEFAULT_PER_PAGE,
        help="maximum paragraphs accepted from one page",
    )
    parser.add_argument(
        "--max-per-page",
        type=int,
        default=DEFAULT_MAX_PER_PAGE,
        help="global manifest cap on prompts drawn from one page",
    )
    parser.add_argument(
        "--sections",
        choices=("lead", "all"),
        default=DEFAULT_SECTIONS,
        help="parse only the lead section or the whole article",
    )
    parser.add_argument(
        "--workers", type=int, default=DEFAULT_WORKERS, help="concurrent workers"
    )
    parser.add_argument(
        "--min-interval",
        type=float,
        default=DEFAULT_MIN_INTERVAL_S,
        help="minimum seconds between requests per worker",
    )
    parser.add_argument(
        "--max-requests",
        type=int,
        default=DEFAULT_MAX_REQUESTS,
        help="stop after this many network requests",
    )
    parser.add_argument(
        "--deadline-seconds",
        type=float,
        default=DEFAULT_DEADLINE_SECONDS,
        help="total wall-clock budget for the build",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=DEFAULT_REQUEST_TIMEOUT_S,
        help="per-request socket timeout in seconds",
    )
    parser.add_argument(
        "--cache-dir", default=None, help="raw response cache directory"
    )
    parser.add_argument(
        "--refresh", action="store_true", help="ignore cached API responses"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)

    audit, exit_code = run_wikitext_build(
        out_path=Path(args.out),
        audit_path=Path(args.audit),
        n=args.n,
        per_page=args.per_page,
        max_per_page=args.max_per_page,
        sections=args.sections,
        workers=args.workers,
        min_interval_s=args.min_interval,
        max_requests=args.max_requests,
        deadline_seconds=args.deadline_seconds,
        refresh=args.refresh,
        cache_dir=args.cache_dir,
        request_timeout_s=args.request_timeout,
    )

    print(
        f"status={audit['status']} selected={audit['selection']['selected']} "
        f"requests={audit['requests']} rate429={audit['rate429']} "
        f"elapsed={audit['elapsed_s']}s"
    )
    if audit["status"] != "ok":
        print(
            f"audit: {args.audit} (manifest written: {audit['output']['count']} rows)",
            file=sys.stderr,
        )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
