"""CLI for the Wikipedia prompt build.

The default ``titles`` mode enumerates page titles cheaply and then fetches
snapshots in 20-title batches.  ``--mode generator`` keeps the original
``generator=allpages`` path for compatibility.

This path writes the same manifest schema as :mod:`ml.e2c1.build_prompts`, but it
derives each paragraph from the same response that carries the revision id.  It
replaces the earlier evidence-replay path, which only verified ~3.5% of rows.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from .api_prompts import (
    DEFAULT_AUDIT,
    DEFAULT_DEADLINE_SECONDS,
    DEFAULT_MAX_REQUESTS,
    DEFAULT_MIN_INTERVAL_S,
    DEFAULT_MODE,
    DEFAULT_OUT,
    DEFAULT_PER_PAGE,
    DEFAULT_REQUEST_TIMEOUT_S,
    DEFAULT_WORKERS,
    run_api_build,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build the E2-C1 prompt manifest from live Wikipedia responses "
            "(titles mode by default; generator mode for compatibility)."
        )
    )
    parser.add_argument("--out", default=DEFAULT_OUT, help="prompt manifest output")
    parser.add_argument("--audit", default=DEFAULT_AUDIT, help="audit JSON output")
    parser.add_argument(
        "--n", type=int, default=5000, help="number of prompts to select"
    )
    parser.add_argument(
        "--mode",
        choices=("titles", "generator"),
        default=DEFAULT_MODE,
        help="fetch strategy: titles (default) or legacy generator=allpages",
    )
    parser.add_argument(
        "--per-page",
        type=int,
        default=DEFAULT_PER_PAGE,
        help="maximum paragraphs derived from one page",
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

    audit, exit_code = run_api_build(
        out_path=Path(args.out),
        audit_path=Path(args.audit),
        n=args.n,
        per_page=args.per_page,
        mode=args.mode,
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
