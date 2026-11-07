"""Command line entry point.

    python -m ml.cli build     # manifests -> canonical records -> splits -> COMBINED.yaml
    python -m ml.cli audit     # run all audit gates, write reports/audit.{md,json}
    python -m ml.cli probe     # shortcut probes only (fast feedback loop)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .audit.audit import run_audit, write_reports
from .ingest.pipeline import BuildConfig, audit_build, build, REPORTS


def _cfg(args) -> BuildConfig:
    return BuildConfig(
        salt=args.salt,
        raid_n=args.raid_n,
        raid_seed=args.raid_seed,
        generator_holdout=args.generator_holdout or None,
        domain_holdout=args.domain_holdout or None,
        write=True,
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="ml.cli", description="StopTheCap dataset pipeline")
    sub = p.add_subparsers(dest="cmd", required=True)

    for name, help_text in (
        ("build", "build canonical records and split manifest"),
        ("audit", "run the full audit (build + gates + reports)"),
        ("probe", "run shortcut probes only"),
        ("baseline", "run the TF-IDF + logistic-regression baseline"),
        ("diagnose", "explain cross-dataset failure without tuning on eval labels"),
    ):
        sp = sub.add_parser(name, help=help_text)
        sp.add_argument("--salt", default="stopthecap-2026-10-29")
        sp.add_argument("--raid-n", type=int, default=2000)
        sp.add_argument("--raid-seed", type=int, default=20261029)
        sp.add_argument("--generator-holdout", default="",
                        help="generator family to hold out (usually empty: RAID provides OOD)")
        sp.add_argument("--domain-holdout", default="news_reuters")
        sp.add_argument("--config", default="all",
                        help="baseline only: 'all', 'word' or 'char'")

    args = p.parse_args(argv)

    if args.cmd == "build":
        res = build(_cfg(args))
        print(f"records: {len(res.records)}  datasets: {res.per_dataset}")
        print(f"duplicate pairs: {len(res.duplicates)}")
        print("wrote datasets/canonical/*.jsonl, datasets/splits/assignments.jsonl, "
              "datasets/manifests/COMBINED.yaml")
        return 0

    if args.cmd == "audit":
        res, audit = audit_build(_cfg(args))
        md, js = write_reports(audit, REPORTS)
        print(f"records: {len(res.records)}")
        for name, ok in audit.gates.items():
            print(f"  {'PASS' if ok else 'FAIL'}  {name}")
        print(f"\noverall: {'PASS' if audit.ok else 'FAIL'}")
        print(f"report: {md}")
        return 0 if audit.ok else 1

    if args.cmd == "probe":
        res = build(_cfg(args))
        audit = run_audit(res.records, duplicates=res.duplicates,
                          salt=args.salt)
        probes = audit.data.get("probes", {})
        print(json.dumps(probes, indent=2))
        return 0

    if args.cmd == "baseline":
        from .baselines.cli import main as baseline_main
        return baseline_main(["--config", args.config])

    if args.cmd == "diagnose":
        from .baselines.diagnose import diagnose, write_report
        d = diagnose()
        md, js = write_report(d)
        print(f"threshold={d.threshold:.4f}")
        for r in d.results:
            auc = "n/a" if r.roc_auc is None else f"{r.roc_auc:.3f}"
            print(f"  {r.name:44} n={r.n:<6} AUC={auc}  OOV={r.oov_rate:.1%}")
        for r in d.raid_by_domain:
            auc = "n/a" if r.roc_auc is None else f"{r.roc_auc:.3f}"
            print(f"  {r.name:44} n={r.n:<6} AUC={auc}  OOV={r.oov_rate:.1%}")
        for n in d.notes:
            print(f"  note: {n}")
        print(f"\nreport: {md}")
        return 0

    return 2


if __name__ == "__main__":
    sys.exit(main())
