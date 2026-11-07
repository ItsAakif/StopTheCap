"""CLI: python -m ml.cli baseline [--config word|char]"""
from __future__ import annotations
import argparse, sys


def main(argv=None) -> int:
    from .tfidf import CONFIGS, run_baseline, write_reports

    p = argparse.ArgumentParser(prog="ml.cli.baseline")
    p.add_argument("--config", default="all", choices=["all", *sorted(CONFIGS)])
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)

    configs = sorted(CONFIGS) if args.config == "all" else [args.config]
    results = [run_baseline(c) for c in configs]
    md, js = write_reports(results)

    for r in results:
        print(f"\n=== config={r.config}  threshold={r.threshold:.4f}  vocab={r.vocab_size:,}")
        for name, m in {**r.metrics, **r.per_source_quarantine}.items():
            if m.roc_auc is None:
                print(f"  {name:24} n={m.n:<6} (undefined: {m.note})")
                continue
            fpr = "n/a" if m.fpr is None else f"{m.fpr:.3f}"
            flag = "" if m.threshold_metrics_usable else "  [insufficient negatives]"
            print(f"  {name:24} n={m.n:<6} neg={m.n_neg:<5} AUC={m.roc_auc:.4f} "
                  f"FPR={fpr}{flag}")
    print(f"\nreport: {md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
