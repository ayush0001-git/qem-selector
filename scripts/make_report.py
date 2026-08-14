"""CLI: generate the final report (report.md + figures) from experiment outputs.

Usage (from the project root, with the project venv):

    python scripts/make_report.py --data results/run1/results.csv \
        --metrics results/run1/metrics.json --out results/run1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main(argv: list[str] | None = None) -> int:
    """Parse args, load inputs, write report.md + PNGs, print the path."""
    parser = argparse.ArgumentParser(
        description="Generate report.md + PNG figures from results.csv and "
        "metrics.json."
    )
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="path to results.csv written by scripts/run_experiment.py",
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        required=True,
        help="path to metrics.json written by scripts/train_model.py",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="output directory for report.md and PNGs (created if missing)",
    )
    parser.add_argument(
        "--stats-json",
        type=Path,
        default=None,
        help="optional stats.json written by scripts/compute_stats.py; when "
        "given, adds the section 8 'Statistical hygiene' block",
    )
    parser.add_argument(
        "--boundary-json",
        type=Path,
        default=None,
        help="optional boundary-overlay JSON (the dict returned by "
        "qemsel.boundary.overlay_selector_vs_theory); when given, adds the "
        "section 9 boundary-overlay block. Its 'plot_path' figure must sit "
        "inside --out",
    )
    args = parser.parse_args(argv)

    if not args.data.exists():
        raise SystemExit(f"error: results file not found: {args.data}")
    if not args.metrics.exists():
        raise SystemExit(f"error: metrics file not found: {args.metrics}")

    df = pd.read_csv(args.data)
    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))

    stats_results = None
    if args.stats_json is not None:
        if not args.stats_json.exists():
            raise SystemExit(f"error: stats file not found: {args.stats_json}")
        stats_results = json.loads(args.stats_json.read_text(encoding="utf-8"))

    boundary_overlay = None
    if args.boundary_json is not None:
        if not args.boundary_json.exists():
            raise SystemExit(
                f"error: boundary file not found: {args.boundary_json}"
            )
        boundary_overlay = json.loads(
            args.boundary_json.read_text(encoding="utf-8")
        )

    # The experiment config (run_meta.json sidecar next to the data file)
    # lets the report disclose circuit-selection conditioning
    # (min_abs_ideal rejection sampling) — best-effort, absence is fine.
    run_config = None
    run_meta_path = args.data.parent / "run_meta.json"
    if run_meta_path.exists():
        try:
            meta = json.loads(run_meta_path.read_text(encoding="utf-8"))
            if isinstance(meta, dict) and isinstance(meta.get("config"), dict):
                run_config = meta["config"]
        except (OSError, json.JSONDecodeError) as exc:
            print(f"warning: could not read {run_meta_path}: {exc}")

    from qemsel.report import generate_report

    report_path = generate_report(
        df,
        metrics,
        args.out,
        run_config=run_config,
        stats_results=stats_results,
        boundary_overlay=boundary_overlay,
    )
    print(f"report written: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
