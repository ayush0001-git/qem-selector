"""CLI entry point: run a QEM benchmark sweep from a YAML config.

Usage (from the project root, venv python)::

    python scripts/run_experiment.py --config configs/tiny.yaml --out results/tiny

Re-running the same command RESUMES a crashed/interrupted run: completed
(circuit, backend) units already present in ``<out>/results.csv`` are skipped.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from qemsel.experiment import run_experiment


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Run the qemsel QEM benchmark sweep from a YAML config."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs") / "tiny.yaml",
        help="YAML experiment config (default: configs/tiny.yaml)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output directory (default: results/<config stem>)",
    )
    parser.add_argument(
        "--force-hardware",
        action="store_true",
        help=(
            "bypass ONLY the free-plan budget-fit refusal for ibm_* configs "
            "(hardware_confirmed: true and credentials are still required)"
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Load the config, run the sweep, print a short summary."""
    args = parse_args(argv)
    config_path: Path = args.config
    if not config_path.exists():
        print(f"error: config file not found: {config_path}", file=sys.stderr)
        return 2
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    out_dir: Path = args.out if args.out is not None else Path("results") / config_path.stem

    # Real-hardware budget gate: refuse ibm_* configs whose conservative
    # cost estimate does not fit the free Open Plan's 10 QPU-min/month,
    # unless --force-hardware. (Credentials + hardware_confirmed: true are
    # additionally enforced inside run_experiment's config validation.)
    ibm_backends = [
        b for b in (config.get("backends") or []) if str(b).startswith("ibm_")
    ]
    if ibm_backends:
        from qemsel import hardware

        from qemsel import mitigation
        
        if args.force_hardware:
            print("real hardware requested: bypassing free-plan cost estimation (--force-hardware).")
        else:
            original_mult = mitigation.SHOT_MULTIPLIER
            mitigation.SHOT_MULTIPLIER = mitigation.SHOT_MULTIPLIER_V2
            try:
                estimate = hardware.estimate_config_qpu_seconds(config)
            finally:
                mitigation.SHOT_MULTIPLIER = original_mult
            total = estimate["est_total_qpu_seconds"]
            budget = estimate["free_plan_monthly_seconds"]
            print(
                f"real hardware requested ({ibm_backends}): "
                f"{estimate['total_jobs']} jobs, est. ~{total:.0f} QPU-seconds "
                f"(~{total / 60.0:.1f} min) of the free {budget / 60.0:.0f} "
                "min/month"
            )
            if not estimate["fits_free_plan"]:
                print(
                    "error: estimated cost exceeds the free monthly budget -- "
                    "refusing to run. Inspect with scripts/estimate_hardware_cost"
                    f".py --config {config_path}, shrink the config, or re-run "
                    "with --force-hardware if you truly intend this.",
                    file=sys.stderr,
                )
                return 3

    try:
        df = run_experiment(config, out_dir)
    except ValueError as exc:
        # Config errors (including the real-hardware credential/confirmation
        # gates) get a clean one-line error instead of a traceback.
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print()
    print(f"results: {len(df)} rows -> {out_dir / 'results.csv'}")
    if len(df) > 0:
        print("best_technique distribution:")
        counts = df["best_technique"].replace("", "<all failed>").value_counts()
        for tech, count in counts.items():
            print(f"  {tech}: {count}")
        print("cost-aware winner distribution:")
        counts = df["best_technique_cost_aware"].replace(
            "", "<all failed>"
        ).value_counts()
        for tech, count in counts.items():
            print(f"  {tech}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
