"""CLI: estimate the real-hardware QPU cost of an experiment config.

Usage (from the project root, venv python)::

    python scripts/estimate_hardware_cost.py --config configs/hw_first_run.yaml

Pure local arithmetic — needs NO credentials, makes NO network calls,
submits NOTHING. Prints per-technique executor-call counts, total jobs, a
conservative QPU-seconds estimate, and whether it fits the free Open Plan's
10 QPU-minutes/month. The assumptions are documented in
``qemsel.hardware`` (module docstring) and echoed in the output.

Exit codes: 0 = estimate fits the free monthly budget (or config has no
ibm_* backends), 2 = config error, 3 = estimate does NOT fit.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from qemsel import hardware


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Estimate real-hardware QPU cost of a qemsel experiment config "
            "(local arithmetic only; nothing is submitted)."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="YAML experiment config to estimate",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Print the cost breakdown; exit 0 fits / 3 does not fit / 2 error."""
    args = parse_args(argv)
    if not args.config.exists():
        print(f"error: config file not found: {args.config}", file=sys.stderr)
        return 2
    try:
        config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        from qemsel import hardware
        from qemsel import mitigation

        original_mult = mitigation.SHOT_MULTIPLIER
        mitigation.SHOT_MULTIPLIER = mitigation.SHOT_MULTIPLIER_V2
        est_config = dict(config)
        if isinstance(est_config.get("shots"), list):
            est_config["shots"] = max(est_config["shots"])
        try:
            estimate = hardware.estimate_config_qpu_seconds(est_config)
        finally:
            mitigation.SHOT_MULTIPLIER = original_mult
    except Exception as exc:  # noqa: BLE001 - CLI boundary: clean one-liner
        print(f"error: could not estimate config: {exc}", file=sys.stderr)
        return 2

    print(f"Hardware cost estimate for {args.config}")
    print("=" * 60)
    if estimate["n_ibm_backends"] == 0:
        print("No ibm_* backends in this config -- simulated backends are")
        print("free; nothing to estimate. (0 QPU seconds)")
        return 0

    print(f"circuits in suite:        {estimate['n_circuits']}")
    print(
        f"real (ibm_*) backends:    {estimate['n_ibm_backends']} "
        f"{estimate['ibm_backends']}"
    )
    print(f"(circuit, backend) units: {estimate['n_units']}")
    print(f"shots per job:            {estimate['shots']}")
    print()
    print("executor calls (= single-circuit jobs) per unit, by technique:")
    for tech, count in estimate["per_technique_jobs_per_unit"].items():
        print(f"  {tech:<4} {count:>3} job(s)")
    print(f"  total per unit: {estimate['jobs_per_unit']}")
    print()
    print(f"TOTAL JOBS:               {estimate['total_jobs']}")
    print(f"est. QPU-seconds per job: {estimate['est_seconds_per_job']:.2f}")
    total = estimate["est_total_qpu_seconds"]
    print(
        f"EST. TOTAL QPU TIME:      ~{total:.0f} s  (~{total / 60.0:.1f} min)"
    )
    budget = estimate["free_plan_monthly_seconds"]
    print(
        f"free Open Plan budget:    {budget:.0f} s/month "
        f"({budget / 60.0:.0f} min)"
    )
    print()
    print(f"assumptions: {estimate['assumptions']}")
    print()
    if estimate["fits_free_plan"]:
        print(
            f"VERDICT: fits -- ~{100.0 * total / budget:.0f}% of the monthly "
            "free budget."
        )
        if config.get("hardware_confirmed") is not True:
            print(
                "NOTE: the config still has hardware_confirmed: false -- "
                "flip it to true (your cost consent) before running."
            )
        return 0
    print(
        f"VERDICT: DOES NOT FIT -- ~{total:.0f}s exceeds the {budget:.0f}s "
        "monthly free budget. Shrink the suite (fewer circuits/seeds), drop "
        "expensive techniques (cdr = 11 jobs/unit), or lower shots."
    )
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
