"""Script to perform cost-aware lambda sweep over benchmark results.

This utility takes a results CSV and sweeps the shot-cost penalty weight lambda
to show the selection shares of each QEM technique under different resource penalties.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep lambda and calculate QEM selection shares.")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("results/research/results.csv"),
        help="Path to the benchmark results CSV file"
    )
    parser.add_argument(
        "--base-shots",
        type=int,
        default=4096,
        help="Base shots used for scaling normalization"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_path = args.data
    base_shots = args.base_shots

    if not data_path.exists():
        print(f"Error: Data file not found at {data_path}", file=sys.stderr, flush=True)
        return 1

    print(f"Loading benchmark results from: {data_path}", flush=True)
    df = pd.read_csv(data_path)

    # Detect available techniques in CSV columns
    techniques = []
    for col in df.columns:
        if col.endswith("_abs_error"):
            tech_name = col[:-10]
            techniques.append(tech_name)

    if not techniques:
        print("Error: No technique columns found in the results CSV.", file=sys.stderr, flush=True)
        return 1

    print(f"Detected techniques: {techniques}", flush=True)
    print(f"Base shots: {base_shots}", flush=True)

    lambdas = [0.0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04]

    # Print markdown table header
    print("\n| Penalty (lambda) | RAW/RAW+ Share | ZNE Share | CDR Share | REM Share |", flush=True)
    print("| :--- | :---: | :---: | :---: | :---: |", flush=True)

    for l in lambdas:
        best_counts = {t: 0 for t in techniques}
        valid_rows = 0

        for _, row in df.iterrows():
            min_loss = float("inf")
            best_tech = None
            for t in techniques:
                err = row[f"{t}_abs_error"]
                shots = row[f"{t}_shots"]
                if pd.isna(err) or pd.isna(shots):
                    continue
                # Cost-aware formula: L_lambda = error + lambda * (shots / base_shots)
                loss = err + l * (shots / base_shots)
                if loss < min_loss:
                    min_loss = loss
                    best_tech = t

            if best_tech is not None:
                best_counts[best_tech] += 1
                valid_rows += 1

        if valid_rows == 0:
            continue

        # Group counts for presentation
        # raw group includes both raw and raw_plus
        raw_count = best_counts.get("raw", 0) + best_counts.get("raw_plus", 0)
        zne_count = best_counts.get("zne", 0) + best_counts.get("zne_fr", 0)
        cdr_count = best_counts.get("cdr", 0) + best_counts.get("cdr_ridge", 0) + best_counts.get("cdr_rf", 0)
        rem_count = best_counts.get("rem", 0)

        raw_share = (raw_count / valid_rows) * 100
        zne_share = (zne_count / valid_rows) * 100
        cdr_share = (cdr_count / valid_rows) * 100
        rem_share = (rem_count / valid_rows) * 100

        print(
            f"| **{l:.4f}** | {raw_share:.1f}% | {zne_share:.1f}% | {cdr_share:.1f}% | {rem_share:.1f}% |",
            flush=True
        )

    print("\nSweep completed successfully.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
