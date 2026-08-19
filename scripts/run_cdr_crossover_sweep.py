"""Driver script to run the Angle 2 CDR crossover sweep and plot the heatmap."""

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qemsel import mitigation
from qemsel.experiment import run_experiment


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/cdr_regressor.yaml"))
    parser.add_argument("--out", type=Path, default=Path("results/cdr_crossover"))
    parser.add_argument("--smoke", action="store_true", help="Run a smaller grid for testing")
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = args.config
    out_dir = args.out

    if not config_path.exists():
        print(f"error: config file not found: {config_path}")
        return 1

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    # Define the sweep grid
    if args.smoke:
        n_values = [10, 25]
        f_values = [0.1, 0.5]
    else:
        n_values = [10, 25, 50, 100]
        f_values = [0.1, 0.3, 0.5, 0.7, 0.9]

    out_dir.mkdir(parents=True, exist_ok=True)

    results_grid = np.zeros((len(f_values), len(n_values)))

    print(f"Running CDR crossover sweep on {len(n_values)}x{len(f_values)} grid...")

    orig_n = mitigation.CDR_SKLEARN_NUM_TRAINING_CIRCUITS
    orig_f = mitigation.CDR_FRACTION_NON_CLIFFORD
    orig_mult_ridge = mitigation.SHOT_MULTIPLIER_V2.get("cdr_ridge")
    orig_mult_rf = mitigation.SHOT_MULTIPLIER_V2.get("cdr_rf")

    try:
        for i, f in enumerate(f_values):
            for j, n in enumerate(n_values):
                print(f"--- Running N={n}, fraction={f} ---")
                mitigation.CDR_SKLEARN_NUM_TRAINING_CIRCUITS = n
                mitigation.CDR_FRACTION_NON_CLIFFORD = f
                # Update derived constants just in case (e.g. SHOT_MULTIPLIER_V2)
                mitigation.SHOT_MULTIPLIER_V2["cdr_ridge"] = 1 + n
                mitigation.SHOT_MULTIPLIER_V2["cdr_rf"] = 1 + n

                run_out = out_dir / f"N_{n}_f_{f}"
                df = run_experiment(config, run_out)

                # Compute difference: linear (cdr_ridge) - nonlinear (cdr_rf)
                # Positive means nonlinear has smaller error (nonlinear wins)
                # Negative means linear has smaller error (linear wins)
                diff = df["cdr_ridge_abs_error"].mean() - df["cdr_rf_abs_error"].mean()
                results_grid[i, j] = diff
    finally:
        mitigation.CDR_SKLEARN_NUM_TRAINING_CIRCUITS = orig_n
        mitigation.CDR_FRACTION_NON_CLIFFORD = orig_f
        if orig_mult_ridge is not None:
            mitigation.SHOT_MULTIPLIER_V2["cdr_ridge"] = orig_mult_ridge
        if orig_mult_rf is not None:
            mitigation.SHOT_MULTIPLIER_V2["cdr_rf"] = orig_mult_rf

    # Plot heatmap
    plt.figure(figsize=(8, 6))
    # We want a diverging colormap centered at 0
    vmax = np.max(np.abs(results_grid))
    im = plt.imshow(results_grid, origin="lower", cmap="RdBu", aspect="auto", vmin=-vmax, vmax=vmax)
    plt.colorbar(im, label="Error Difference (Linear - Nonlinear)")

    plt.xticks(range(len(n_values)), n_values)
    plt.yticks(range(len(f_values)), f_values)
    plt.xlabel("CDR Training Set Size (N)")
    plt.ylabel("Non-Clifford Fraction")
    plt.title("CDR Crossover: Linear (Ridge) vs Nonlinear (RF) Regressor\n(Red/Positive = RF wins; Blue/Negative = Ridge wins)")

    # Annotate winners
    for i in range(len(f_values)):
        for j in range(len(n_values)):
            val = results_grid[i, j]
            winner = "RF" if val > 0 else "Ridge"
            plt.text(j, i, winner, ha="center", va="center",
                     color="black" if abs(val) < vmax*0.5 else "white")

    plot_path = out_dir / "cdr_crossover_heatmap.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"Saved heatmap to {plot_path}")

    return 0

if __name__ == "__main__":
    sys.exit(main())
