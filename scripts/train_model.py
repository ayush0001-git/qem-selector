"""CLI: train the QEM-technique recommender from an experiment results CSV.

Usage:
    python scripts/train_model.py --data results/run1/results.csv --out results/run1

``--results`` is accepted as an alias for ``--data``. Writes
``model.joblib`` + ``metrics.json`` into --out and prints a summary.

RECOMMENDED for headline numbers: pass the SEED-AVERAGED
``aggregated.csv`` (written next to results.csv by run_experiment) instead
of the per-seed ``results.csv``. Since the fixer pass 2026-07-21 the
aggregated file carries seed-mean ``feat_*`` columns and
maximum-coverage winner labels, so it trains directly; per-seed winner
labels disagree with the seed-averaged winner on a meaningful fraction of
rows (measured 28.9% on the research smoke), which is pure label noise.

``--label both`` trains BOTH winner labels in one call via
``qemsel.model.train_and_eval_all``: the accuracy-at-any-cost model
(``model.joblib`` + ``metrics.json``) and, when the data has a usable
``best_technique_cost_aware`` column, the equal-shot-budget model
(``model_cost_aware.joblib`` + ``metrics_cost_aware.json``), embedding the
cost-aware metrics into ``metrics.json`` under the ``cost_aware`` key so
``make_report.py`` renders both side by side with no extra flags.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from qemsel.model import (
    SIGNIFICANT_LABEL,
    derive_significant_label,
    train_and_eval,
    train_and_eval_all,
)


def _print_summary(metrics: dict, out: Path, bundle_name: str) -> None:
    """Print the human-readable training summary for one metrics dict."""
    print(f"\n=== qemsel model training summary ({bundle_name}) ===")
    print(f"training label    : {metrics.get('label_column', '?')}")
    print(f"samples used      : {metrics['n_samples']}")
    print(f"cv folds          : {metrics['cv_folds']}"
          + ("  (0 = too little data; trained/evaluated on all rows)"
             if metrics["cv_folds"] == 0 else
             f"  (grouped by circuit config: {metrics.get('cv_grouping')})"))
    dropped = metrics.get("dropped_classes")
    if dropped:
        print(f"dropped from CV   : {dropped} (singleton classes; still "
              "trained into the final bundle)")
    print(f"best model        : {metrics['best_model_name']}")
    print(f"accuracy          : {metrics['accuracy']:.3f}")
    print(f"macro F1          : {metrics['macro_f1']:.3f}")
    print(f"baseline accuracy : {metrics['baseline_accuracy']:.3f}")
    lofo = metrics.get("lofo")
    if isinstance(lofo, dict):
        print(
            f"leave-one-family-out accuracy : {lofo['accuracy']:.3f} "
            f"(macro F1 {lofo['macro_f1']:.3f}, {lofo['n_families']} families)"
            " <- honest 'new circuit' generalization"
        )
    lobo = metrics.get("lobo")
    if isinstance(lobo, dict):
        print(
            f"leave-one-backend-out accuracy: {lobo['accuracy']:.3f} "
            f"(macro F1 {lobo['macro_f1']:.3f}, {lobo['n_backends']} backends)"
            " <- noise-level INTERPOLATION (scale-siblings of a held-out"
            " backend may remain in training)"
        )
    lodo = metrics.get("lodo")
    if isinstance(lodo, dict):
        print(
            f"leave-one-device-out accuracy : {lodo['accuracy']:.3f} "
            f"(macro F1 {lodo['macro_f1']:.3f}, {lodo['n_devices']} devices)"
            " <- honest 'new noise environment' generalization"
            " (all scales of a device held out together)"
        )
    print("per-model (CV accuracy mean +/- std | macro F1):")
    for name, m in metrics["per_model"].items():
        print(
            f"  {name:<18}: {m['accuracy']:.3f} +/- {m['accuracy_std']:.3f}"
            f" | {m['macro_f1']:.3f}"
        )
    print(f"labels            : {metrics['labels']}")
    print(f"artifacts         : {out / bundle_name}")


def main(argv: list[str] | None = None) -> int:
    """Parse args, train, print a human-readable metrics summary."""
    parser = argparse.ArgumentParser(
        description="Train the QEM recommender model from results.csv"
    )
    parser.add_argument(
        "--data",
        "--results",
        dest="data",
        type=Path,
        required=True,
        help="path to results.csv produced by run_experiment",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="output directory for model.joblib + metrics.json",
    )
    parser.add_argument(
        "--label",
        choices=[
            "best_technique",
            "best_technique_cost_aware",
            "both",
            "significant",
        ],
        default="best_technique",
        help="winner column to train on: 'best_technique' (accuracy at any "
        "shot cost, default), 'best_technique_cost_aware' (winner at an "
        "equal shot budget; makes 'raw' a reachable class), 'both' "
        "(train both models; the cost-aware one goes to "
        "model_cost_aware.joblib + metrics_cost_aware.json and is embedded "
        "into metrics.json for the side-by-side report section), or "
        "'significant' (V2: derive a significance-aware 'tie'-class label "
        "via derive_significant_label and train it to model_significant.joblib "
        "+ metrics_significant.json)",
    )
    # ---- V2 flags (all default-off; defaults reproduce the V1 CLI) --------
    parser.add_argument(
        "--k-sigma",
        type=float,
        default=2.0,
        help="V2: significance margin in combined shot-noise sigma units for "
        "--label significant (default 2.0)",
    )
    parser.add_argument(
        "--feature-version",
        type=int,
        choices=[1, 2],
        default=1,
        help="V2: feature-set version (1 = frozen 10-feature vector, default; "
        "2 = 15-feature shots-aware vector — the df must carry those feat_* "
        "columns)",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="V2: sigmoid-calibrate the refit model (CalibratedClassifierCV, "
        "grouped folds) so recommend's probabilities can be thresholded",
    )
    parser.add_argument(
        "--abstain-threshold",
        type=float,
        default=None,
        help="V2: store an abstain threshold in (0, 1) in the bundle; "
        "recommend returns 'abstain' when the top probability falls below it",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="V2: emit extended fold statistics (per-fold accuracies + "
        "fold_summary) into the metrics JSON",
    )
    args = parser.parse_args(argv)

    v2_kwargs = dict(
        feature_version=args.feature_version,
        calibrate=args.calibrate,
        abstain_threshold=args.abstain_threshold,
        extended_stats=args.stats,
    )

    data_path = Path(args.data)
    if not data_path.exists():
        parser.error(f"data file not found: {data_path}")
    df = pd.read_csv(data_path)
    out = Path(args.out)

    # Seed-averaged labels are the lower-noise choice — nudge, don't force.
    sibling_agg = data_path.parent / "aggregated.csv"
    if data_path.name == "results.csv" and sibling_agg.exists():
        print(
            "[train_model] NOTE: training on PER-SEED labels "
            f"({data_path.name}). A seed-averaged {sibling_agg} exists — "
            "pass it as --data for the lower-label-noise headline numbers."
        )

    if args.label == "both":
        results = train_and_eval_all(df, out, **v2_kwargs)
        metrics = results["best_technique"]
        _print_summary(metrics, out, "model.joblib")
        cost_metrics = results["best_technique_cost_aware"]
        if cost_metrics is not None:
            _print_summary(cost_metrics, out, "model_cost_aware.joblib")
            json.dumps(cost_metrics)
    elif args.label == "significant":
        # V2: derive the significance-aware label column (never written to
        # the CSV), then train it to its own bundle/metrics filenames.
        df[SIGNIFICANT_LABEL] = derive_significant_label(df, k_sigma=args.k_sigma)
        metrics = train_and_eval(
            df,
            out,
            label_column=SIGNIFICANT_LABEL,
            bundle_filename="model_significant.joblib",
            metrics_filename="metrics_significant.json",
            **v2_kwargs,
        )
        _print_summary(metrics, out, "model_significant.joblib")
    else:
        metrics = train_and_eval(df, out, label_column=args.label, **v2_kwargs)
        _print_summary(metrics, out, "model.joblib")

    # Round-trip check: metrics must be JSON-serializable (contract).
    json.dumps(metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
