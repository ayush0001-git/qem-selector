"""Unit tests for qemsel.model.train_and_eval and scripts/train_model.py.

Runs standalone (no quantum simulation): synthetic DataFrames in the exact
experiment schema plus the architect-provided ``tiny_results_df`` fixture.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from qemsel.features import FEATURE_NAMES
from qemsel.model import train_and_eval, train_and_eval_all

FEAT_COLS = ["feat_" + n for n in FEATURE_NAMES]
TECHNIQUES = ["raw", "zne", "cdr", "rem"]
EXPECTED_KEYS = {
    "best_model_name",
    "accuracy",
    "macro_f1",
    "baseline_accuracy",
    "labels",
    "confusion_matrix",
    "feature_importances",
    "feature_importances_note",
    "per_model",
    "n_samples",
    "cv_n_samples",
    "cv_folds",
    "cv_grouping",
    "dropped_classes",
    "label_column",
    "lofo",
    "lobo",
    "lodo",
}


def _make_df(labels: list[str], seed: int = 0) -> pd.DataFrame:
    """Synthetic results-schema DataFrame, one row per label.

    All features are random EXCEPT feat_depth, which is written from the
    row index so callers can build learnable rules against it.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for i, best in enumerate(labels):
        n_qubits = 2 + i % 3
        # Learnable-rule hook: depth ~5 for even i, ~35 for odd i
        # (callers relying on this must construct labels accordingly).
        depth = (5 if i % 2 == 0 else 35) + int(rng.integers(0, 5))
        row = {
            "circuit_id": f"synth_q{n_qubits}_d{depth}_s{i}",
            "family": "layered_random",
            "n_qubits": n_qubits,
            "depth": depth,
            "seed": i,
            "backend": "FakeManilaV2",
            "pauli": "Z" * n_qubits,
            "ideal": float(rng.uniform(-1, 1)),
            "feat_n_qubits": float(n_qubits),
            "feat_depth": float(depth),
            "feat_n_1q_gates": float(rng.integers(1, 50)),
            "feat_n_2q_gates": float(rng.integers(1, 30)),
            "feat_n_cnot": float(rng.integers(0, 30)),
            "feat_n_non_clifford": float(rng.integers(0, 10)),
            "feat_clifford_fraction": float(rng.uniform(0.0, 1.0)),
            "feat_depth_per_qubit": float(depth / n_qubits),
            "feat_backend_avg_2q_error": float(rng.uniform(0.005, 0.03)),
            "feat_backend_avg_readout_error": float(rng.uniform(0.01, 0.2)),
        }
        for tech in TECHNIQUES:
            err = 0.01 if tech == best else 0.2
            row[f"{tech}_value"] = row["ideal"] + err
            row[f"{tech}_abs_error"] = err
            row[f"{tech}_shots"] = 1024
        row["best_technique"] = best
        rows.append(row)
    return pd.DataFrame(rows)


def _learnable_df(n_rows: int = 60, seed: int = 0) -> pd.DataFrame:
    """60 rows with rule: winner == 'zne' iff depth > 20, else 'raw'."""
    # even index -> depth ~5 -> raw; odd index -> depth ~35 -> zne
    labels = ["raw" if i % 2 == 0 else "zne" for i in range(n_rows)]
    return _make_df(labels, seed=seed)


def _random_label_df(n_rows: int = 60, seed: int = 1) -> pd.DataFrame:
    """60 rows, balanced 4-class labels shuffled independently of features."""
    rng = np.random.default_rng(seed)
    labels = [TECHNIQUES[i % 4] for i in range(n_rows)]
    labels = list(rng.permutation(labels))
    return _make_df(labels, seed=seed)


# --------------------------------------------------------------------------
# Learnability / sanity
# --------------------------------------------------------------------------


def test_learnable_rule_beats_baseline(out_dir: Path) -> None:
    df = _learnable_df()
    metrics = train_and_eval(df, out_dir)
    assert metrics["cv_folds"] == 5
    assert metrics["n_samples"] == 60
    # winner = zne iff depth > 20 is trivially learnable from feat_depth:
    # the model must clearly beat the majority-class baseline (~0.5).
    assert metrics["accuracy"] >= metrics["baseline_accuracy"] + 0.2
    assert metrics["accuracy"] >= 0.9
    assert metrics["macro_f1"] >= 0.9
    # depth should carry real permutation importance
    assert metrics["feature_importances"]["feat_depth"] > 0.0
    # healthy balanced data: nothing dropped from CV, all rows in CV
    assert metrics["dropped_classes"] == []
    assert metrics["cv_n_samples"] == metrics["n_samples"]
    # single family / single backend -> hold-out evaluations undefined
    assert metrics["lofo"] is None
    assert metrics["lobo"] is None


def test_random_labels_do_not_beat_baseline_by_much(out_dir: Path) -> None:
    df = _random_label_df()
    metrics = train_and_eval(df, out_dir)
    # Labels are independent of features: no model should look magically
    # better than the majority baseline (allow small-sample wiggle room).
    assert metrics["accuracy"] <= metrics["baseline_accuracy"] + 0.25
    for m in metrics["per_model"].values():
        assert m["accuracy"] <= metrics["baseline_accuracy"] + 0.3


def test_seed_duplicate_rows_do_not_inflate_accuracy(out_dir: Path) -> None:
    """Leakage regression (stats review 2026-07-21): rows of the same
    (family, n_qubits, depth) cell share BYTE-IDENTICAL feature vectors
    (features are angle-blind), and per-cell labels here carry ZERO feature
    signal. Row-level StratifiedKFold scored ~+0.2 above baseline on this
    structure by memorizing a test row's twin; grouped CV must not."""
    rng = np.random.default_rng(7)
    rows = []
    for cell in range(24):
        n_qubits = 2 + cell % 3
        depth = 4 * (1 + cell % 4)
        label = TECHNIQUES[int(rng.integers(0, 4))]  # independent of features
        for seed in range(3):  # 3 seed-duplicates, identical features
            row = {
                "circuit_id": f"layered_random_q{n_qubits}_d{depth}_s{seed}",
                "family": "layered_random",
                "n_qubits": n_qubits,
                "depth": depth,
                "seed": seed,
                "backend": "FakeManilaV2",
                "pauli": "Z" * n_qubits,
                "ideal": 0.5,
                "feat_n_qubits": float(n_qubits),
                "feat_depth": float(depth),
                "feat_n_1q_gates": float(n_qubits * depth),
                "feat_n_2q_gates": float((n_qubits - 1) * depth // 2),
                "feat_n_cnot": float((n_qubits - 1) * depth // 2),
                "feat_n_non_clifford": float(n_qubits + depth),
                "feat_clifford_fraction": 0.5,
                "feat_depth_per_qubit": float(depth / n_qubits),
                "feat_backend_avg_2q_error": 0.01,
                "feat_backend_avg_readout_error": 0.03,
            }
            for tech in TECHNIQUES:
                err = 0.01 if tech == label else 0.2
                row[f"{tech}_value"] = 0.5 + err
                row[f"{tech}_abs_error"] = err
                row[f"{tech}_shots"] = 1024
            row["best_technique"] = label
            rows.append(row)
    df = pd.DataFrame(rows)
    metrics = train_and_eval(df, out_dir)
    assert metrics["cv_folds"] >= 2  # grouped CV actually ran
    # No generalizable signal exists -> grouped CV must sit near baseline.
    assert metrics["accuracy"] <= metrics["baseline_accuracy"] + 0.15


# --------------------------------------------------------------------------
# Contract: return keys, artifacts, JSON round-trip
# --------------------------------------------------------------------------


def test_returned_keys_exact(out_dir: Path, tiny_results_df: pd.DataFrame) -> None:
    metrics = train_and_eval(tiny_results_df, out_dir)
    assert set(metrics.keys()) == EXPECTED_KEYS
    assert metrics["best_model_name"] in {"random_forest", "gradient_boosting"}
    # tiny_results_df: 16 rows, 4 per class -> min(5, 4) = 4 folds
    assert metrics["cv_folds"] == 4
    assert metrics["n_samples"] == 16
    assert metrics["labels"] == sorted(TECHNIQUES)
    n = len(metrics["labels"])
    assert len(metrics["confusion_matrix"]) == n
    assert all(len(r) == n for r in metrics["confusion_matrix"])
    assert sum(sum(r) for r in metrics["confusion_matrix"]) == 16
    assert set(metrics["feature_importances"].keys()) == set(FEAT_COLS)
    # both real models AND the dummy baseline reported, with mean AND std
    assert {"random_forest", "gradient_boosting"} <= set(metrics["per_model"])
    for m in metrics["per_model"].values():
        assert {"accuracy", "accuracy_std", "macro_f1"} <= set(m.keys())
    # balanced 4x4 data: nothing dropped, every row cross-validated
    assert metrics["dropped_classes"] == []
    assert metrics["cv_n_samples"] == 16
    # 4 families -> LOFO defined, with per-family accuracy AND macro-F1
    lofo = metrics["lofo"]
    assert isinstance(lofo, dict)
    assert lofo["n_families"] == 4
    assert len(lofo["per_family_accuracy"]) == 4
    assert set(lofo["per_family_macro_f1"]) == set(lofo["per_family_accuracy"])
    assert 0.0 <= lofo["accuracy"] <= 1.0
    # 2 backends -> LOBO defined (noise-level interpolation number)
    lobo = metrics["lobo"]
    assert isinstance(lobo, dict)
    assert lobo["n_backends"] == 2
    assert set(lobo["per_backend_accuracy"]) == {"FakeManilaV2", "FakeLagosV2"}
    assert set(lobo["per_backend_macro_f1"]) == set(lobo["per_backend_accuracy"])
    assert 0.0 <= lobo["accuracy"] <= 1.0
    # 2 plain backends = 2 base devices -> LODO defined too (and == LOBO
    # folds, since no '@x<scale>' siblings exist here)
    lodo = metrics["lodo"]
    assert isinstance(lodo, dict)
    assert lodo["n_devices"] == 2
    assert set(lodo["per_device_accuracy"]) == {"FakeManilaV2", "FakeLagosV2"}
    assert set(lodo["per_device_macro_f1"]) == set(lodo["per_device_accuracy"])
    assert lodo["accuracy"] == pytest.approx(lobo["accuracy"])
    assert 0.0 <= lodo["accuracy"] <= 1.0


def test_metrics_json_serializable(out_dir: Path, tiny_results_df: pd.DataFrame) -> None:
    metrics = train_and_eval(tiny_results_df, out_dir)
    dumped = json.dumps(metrics)  # raises TypeError on numpy leakage
    assert json.loads(dumped) == metrics


def test_artifacts_saved_and_loadable(out_dir: Path) -> None:
    df = _learnable_df()
    metrics = train_and_eval(df, out_dir)

    model_path = out_dir / "model.joblib"
    metrics_path = out_dir / "metrics.json"
    assert model_path.exists()
    assert metrics_path.exists()

    bundle = joblib.load(model_path)
    assert set(bundle.keys()) == {
        "model", "feature_names", "classes", "model_name", "label_column",
        "qemsel_version",
    }
    assert bundle["label_column"] == "best_technique"
    assert bundle["feature_names"] == FEAT_COLS
    assert bundle["model_name"] == metrics["best_model_name"]
    assert bundle["classes"] == metrics["labels"]
    import qemsel

    assert bundle["qemsel_version"] == qemsel.__version__

    # refit model must predict a known label from an in-order feature row.
    # Integrator note: predict with a NAMED DataFrame — the bundle's model is
    # fitted with feature names (recommend.py's consumption pattern), so
    # sklearn validates the column names instead of warning about their lack.
    X_one = df[FEAT_COLS].astype(float).iloc[:1]
    pred = bundle["model"].predict(X_one)
    assert pred[0] in metrics["labels"]

    saved = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert saved == metrics


# --------------------------------------------------------------------------
# Row dropping + degenerate data
# --------------------------------------------------------------------------


def test_drops_nan_and_empty_winners_and_nan_features(out_dir: Path) -> None:
    df = _learnable_df()
    df.loc[0, "best_technique"] = ""
    df.loc[1, "best_technique"] = np.nan
    df.loc[2, "feat_depth"] = np.nan
    metrics = train_and_eval(df, out_dir)
    assert metrics["n_samples"] == 57


def test_singleton_class_dropped_from_cv_not_collapsed(out_dir: Path) -> None:
    """Research-pass spec 2026-07-21 (SUPERSEDES the old cv_folds=0
    behaviour for this case): one singleton class must no longer disable
    the whole CV — it is DROPPED from the CV evaluation (recorded in
    dropped_classes) while the refit-on-all model still sees every row.
    Mirrors the actual small-run failure: rem 38 / cdr 35 / zne 1."""
    labels = ["rem"] * 10 + ["cdr"] * 10 + ["zne"]
    df = _make_df(labels)
    metrics = train_and_eval(df, out_dir)
    assert set(metrics.keys()) == EXPECTED_KEYS
    # CV ran (no cv_folds=0 collapse) on the 20 non-singleton rows only
    assert metrics["cv_folds"] >= 2
    assert metrics["cv_grouping"] in {"stratified_group", "group"}
    assert metrics["dropped_classes"] == ["zne"]
    assert metrics["n_samples"] == 21
    assert metrics["cv_n_samples"] == 20
    # confusion matrix covers only the CV rows; zne row is all-zero
    assert sum(sum(r) for r in metrics["confusion_matrix"]) == 20
    zne_idx = metrics["labels"].index("zne")
    assert sum(metrics["confusion_matrix"][zne_idx]) == 0
    # the persisted refit model was trained on ALL rows: zne stays a class
    assert "zne" in metrics["labels"]
    bundle = joblib.load(out_dir / "model.joblib")
    assert "zne" in bundle["classes"]
    assert "zne" in list(bundle["model"].classes_)


def test_single_surviving_class_falls_back(out_dir: Path) -> None:
    # 10x raw + 1x zne: dropping the singleton would leave ONE class — a
    # single-class "CV" would score a meaningless 1.0 (and GBC cannot fit
    # it), so the honest cv_folds=0 fallback on ALL rows runs instead.
    labels = ["raw"] * 10 + ["zne"]
    df = _make_df(labels)
    metrics = train_and_eval(df, out_dir)
    assert metrics["cv_folds"] == 0
    assert metrics["dropped_classes"] == []
    assert metrics["n_samples"] == 11
    assert metrics["cv_n_samples"] == 11
    assert (out_dir / "model.joblib").exists()
    # baseline = majority fraction on the training set
    assert metrics["baseline_accuracy"] == pytest.approx(10 / 11)


def test_all_singleton_classes_still_fall_back(out_dir: Path) -> None:
    # 1x raw + 1x zne -> EVERY class is a singleton; after dropping there is
    # nothing left to cross-validate -> honest cv_folds=0 fallback on all
    # rows (and dropped_classes is [] because nothing was excluded from the
    # evaluation actually performed).
    df = _make_df(["raw", "zne"])
    metrics = train_and_eval(df, out_dir)
    assert metrics["cv_folds"] == 0
    assert metrics["cv_grouping"] == "none (degenerate)"
    assert metrics["dropped_classes"] == []
    assert metrics["n_samples"] == 2
    assert metrics["cv_n_samples"] == 2
    assert (out_dir / "model.joblib").exists()


def test_reduced_fold_count_between_2_and_5(out_dir: Path) -> None:
    # 12x raw + 3x zne -> smallest class 3 (>= 2, so NOT dropped) -> 3-fold CV
    labels = ["raw"] * 12 + ["zne"] * 3
    df = _make_df(labels)
    metrics = train_and_eval(df, out_dir)
    assert metrics["cv_folds"] == 3
    assert metrics["dropped_classes"] == []
    assert metrics["cv_n_samples"] == 15


# --------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------


def test_missing_columns_raises(out_dir: Path) -> None:
    df = _learnable_df().drop(columns=["feat_depth"])
    with pytest.raises(ValueError, match="feat_depth"):
        train_and_eval(df, out_dir)

    df2 = _learnable_df().drop(columns=["best_technique"])
    with pytest.raises(ValueError, match="best_technique"):
        train_and_eval(df2, out_dir)


def test_zero_usable_rows_raises(out_dir: Path) -> None:
    df = _learnable_df(n_rows=4)
    df["best_technique"] = ""
    with pytest.raises(ValueError, match="zero usable rows"):
        train_and_eval(df, out_dir)


# --------------------------------------------------------------------------
# Aggregated (seed-averaged) schema + hold-out evaluations
# --------------------------------------------------------------------------


def test_aggregated_schema_accepted(out_dir: Path) -> None:
    """aggregated.csv schema: n_seeds column present, NO seed column —
    must train exactly like the raw schema (nothing reads 'seed')."""
    df = _learnable_df().drop(columns=["seed"])
    df["n_seeds"] = 3
    metrics = train_and_eval(df, out_dir)
    assert metrics["cv_folds"] == 5
    assert metrics["n_samples"] == 60
    assert metrics["accuracy"] >= metrics["baseline_accuracy"] + 0.2
    assert (out_dir / "model.joblib").exists()


def test_lobo_two_backends(out_dir: Path) -> None:
    """Leave-one-backend-out: each distinct backend string (incl. noise-
    scaled '@x<scale>' names) is one held-out noise environment."""
    df = _learnable_df()
    backends = ["FakeLagosV2@x2.0"] * 30 + ["FakeManilaV2"] * 30
    df["backend"] = backends
    metrics = train_and_eval(df, out_dir)
    lobo = metrics["lobo"]
    assert isinstance(lobo, dict)
    assert lobo["n_backends"] == 2
    assert set(lobo["per_backend_accuracy"]) == {
        "FakeLagosV2@x2.0",
        "FakeManilaV2",
    }
    assert set(lobo["per_backend_macro_f1"]) == set(lobo["per_backend_accuracy"])
    assert 0.0 <= lobo["accuracy"] <= 1.0
    assert 0.0 <= lobo["macro_f1"] <= 1.0
    # the depth->winner rule holds on both halves, so a model trained on one
    # backend's rows must recover it on the held-out backend
    assert lobo["accuracy"] >= 0.8
    # base devices differ too -> LODO also defined with 2 devices
    lodo = metrics["lodo"]
    assert isinstance(lodo, dict)
    assert lodo["n_devices"] == 2
    assert set(lodo["per_device_accuracy"]) == {"FakeLagosV2", "FakeManilaV2"}


def test_lodo_pools_scale_siblings_of_one_device(out_dir: Path) -> None:
    """The LOBO-leakage fix (fixer 2026-07-21): scale-siblings like
    FakeManilaV2 / @x1.5 / @x2.0 are ONE device for the leave-one-device-out
    evaluation, so 'new noise environment' can never be claimed from a fold
    whose device stayed in training at other scales."""
    df = _learnable_df()
    df["backend"] = (
        ["FakeManilaV2"] * 15
        + ["FakeManilaV2@x1.5"] * 15
        + ["FakeManilaV2@x2.0"] * 15
        + ["FakeLagosV2"] * 15
    )
    metrics = train_and_eval(df, out_dir)
    assert metrics["lobo"]["n_backends"] == 4  # one fold per backend STRING
    lodo = metrics["lodo"]
    assert isinstance(lodo, dict)
    assert lodo["n_devices"] == 2  # Manila (all 3 scales pooled) + Lagos
    assert set(lodo["per_device_accuracy"]) == {"FakeManilaV2", "FakeLagosV2"}
    # depth->winner rule is device-independent -> the held-out device is
    # still predictable
    assert lodo["accuracy"] >= 0.8


def test_lodo_none_for_single_device_multi_scale(out_dir: Path) -> None:
    """All scales of ONE device: LOBO still runs (scale interpolation) but
    LODO must be None — there is no second device to generalize to."""
    df = _learnable_df()
    df["backend"] = ["FakeManilaV2"] * 30 + ["FakeManilaV2@x1.5"] * 30
    metrics = train_and_eval(df, out_dir)
    assert isinstance(metrics["lobo"], dict)
    assert metrics["lobo"]["n_backends"] == 2
    assert metrics["lodo"] is None


def test_lobo_none_without_backend_column(out_dir: Path) -> None:
    df = _learnable_df().drop(columns=["backend"])
    metrics = train_and_eval(df, out_dir)
    assert metrics["lobo"] is None
    assert metrics["lodo"] is None


def test_lofo_computed_even_when_cv_degenerate(out_dir: Path) -> None:
    """LOFO/LOBO are independent of CV feasibility: 2 families of 1 row
    each -> cv_folds=0 fallback, but LOFO still runs on all rows."""
    df = _make_df(["raw", "zne"])
    df.loc[df.index[1], "family"] = "ghz_plus"
    metrics = train_and_eval(df, out_dir)
    assert metrics["cv_folds"] == 0
    assert isinstance(metrics["lofo"], dict)
    assert metrics["lofo"]["n_families"] == 2


# --------------------------------------------------------------------------
# train_and_eval_all: both winner labels -> two bundles
# --------------------------------------------------------------------------


def _dual_label_df(n_rows: int = 60) -> pd.DataFrame:
    """Learnable df with BOTH label columns; the cost-aware rule differs
    (rem/raw instead of zne/raw) so the two models are genuinely distinct."""
    df = _learnable_df(n_rows=n_rows)
    df["best_technique_cost_aware"] = [
        "rem" if i % 2 == 0 else "raw" for i in range(n_rows)
    ]
    return df


def test_train_and_eval_all_trains_both_bundles(out_dir: Path) -> None:
    df = _dual_label_df()
    result = train_and_eval_all(df, out_dir)
    assert set(result.keys()) == {"best_technique", "best_technique_cost_aware"}

    primary = result["best_technique"]
    cost = result["best_technique_cost_aware"]
    assert set(primary.keys()) == EXPECTED_KEYS  # exact schema, no extras
    assert set(cost.keys()) == EXPECTED_KEYS
    assert primary["label_column"] == "best_technique"
    assert cost["label_column"] == "best_technique_cost_aware"
    assert sorted(primary["labels"]) == ["raw", "zne"]
    assert sorted(cost["labels"]) == ["raw", "rem"]

    # two independent bundles + two metrics files on disk
    for name in (
        "model.joblib",
        "model_cost_aware.joblib",
        "metrics.json",
        "metrics_cost_aware.json",
    ):
        assert (out_dir / name).exists(), f"missing artifact: {name}"
    bundle = joblib.load(out_dir / "model.joblib")
    assert bundle["label_column"] == "best_technique"
    cost_bundle = joblib.load(out_dir / "model_cost_aware.joblib")
    assert cost_bundle["label_column"] == "best_technique_cost_aware"
    assert cost_bundle["classes"] == cost["labels"]

    # metrics.json embeds the cost-aware metrics under 'cost_aware' so the
    # report CLI renders both label variants without extra plumbing
    saved = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
    assert saved["cost_aware"] == cost
    for key in EXPECTED_KEYS:
        assert saved[key] == primary[key]
    saved_cost = json.loads(
        (out_dir / "metrics_cost_aware.json").read_text(encoding="utf-8")
    )
    assert saved_cost == cost


def test_train_and_eval_all_legacy_schema_no_cost_column(out_dir: Path) -> None:
    """Backward compat: old tiny/small CSVs without the cost-aware column
    must still train (accuracy-only model) without crashing."""
    df = _learnable_df()
    assert "best_technique_cost_aware" not in df.columns
    result = train_and_eval_all(df, out_dir)
    assert result["best_technique_cost_aware"] is None
    assert set(result["best_technique"].keys()) == EXPECTED_KEYS
    assert (out_dir / "model.joblib").exists()
    assert not (out_dir / "model_cost_aware.joblib").exists()
    saved = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
    assert "cost_aware" not in saved


def test_train_and_eval_all_unusable_cost_column_skips_gracefully(
    out_dir: Path,
) -> None:
    df = _learnable_df()
    df["best_technique_cost_aware"] = ""  # column present but all empty
    result = train_and_eval_all(df, out_dir)  # must NOT raise
    assert result["best_technique_cost_aware"] is None
    assert (out_dir / "model.joblib").exists()
    assert not (out_dir / "model_cost_aware.joblib").exists()


# --------------------------------------------------------------------------
# CLI script end-to-end
# --------------------------------------------------------------------------


def test_train_model_cli(tmp_path: Path) -> None:
    csv_path = tmp_path / "results.csv"
    out = tmp_path / "run"
    _learnable_df().to_csv(csv_path, index=False)
    script = (
        Path(__file__).resolve().parents[1] / "scripts" / "train_model.py"
    )
    proc = subprocess.run(
        [sys.executable, str(script), "--data", str(csv_path), "--out", str(out)],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, proc.stderr
    assert (out / "model.joblib").exists()
    assert (out / "metrics.json").exists()
    assert "best model" in proc.stdout
    assert "class balance" in proc.stdout
