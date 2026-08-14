"""V2 unit tests for qemsel.model (builder-model / B7).

Covers the four additive train_and_eval options (feature_version, calibrate,
abstain_threshold, extended_stats), the automatic leave-one-shot-budget-out
metric, the significance-aware 'tie' label, and the new CLI flags — all
without weakening the frozen V1 behaviour pinned by tests/test_model.py.

Cross-builder discipline (INTERFACES.md V2 convention 15): qemsel.stats is
owned by B6 and may still be a NotImplementedError stub, so every test that
needs ``sigma_shot`` / ``summarize_folds`` monkeypatches them with the
documented reference formulas (identical to B6's contract) via the
``ref_stats`` fixture. This keeps the suite green regardless of landing order.

Byte-identical duty: ``test_research_default_path_byte_identical`` re-runs the
default training on the real research aggregated.csv and asserts the full
metrics dict equals the stored results/research/metrics.json (captured from
the pre-edit code) for BOTH labels.
"""

from __future__ import annotations

import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from qemsel.features import FEATURE_NAMES, FEATURE_NAMES_V2
from qemsel.model import (
    SIGNIFICANT_LABEL,
    TIE_CLASS,
    derive_significant_label,
    train_and_eval,
    train_and_eval_all,
)

ROOT = Path(__file__).resolve().parents[1]
FEAT_COLS_V1 = ["feat_" + n for n in FEATURE_NAMES]
FEAT_COLS_V2 = ["feat_" + n for n in FEATURE_NAMES_V2]
V1_BUNDLE_KEYS = {
    "model", "feature_names", "classes", "model_name", "label_column",
    "qemsel_version",
}
V1_METRICS_KEYS = {
    "best_model_name", "accuracy", "macro_f1", "baseline_accuracy", "labels",
    "confusion_matrix", "feature_importances", "feature_importances_note",
    "per_model", "n_samples", "cv_n_samples", "cv_folds", "cv_grouping",
    "dropped_classes", "label_column", "lofo", "lobo", "lodo",
}


# ==========================================================================
# Reference stats (B6 contract) used to monkeypatch qemsel.stats stubs.
# ==========================================================================


def _ref_sigma_shot(value: float, shots: float) -> float:
    if shots <= 0:
        raise ValueError("shots must be > 0")
    if float(value) != float(value):  # NaN
        raise ValueError("value is NaN")
    return math.sqrt((1.0 - min(float(value) ** 2, 1.0)) / float(shots))


def _ref_summarize_folds(fold_scores) -> dict:
    arr = np.asarray(list(fold_scores), dtype=float)
    if arr.size == 0:
        raise ValueError("empty fold_scores")
    if np.isnan(arr).any():
        raise ValueError("NaN in fold_scores")
    return {
        "mean": float(arr.mean()),
        "std": float(np.std(arr, ddof=1)) if arr.size >= 2 else 0.0,
        "min": float(arr.min()),
        "max": float(arr.max()),
        "n_folds": int(arr.size),
    }


@pytest.fixture()
def ref_stats(monkeypatch):
    """Patch the B6 stats stubs with the documented reference formulas."""
    monkeypatch.setattr("qemsel.stats.sigma_shot", _ref_sigma_shot)
    monkeypatch.setattr("qemsel.stats.summarize_folds", _ref_summarize_folds)
    return None


# ==========================================================================
# DataFrame builders.
# ==========================================================================


def _learnable_df(n_rows: int = 60, *, version: int = 1, seed: int = 0) -> pd.DataFrame:
    """Balanced 2-class df with the rule winner=='zne' iff depth>20, else raw.

    version=2 additionally carries the five V2 feat_* columns. feat_depth
    holds the learnable signal; the other feats are random noise.
    """
    rng = np.random.default_rng(seed)
    names = FEATURE_NAMES if version == 1 else FEATURE_NAMES_V2
    rows = []
    for i in range(n_rows):
        depth = (5 if i % 2 == 0 else 35) + int(rng.integers(0, 5))
        n_qubits = 2 + i % 3
        row = {
            "circuit_id": f"c_q{n_qubits}_d{depth}_s{i}",
            "family": "layered_random",
            "n_qubits": n_qubits,
            "depth": depth,
            "seed": i,
            "backend": "FakeManilaV2",
            "pauli": "Z" * n_qubits,
            "ideal": 0.3,
        }
        for nm in names:
            row["feat_" + nm] = float(rng.uniform(0.0, 1.0))
        row["feat_n_qubits"] = float(n_qubits)
        row["feat_depth"] = float(depth)
        row["best_technique"] = "raw" if i % 2 == 0 else "zne"
        rows.append(row)
    return pd.DataFrame(rows)


def _perseed_row(techs: dict[str, tuple[float, float, float]]) -> dict:
    """Build the per-seed error/value/shots columns from {tech:(err,val,base)}.

    The ``base`` in each tuple is the per-execution BASE shot count (what the
    variance model uses). Real results.csv stores ``<tech>_shots`` as the
    CONSUMED-budget ledger (= multiplier x base; verified: raw 4096, rem
    3x, zne 3x, cdr/raw_plus 11x), so we materialise that here — the
    significance code divides it back out by ``shots_consumed(t, 1)`` to
    recover base. Storing base directly would understate cdr/rem/zne sigma.
    """
    from qemsel.mitigation import shots_consumed

    d: dict[str, float] = {}
    for t, (err, val, base) in techs.items():
        d[f"{t}_abs_error"] = err
        d[f"{t}_value"] = val
        try:
            mult = shots_consumed(t, 1)
        except Exception:
            mult = 1
        d[f"{t}_shots"] = base * mult
    return d


def _perseed_significance_df() -> pd.DataFrame:
    """Four rows exercising the four significance outcomes."""
    rows = [
        # 1) clear cdr winner: big margin, high shots -> 'cdr'
        _perseed_row({"raw": (0.30, 0.70, 4096), "cdr": (0.01, 0.99, 4096),
                      "rem": (0.25, 0.75, 4096)}),
        # 2) cdr 0.10 vs rem 0.11 at tiny shots -> within margin -> 'tie'
        _perseed_row({"raw": (0.30, 0.70, 16), "cdr": (0.10, 0.90, 16),
                      "rem": (0.11, 0.89, 16)}),
        # 3) all failed -> ''
        _perseed_row({"raw": (np.nan, np.nan, 4096), "cdr": (np.nan, np.nan, 4096),
                      "rem": (np.nan, np.nan, 4096)}),
        # 4) exactly one valid technique -> that technique outright
        _perseed_row({"raw": (np.nan, np.nan, 4096), "cdr": (np.nan, np.nan, 4096),
                      "rem": (0.05, 0.95, 4096)}),
    ]
    return pd.DataFrame(rows)


def _aggregated_significance_df() -> pd.DataFrame:
    """Two rows on the V2 aggregated schema (mean errors + n_seeds + base_shots)."""
    rows = [
        {"base_shots": 4096, "cdr_mean_abs_error": 0.01, "cdr_n_seeds": 3,
         "rem_mean_abs_error": 0.30, "rem_n_seeds": 3},   # clear cdr
        {"base_shots": 4, "cdr_mean_abs_error": 0.10, "cdr_n_seeds": 1,
         "rem_mean_abs_error": 0.11, "rem_n_seeds": 1},   # tie
    ]
    return pd.DataFrame(rows)


def _tie_training_df(n_rows: int = 48) -> pd.DataFrame:
    """Per-seed df whose DERIVED significance label is a 2-class {'cdr','tie'}
    problem, spread over 3 families and 2 backends for LOFO/LODO."""
    fams = ["layered_random", "ghz_plus", "mirror_circuit"]
    bks = ["FakeManilaV2", "FakeLagosV2"]
    rng = np.random.default_rng(3)
    rows = []
    for i in range(n_rows):
        even = i % 2 == 0
        depth = 5 if even else 35  # learnable signal, decorrelated from fam/bk
        n_qubits = 3
        if even:  # clear cdr winner
            techs = {"raw": (0.40, 0.60, 4096), "cdr": (0.01, 0.99, 4096),
                     "rem": (0.30, 0.70, 4096)}
        else:  # cdr vs rem within margin at tiny shots -> tie
            techs = {"raw": (0.40, 0.60, 16), "cdr": (0.10, 0.90, 16),
                     "rem": (0.11, 0.89, 16)}
        row = {
            "circuit_id": f"c{i}",
            "family": fams[i % 3],
            "n_qubits": n_qubits,
            "depth": depth,
            "seed": i,
            "backend": bks[(i // 3) % 2],
            "pauli": "ZZZ",
            "ideal": 0.3,
        }
        for nm in FEATURE_NAMES:
            row["feat_" + nm] = float(rng.uniform(0.0, 1.0))
        row["feat_depth"] = float(depth)
        row.update(_perseed_row(techs))
        rows.append(row)
    return pd.DataFrame(rows)


# ==========================================================================
# derive_significant_label
# ==========================================================================


def test_derive_perseed_four_outcomes(ref_stats) -> None:
    labels = derive_significant_label(_perseed_significance_df())
    assert list(labels) == ["cdr", TIE_CLASS, "", "rem"]
    # returned Series is aligned to df.index, object dtype
    assert list(labels.index) == [0, 1, 2, 3]
    assert labels.dtype == object


def test_derive_perseed_k_sigma_controls_tie(ref_stats) -> None:
    df = _perseed_significance_df()
    # row 1 (cdr 0.10 vs rem 0.11, shots 16): a very small k_sigma removes the
    # tie (margin now clears the tighter bar) -> the winner name appears.
    strict = derive_significant_label(df, k_sigma=0.05)
    assert strict.iloc[1] == "cdr"
    # a large k_sigma turns even the clear row-0 winner into a tie
    loose = derive_significant_label(df, k_sigma=1000.0)
    assert loose.iloc[0] == TIE_CLASS


def test_derive_aggregated_route(ref_stats) -> None:
    labels = derive_significant_label(_aggregated_significance_df())
    assert list(labels) == ["cdr", TIE_CLASS]


def test_derive_aggregated_requires_base_shots(ref_stats) -> None:
    df = _aggregated_significance_df().drop(columns=["base_shots"])
    with pytest.raises(ValueError, match="base_shots"):
        derive_significant_label(df)


def test_derive_techniques_filter(ref_stats) -> None:
    # Restrict to raw+rem: row 0's cdr winner is now ignored; among raw(0.30)
    # and rem(0.25) rem wins by a huge margin -> 'rem'.
    labels = derive_significant_label(
        _perseed_significance_df(), techniques=["raw", "rem"]
    )
    assert labels.iloc[0] == "rem"


def test_derive_techniques_filter_none_present_raises(ref_stats) -> None:
    with pytest.raises(ValueError, match="none of techniques"):
        derive_significant_label(_perseed_significance_df(), techniques=["nope"])


def test_derive_no_error_columns_raises(ref_stats) -> None:
    with pytest.raises(ValueError, match="no technique error columns"):
        derive_significant_label(pd.DataFrame({"family": ["a"], "ideal": [0.1]}))


def test_derive_does_not_call_sigma_when_stub(monkeypatch) -> None:
    """A single-valid-technique row must NOT need sigma_shot (no runner-up)."""
    def _boom(*a, **k):  # simulate B6 not landed
        raise NotImplementedError

    monkeypatch.setattr("qemsel.stats.sigma_shot", _boom)
    df = pd.DataFrame(_perseed_row(
        {"raw": (np.nan, np.nan, 4096), "cdr": (0.05, 0.95, 4096)}
    ), index=[0])
    labels = derive_significant_label(df)
    assert labels.iloc[0] == "cdr"


# ==========================================================================
# feature_version plumbing
# ==========================================================================


def test_feature_version_2_trains_and_records(out_dir: Path) -> None:
    df = _learnable_df(version=2)
    metrics = train_and_eval(df, out_dir, feature_version=2)
    assert metrics["feature_version"] == 2
    assert metrics["cv_folds"] == 5
    # importances keyed by the 15-feature V2 vector
    assert set(metrics["feature_importances"]) == set(FEAT_COLS_V2)
    bundle = joblib.load(out_dir / "model.joblib")
    assert bundle["feature_names"] == FEAT_COLS_V2
    assert bundle["feature_version"] == 2
    assert bundle["calibrated"] is False
    assert bundle["abstain_threshold"] is None
    # the persisted model predicts from a 15-wide named row
    pred = bundle["model"].predict(df[FEAT_COLS_V2].astype(float).iloc[:1])
    assert pred[0] in metrics["labels"]


def test_feature_version_unknown_raises(out_dir: Path) -> None:
    with pytest.raises(ValueError, match="feature_version"):
        train_and_eval(_learnable_df(), out_dir, feature_version=3)


def test_feature_version_1_default_has_no_feature_version_key(out_dir: Path) -> None:
    metrics = train_and_eval(_learnable_df(), out_dir)
    assert "feature_version" not in metrics
    assert "loso" not in metrics  # no base_shots column -> no shots-axis metric
    bundle = joblib.load(out_dir / "model.joblib")
    assert set(bundle) == V1_BUNDLE_KEYS  # no feature_version on the V1 path


# ==========================================================================
# calibrate
# ==========================================================================


def test_calibrate_bundle_and_metrics(out_dir: Path) -> None:
    df = _learnable_df(n_rows=40)
    base_metrics = train_and_eval(df, out_dir / "plain")
    metrics = train_and_eval(df, out_dir / "cal", calibrate=True)

    # calibration does not change the reported CV accuracy (same OOF loop)
    assert metrics["accuracy"] == base_metrics["accuracy"]
    assert metrics["macro_f1"] == base_metrics["macro_f1"]

    cal = metrics["calibration"]
    assert set(cal) == {"method", "brier_before", "brier_after"}
    assert cal["method"] == "sigmoid"
    assert isinstance(cal["brier_before"], float)
    assert isinstance(cal["brier_after"], float)

    bundle = joblib.load(out_dir / "cal" / "model.joblib")
    assert bundle["calibrated"] is True
    assert bundle["feature_version"] == 1
    assert bundle["abstain_threshold"] is None
    assert hasattr(bundle["model"], "predict_proba")
    assert list(bundle["model"].classes_) == bundle["classes"]

    # calibrated probabilities differ from a plain refit's probabilities
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

    factory = {
        "random_forest": RandomForestClassifier,
        "gradient_boosting": GradientBoostingClassifier,
    }[bundle["model_name"]]
    plain = factory(random_state=0).fit(df[FEAT_COLS_V1].astype(float), df["best_technique"])
    X = df[FEAT_COLS_V1].astype(float)
    p_cal = bundle["model"].predict_proba(X)
    p_plain = plain.predict_proba(X)
    assert np.abs(p_cal - p_plain).max() > 1e-3


# ==========================================================================
# abstain_threshold
# ==========================================================================


def test_abstain_threshold_stored_and_rate(out_dir: Path) -> None:
    metrics = train_and_eval(_learnable_df(), out_dir, abstain_threshold=0.6)
    assert metrics["abstain_threshold"] == 0.6
    assert 0.0 <= metrics["abstain_rate_cv"] <= 1.0
    bundle = joblib.load(out_dir / "model.joblib")
    assert bundle["abstain_threshold"] == 0.6
    assert bundle["calibrated"] is False
    assert bundle["feature_version"] == 1


def test_abstain_high_threshold_abstains_more(out_dir: Path) -> None:
    lo = train_and_eval(_learnable_df(), out_dir / "lo", abstain_threshold=0.5)
    hi = train_and_eval(_learnable_df(), out_dir / "hi", abstain_threshold=0.999)
    assert hi["abstain_rate_cv"] >= lo["abstain_rate_cv"]


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.5])
def test_abstain_out_of_range_raises(out_dir: Path, bad: float) -> None:
    with pytest.raises(ValueError, match="abstain_threshold"):
        train_and_eval(_learnable_df(), out_dir, abstain_threshold=bad)


def test_abstain_with_calibrate_uses_calibrated_scores(out_dir: Path) -> None:
    metrics = train_and_eval(
        _learnable_df(n_rows=40), out_dir, calibrate=True, abstain_threshold=0.6
    )
    assert "calibration" in metrics
    assert metrics["abstain_threshold"] == 0.6
    bundle = joblib.load(out_dir / "model.joblib")
    assert bundle["calibrated"] is True
    assert bundle["abstain_threshold"] == 0.6


# ==========================================================================
# extended_stats
# ==========================================================================


def test_extended_stats_adds_fold_info(out_dir: Path, ref_stats) -> None:
    metrics = train_and_eval(_learnable_df(), out_dir, extended_stats=True)
    assert set(metrics["fold_summary"]) == {"mean", "std", "min", "max", "n_folds"}
    assert metrics["fold_summary"]["n_folds"] == metrics["cv_folds"]
    for name, entry in metrics["per_model"].items():
        assert "fold_accuracies" in entry
        assert len(entry["fold_accuracies"]) == metrics["cv_folds"]
    # JSON round-trips
    assert json.loads(json.dumps(metrics))["fold_summary"]["n_folds"] == metrics[
        "cv_folds"
    ]


def test_extended_stats_off_keeps_v1_shape(out_dir: Path) -> None:
    metrics = train_and_eval(_learnable_df(), out_dir)
    assert "fold_summary" not in metrics
    for entry in metrics["per_model"].values():
        assert "fold_accuracies" not in entry
        assert set(entry) == {"accuracy", "accuracy_std", "macro_f1"}


# ==========================================================================
# loso (leave-one-shot-budget-out)
# ==========================================================================


def test_loso_two_budgets(out_dir: Path) -> None:
    df = pd.concat(
        [
            _learnable_df(seed=0).assign(base_shots=256),
            _learnable_df(seed=1).assign(base_shots=4096),
        ],
        ignore_index=True,
    )
    metrics = train_and_eval(df, out_dir)
    loso = metrics["loso"]
    assert isinstance(loso, dict)
    assert loso["n_budgets"] == 2
    assert set(loso["per_budget_accuracy"]) == {"256", "4096"}
    assert set(loso["per_budget_macro_f1"]) == set(loso["per_budget_accuracy"])
    assert 0.0 <= loso["accuracy"] <= 1.0


def test_loso_absent_single_budget(out_dir: Path) -> None:
    # A base_shots column with a single distinct value must NOT trigger loso.
    df = _learnable_df().assign(base_shots=1024)
    metrics = train_and_eval(df, out_dir)
    assert "loso" not in metrics


# ==========================================================================
# 'tie' class flows through CV / LOFO / LODO like any class
# ==========================================================================


def test_tie_class_is_first_class_citizen(out_dir: Path, ref_stats) -> None:
    df = _tie_training_df()
    df[SIGNIFICANT_LABEL] = derive_significant_label(df)
    # sanity: the derived label really contains the tie class with >= 2 members
    counts = df[SIGNIFICANT_LABEL].value_counts()
    assert counts.get(TIE_CLASS, 0) >= 2
    metrics = train_and_eval(
        df,
        out_dir,
        label_column=SIGNIFICANT_LABEL,
        bundle_filename="model_significant.joblib",
        metrics_filename="metrics_significant.json",
    )
    assert TIE_CLASS in metrics["labels"]
    # tie participates in CV (its confusion-matrix row is not forced to zero)
    tie_idx = metrics["labels"].index(TIE_CLASS)
    assert sum(metrics["confusion_matrix"][tie_idx]) > 0
    # LOFO / LODO evaluate with tie as an ordinary class
    assert isinstance(metrics["lofo"], dict)
    assert metrics["lofo"]["n_families"] == 3
    assert isinstance(metrics["lodo"], dict)
    assert metrics["lodo"]["n_devices"] == 2
    bundle = joblib.load(out_dir / "model_significant.joblib")
    assert TIE_CLASS in bundle["classes"]
    assert bundle["label_column"] == SIGNIFICANT_LABEL


# ==========================================================================
# train_and_eval_all forwards the four kwargs
# ==========================================================================


def test_train_and_eval_all_forwards_v2_kwargs(out_dir: Path) -> None:
    df = _learnable_df()
    df["best_technique_cost_aware"] = [
        "rem" if i % 2 == 0 else "raw" for i in range(len(df))
    ]
    # abstain_threshold reaches BOTH underlying train_and_eval calls (kept
    # abstain-only, not calibrate, so the forwarding test stays cheap).
    results = train_and_eval_all(df, out_dir, abstain_threshold=0.6)
    for name in ("model.joblib", "model_cost_aware.joblib"):
        bundle = joblib.load(out_dir / name)
        assert bundle["abstain_threshold"] == 0.6
        assert bundle["calibrated"] is False
    assert results["best_technique"]["abstain_threshold"] == 0.6
    assert results["best_technique_cost_aware"]["abstain_threshold"] == 0.6


# ==========================================================================
# Byte-identical regression on the real research data (both labels).
# ==========================================================================


@pytest.mark.slow
def test_research_default_path_byte_identical(out_dir: Path) -> None:
    agg_path = ROOT / "results" / "research" / "aggregated.csv"
    ref_path = ROOT / "results" / "research" / "metrics.json"
    if not (agg_path.exists() and ref_path.exists()):
        pytest.skip("research artifacts not present")
    agg = pd.read_csv(agg_path)
    stored = json.loads(ref_path.read_text(encoding="utf-8"))
    stored_primary = {k: v for k, v in stored.items() if k != "cost_aware"}

    primary = train_and_eval(agg, out_dir, "best_technique")
    assert primary == stored_primary  # full-dict byte-identical
    # no V2 keys leak onto the default path
    assert V1_METRICS_KEYS == set(primary)

    cost = train_and_eval(
        agg,
        out_dir,
        "best_technique_cost_aware",
        bundle_filename="model_cost_aware.joblib",
        metrics_filename="metrics_cost_aware.json",
    )
    assert cost == stored["cost_aware"]


# ==========================================================================
# CLI: new flags.
# ==========================================================================


def _load_cli():
    """Import scripts/train_model.py as a module for in-process main() calls."""
    path = ROOT / "scripts" / "train_model.py"
    spec = importlib.util.spec_from_file_location("qemsel_train_model_cli", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_cli_calibrate_subprocess(tmp_path: Path) -> None:
    """--calibrate needs no B6 stats, so it runs end-to-end via subprocess."""
    csv = tmp_path / "results.csv"
    out = tmp_path / "run"
    _learnable_df().to_csv(csv, index=False)
    script = ROOT / "scripts" / "train_model.py"
    proc = subprocess.run(
        [sys.executable, str(script), "--data", str(csv), "--out", str(out),
         "--calibrate", "--abstain-threshold", "0.6"],
        capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, proc.stderr
    bundle = joblib.load(out / "model.joblib")
    assert bundle["calibrated"] is True
    assert bundle["abstain_threshold"] == 0.6
    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert "calibration" in metrics
    assert metrics["abstain_threshold"] == 0.6


def test_cli_label_significant_in_process(tmp_path: Path, monkeypatch) -> None:
    """--label significant derives the tie label and writes its own bundle.

    Run in-process (not subprocess) so the B6 sigma_shot stub can be
    monkeypatched with the reference formula.
    """
    monkeypatch.setattr("qemsel.stats.sigma_shot", _ref_sigma_shot)
    df = _tie_training_df()
    csv = tmp_path / "results.csv"
    out = tmp_path / "run"
    df.to_csv(csv, index=False)
    cli = _load_cli()
    rc = cli.main(["--data", str(csv), "--out", str(out), "--label", "significant"])
    assert rc == 0
    assert (out / "model_significant.joblib").exists()
    assert (out / "metrics_significant.json").exists()
    metrics = json.loads((out / "metrics_significant.json").read_text(encoding="utf-8"))
    assert metrics["label_column"] == SIGNIFICANT_LABEL
    assert TIE_CLASS in metrics["labels"]
    # the raw CSV was never mutated with the derived label column
    assert SIGNIFICANT_LABEL not in pd.read_csv(csv).columns
