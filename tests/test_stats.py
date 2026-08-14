"""Tests for qemsel.stats (B6 statistical-hygiene module).

All synthetic inputs are hand-checkable; the real-data anchors at the bottom
run the checklist against the frozen research sweep and pin the counts
recorded in notes/B6-stats.md. Determinism (fixed seed -> identical output)
and NaN policy are exercised for every stochastic function.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qemsel import stats

_RESEARCH = Path(__file__).resolve().parents[1] / "results" / "research"


# --------------------------------------------------------------------------- #
# sigma_shot
# --------------------------------------------------------------------------- #

def test_sigma_shot_known_values():
    assert stats.sigma_shot(0.0, 100) == pytest.approx(0.1)
    assert stats.sigma_shot(1.0, 100) == 0.0
    assert stats.sigma_shot(-1.0, 100) == 0.0
    assert stats.sigma_shot(0.6, 64) == pytest.approx(math.sqrt((1 - 0.36) / 64))


def test_sigma_shot_unphysical_value_clamps_to_zero():
    # |value| > 1 clamps variance term to 0 -> sigma 0 (honest lower bound).
    assert stats.sigma_shot(1.5, 100) == 0.0
    assert stats.sigma_shot(-2.0, 500) == 0.0


def test_sigma_shot_returns_plain_float():
    v = stats.sigma_shot(0.0, 100)
    assert type(v) is float


@pytest.mark.parametrize("shots", [0, -5, float("nan")])
def test_sigma_shot_bad_shots_raises(shots):
    with pytest.raises(ValueError):
        stats.sigma_shot(0.0, shots)


def test_sigma_shot_nan_value_raises():
    with pytest.raises(ValueError):
        stats.sigma_shot(float("nan"), 100)


# --------------------------------------------------------------------------- #
# win_shares
# --------------------------------------------------------------------------- #

def test_win_shares_basic_and_sums_to_one():
    labels = ["a", "a", "b", "", "a", np.nan]  # valid: a,a,b,a -> denom 4
    ws = stats.win_shares(labels)
    assert ws == {"a": 0.75, "b": 0.25}
    assert sum(ws.values()) == pytest.approx(1.0)


def test_win_shares_explicit_techniques_absent_get_zero():
    ws = stats.win_shares(["a", "a", "b"], techniques=["a", "b", "c"])
    assert ws == {"a": pytest.approx(2 / 3), "b": pytest.approx(1 / 3), "c": 0.0}


def test_win_shares_all_excluded_returns_empty_or_zeros():
    assert stats.win_shares(["", np.nan, None]) == {}
    assert stats.win_shares([""], techniques=["a"]) == {"a": 0.0}


def test_win_shares_accepts_series():
    ws = stats.win_shares(pd.Series(["zne", "zne", "cdr"]))
    assert ws == {"cdr": pytest.approx(1 / 3), "zne": pytest.approx(2 / 3)}


# --------------------------------------------------------------------------- #
# bootstrap_ci
# --------------------------------------------------------------------------- #

def test_bootstrap_ci_constant_data_is_exact():
    # Every resample of a constant vector has the same mean -> lo=hi=estimate.
    out = stats.bootstrap_ci([5.0, 5.0, 5.0, 5.0], n_boot=200, seed=1)
    assert out["estimate"] == 5.0
    assert out["lo"] == 5.0 and out["hi"] == 5.0
    assert out["n"] == 4 and out["n_dropped_nan"] == 0


def test_bootstrap_ci_keys_exact():
    out = stats.bootstrap_ci([1.0, 2.0, 3.0], n_boot=100, seed=0)
    assert set(out) == {
        "estimate", "lo", "hi", "ci", "n", "n_dropped_nan", "n_boot", "seed",
    }
    assert out["estimate"] == pytest.approx(2.0)
    assert out["ci"] == 0.95 and out["n_boot"] == 100 and out["seed"] == 0


def test_bootstrap_ci_drops_nan():
    out = stats.bootstrap_ci([1.0, 2.0, 3.0, np.nan], n_boot=100, seed=0)
    assert out["n"] == 3 and out["n_dropped_nan"] == 1
    assert out["estimate"] == pytest.approx(2.0)


def test_bootstrap_ci_deterministic():
    a = stats.bootstrap_ci([1.0, 5.0, 2.0, 8.0, 3.0], n_boot=500, seed=7)
    b = stats.bootstrap_ci([1.0, 5.0, 2.0, 8.0, 3.0], n_boot=500, seed=7)
    assert a == b
    c = stats.bootstrap_ci([1.0, 5.0, 2.0, 8.0, 3.0], n_boot=500, seed=8)
    # Different seed -> generally different bounds, still bracketing estimate.
    assert c["lo"] <= c["estimate"] <= c["hi"]


def test_bootstrap_ci_custom_statistic():
    out = stats.bootstrap_ci(
        [1.0, 2.0, 3.0, 4.0, 100.0], statistic=lambda a: float(np.median(a)),
        n_boot=300, seed=3,
    )
    assert out["estimate"] == 3.0
    assert out["lo"] <= 3.0 <= out["hi"]


def test_bootstrap_ci_bad_ci_raises():
    with pytest.raises(ValueError):
        stats.bootstrap_ci([1.0, 2.0], ci=1.5)
    with pytest.raises(ValueError):
        stats.bootstrap_ci([1.0, 2.0], ci=0.0)


def test_bootstrap_ci_too_few_values_raises():
    with pytest.raises(ValueError):
        stats.bootstrap_ci([1.0])
    with pytest.raises(ValueError):
        stats.bootstrap_ci([1.0, np.nan])


def test_bootstrap_ci_json_serializable():
    out = stats.bootstrap_ci([1.0, 2.0, 3.0], n_boot=50, seed=0)
    json.dumps(out)  # must not raise
    assert all(type(out[k]) in (float, int) for k in out)


# --------------------------------------------------------------------------- #
# win_share_ci
# --------------------------------------------------------------------------- #

def test_win_share_ci_all_wins_is_one():
    out = stats.win_share_ci(["a", "a", "a", "a"], "a", n_boot=200, seed=0)
    assert out["estimate"] == 1.0 and out["lo"] == 1.0 and out["hi"] == 1.0
    assert out["technique"] == "a"


def test_win_share_ci_absent_technique_is_zero():
    out = stats.win_share_ci(["a", "a", "b"], "zzz", n_boot=200, seed=0)
    assert out["estimate"] == 0.0 and out["lo"] == 0.0 and out["hi"] == 0.0


def test_win_share_ci_excludes_empty_and_nan():
    # valid labels a,a,b -> share of a = 2/3.
    out = stats.win_share_ci(["a", "a", "b", "", np.nan], "a", n_boot=400, seed=2)
    assert out["estimate"] == pytest.approx(2 / 3)
    assert out["n"] == 3
    assert set(out) >= {"estimate", "lo", "hi", "technique"}


# --------------------------------------------------------------------------- #
# paired_permutation_test
# --------------------------------------------------------------------------- #

def test_permutation_identical_arrays_pvalue_one():
    a = [0.1, 0.2, 0.3, 0.4]
    out = stats.paired_permutation_test(a, a, n_perm=200, seed=0)
    assert out["mean_diff"] == 0.0
    # Every permutation mean is 0 = |observed| -> all extreme -> p = 1.
    assert out["p_value"] == 1.0
    assert out["n_pairs"] == 4 and out["n_dropped_nan"] == 0


def test_permutation_strong_shift_small_pvalue():
    rng = np.random.default_rng(0)
    b = rng.normal(size=40)
    a = b + 1.0  # A uniformly worse by 1
    out = stats.paired_permutation_test(a, b, n_perm=5000, seed=1)
    assert out["mean_diff"] == pytest.approx(1.0)
    assert out["p_value"] < 0.01


def test_permutation_one_sided_directions():
    rng = np.random.default_rng(1)
    b = rng.normal(size=40)
    a = b - 0.8  # A better (smaller)
    less = stats.paired_permutation_test(a, b, n_perm=5000, seed=2, alternative="less")
    greater = stats.paired_permutation_test(
        a, b, n_perm=5000, seed=2, alternative="greater"
    )
    assert less["mean_diff"] == pytest.approx(-0.8)
    assert less["p_value"] < 0.01
    assert greater["p_value"] > 0.9


def test_permutation_pairwise_nan_drop():
    a = [1.0, 2.0, np.nan, 4.0]
    b = [0.0, 0.0, 0.0, 0.0]
    out = stats.paired_permutation_test(a, b, n_perm=100, seed=0)
    assert out["n_pairs"] == 3 and out["n_dropped_nan"] == 1
    assert out["mean_diff"] == pytest.approx((1 + 2 + 4) / 3)


def test_permutation_deterministic():
    a = [0.1, 0.5, 0.2, 0.9, 0.3]
    b = [0.2, 0.1, 0.4, 0.2, 0.7]
    x = stats.paired_permutation_test(a, b, n_perm=1000, seed=5)
    y = stats.paired_permutation_test(a, b, n_perm=1000, seed=5)
    assert x == y


def test_permutation_keys_exact():
    out = stats.paired_permutation_test([1.0, 2.0], [0.0, 1.0], n_perm=50, seed=0)
    assert set(out) == {
        "mean_diff", "p_value", "n_pairs", "n_dropped_nan",
        "n_perm", "alternative", "seed",
    }
    json.dumps(out)


def test_permutation_errors():
    with pytest.raises(ValueError):
        stats.paired_permutation_test([1.0, 2.0], [1.0])  # length mismatch
    with pytest.raises(ValueError):
        stats.paired_permutation_test([1.0], [2.0])  # < 2 pairs
    with pytest.raises(ValueError):
        stats.paired_permutation_test([1.0, 2.0], [3.0, 4.0], alternative="bogus")


# --------------------------------------------------------------------------- #
# cliffs_delta
# --------------------------------------------------------------------------- #

def test_cliffs_delta_full_separation():
    assert stats.cliffs_delta([10, 11, 12], [1, 2, 3]) == 1.0
    assert stats.cliffs_delta([1, 2, 3], [10, 11, 12]) == -1.0


def test_cliffs_delta_symmetric_zero():
    assert stats.cliffs_delta([1, 2, 3], [1, 2, 3]) == 0.0


def test_cliffs_delta_known_fraction():
    # a=[1,2], b=[2]: pairs (1<2)=less, (2==2)=tie -> (0-1)/(2*1) = -0.5.
    assert stats.cliffs_delta([1, 2], [2]) == -0.5


def test_cliffs_delta_drops_nan_per_array():
    assert stats.cliffs_delta([10, np.nan, 12], [1, 2, 3]) == 1.0


def test_cliffs_delta_empty_raises():
    with pytest.raises(ValueError):
        stats.cliffs_delta([np.nan], [1, 2, 3])
    with pytest.raises(ValueError):
        stats.cliffs_delta([1, 2, 3], [])


def test_cliffs_delta_returns_plain_float():
    assert type(stats.cliffs_delta([1, 2], [0, 3])) is float


# --------------------------------------------------------------------------- #
# summarize_folds
# --------------------------------------------------------------------------- #

def test_summarize_folds_known():
    out = stats.summarize_folds([0.8, 0.9, 1.0])
    assert out["mean"] == pytest.approx(0.9)
    assert out["std"] == pytest.approx(0.1)  # ddof=1
    assert out["min"] == 0.8 and out["max"] == 1.0 and out["n_folds"] == 3
    assert set(out) == {"mean", "std", "min", "max", "n_folds"}


def test_summarize_folds_single_fold_std_zero():
    out = stats.summarize_folds([0.5])
    assert out["std"] == 0.0 and out["n_folds"] == 1 and out["mean"] == 0.5


def test_summarize_folds_errors():
    with pytest.raises(ValueError):
        stats.summarize_folds([])
    with pytest.raises(ValueError):
        stats.summarize_folds([0.9, float("nan")])


def test_summarize_folds_json_serializable():
    json.dumps(stats.summarize_folds([0.7, 0.8]))


# --------------------------------------------------------------------------- #
# koester_checklist — synthetic per-seed
# --------------------------------------------------------------------------- #

def _per_seed_df() -> pd.DataFrame:
    """Three rows, techniques raw & zne, with hand-checked flags:

    row0: clean, best=zne (0.05 < 0.10)
    row1: zne overshoots (value 1.5, ideal 0.0 -> err 1.5 beyond physical),
          best=raw
    row2: raw refuses (NaN) -> best=zne, nan_rate raw = 1/3
    """
    return pd.DataFrame(
        {
            "ideal": [0.5, 0.0, 0.3],
            "raw_value": [0.4, 0.2, np.nan],
            "raw_abs_error": [0.10, 0.20, np.nan],
            "raw_shots": [100, 100, 100],
            "zne_value": [0.45, 1.5, 0.35],
            "zne_abs_error": [0.05, 1.50, 0.05],
            "zne_shots": [100, 100, 100],
            "best_technique": ["zne", "raw", "zne"],
        }
    )


def test_checklist_per_seed_structure_and_flags():
    res = stats.koester_checklist(_per_seed_df())
    assert res["schema"] == "per_seed"
    assert res["techniques"] == ["raw", "zne"]
    assert res["n_rows"] == 3
    assert set(res) == {"schema", "n_rows", "techniques", "checks", "passed"}
    c = res["checks"]
    assert c["overshoot_beyond_physical_max"] == {"raw": 0, "zne": 1}
    assert c["error_beyond_physical_max"] == {"raw": 0, "zne": 1}
    assert c["nan_rate"]["raw"] == pytest.approx(1 / 3)
    assert c["nan_rate"]["zne"] == 0.0
    assert c["label_argmin_consistent"] == {"n_checked": 3, "n_mismatch": 0}
    assert c["partial_coverage_winners"] is None
    assert res["passed"] is True
    json.dumps(res)


def test_checklist_detects_label_mismatch():
    df = _per_seed_df()
    df.loc[0, "best_technique"] = "raw"  # wrong: zne is the true argmin
    res = stats.koester_checklist(df)
    assert res["checks"]["label_argmin_consistent"]["n_mismatch"] == 1
    assert res["passed"] is False


def test_checklist_winner_margin_flags_ties():
    # Both rows: values 0 -> sigma_shot(0,100)=0.1 each, combined ~0.1414,
    # k_sigma=2 -> threshold ~0.283.
    df = pd.DataFrame(
        {
            "raw_value": [0.0, 0.0],
            "raw_abs_error": [0.10, 0.10],
            "raw_shots": [100, 100],
            "zne_value": [0.0, 0.0],
            "zne_abs_error": [0.12, 0.90],  # row0 margin 0.02 (tie), row1 0.80
            "zne_shots": [100, 100],
            "best_technique": ["raw", "raw"],
        }
    )
    wm = stats.koester_checklist(df, k_sigma=2.0)["checks"]["winner_margin_below_k_sigma"]
    assert wm["k_sigma"] == 2.0
    assert wm["n_flagged"] == 1
    assert wm["fraction"] == pytest.approx(0.5)


def test_checklist_error_beyond_none_without_ideal():
    df = _per_seed_df().drop(columns=["ideal"])
    res = stats.koester_checklist(df)
    assert res["checks"]["error_beyond_physical_max"] is None
    # overshoot still available (value columns present)
    assert res["checks"]["overshoot_beyond_physical_max"] == {"raw": 0, "zne": 1}


# --------------------------------------------------------------------------- #
# koester_checklist — synthetic aggregated
# --------------------------------------------------------------------------- #

def _aggregated_df(partial: bool) -> pd.DataFrame:
    """Two rows, techniques raw & zne. When ``partial`` the winner (zne on
    row0) has coverage below n_seeds, exercising both the coverage-aware
    argmin and the partial_coverage_winners flag."""
    zne_cov = 1 if partial else 3
    return pd.DataFrame(
        {
            "n_seeds": [3, 3],
            "raw_mean_abs_error": [0.20, 0.10],
            "raw_n_seeds": [3, 3],
            "zne_mean_abs_error": [0.05, 0.30],
            "zne_n_seeds": [zne_cov, 3],
            # With full coverage zne wins row0; with partial coverage zne is
            # ineligible so raw wins row0. best_technique is set accordingly.
            "best_technique": ["zne" if not partial else "raw", "raw"],
        }
    )


def test_checklist_aggregated_full_coverage():
    res = stats.koester_checklist(_aggregated_df(partial=False))
    assert res["schema"] == "aggregated"
    c = res["checks"]
    assert c["overshoot_beyond_physical_max"] is None
    assert c["error_beyond_physical_max"] is None
    assert c["winner_margin_below_k_sigma"]["n_flagged"] is None
    assert c["winner_margin_below_k_sigma"]["fraction"] is None
    assert c["label_argmin_consistent"] == {"n_checked": 2, "n_mismatch": 0}
    assert c["partial_coverage_winners"] == 0
    assert res["passed"] is True


def test_checklist_aggregated_coverage_restricted_argmin():
    # zne has the smaller mean on row0 but only 1/3 coverage -> ineligible;
    # experiment picks raw, and our coverage-aware argmin must agree (no
    # spurious mismatch). partial_coverage_winners stays 0 because the
    # WINNER (raw) has full coverage.
    res = stats.koester_checklist(_aggregated_df(partial=True))
    c = res["checks"]
    assert c["label_argmin_consistent"]["n_mismatch"] == 0
    assert c["partial_coverage_winners"] == 0
    assert res["passed"] is True


def test_checklist_aggregated_partial_winner_flagged():
    # Force a partial-coverage technique to be the recorded winner -> the
    # coverage guard flags it and 'passed' goes False.
    df = _aggregated_df(partial=True)
    df.loc[0, "best_technique"] = "zne"  # zne only had 1/3 seeds
    res = stats.koester_checklist(df)
    assert res["checks"]["partial_coverage_winners"] == 1
    assert res["passed"] is False


# --------------------------------------------------------------------------- #
# koester_checklist — error handling
# --------------------------------------------------------------------------- #

def test_checklist_no_error_columns_raises():
    with pytest.raises(ValueError):
        stats.koester_checklist(pd.DataFrame({"best_technique": ["raw"]}))


def test_checklist_no_best_technique_raises():
    with pytest.raises(ValueError):
        stats.koester_checklist(pd.DataFrame({"raw_abs_error": [0.1, 0.2]}))


# --------------------------------------------------------------------------- #
# Real-data anchors (frozen research sweep) — pin the notes numbers
# --------------------------------------------------------------------------- #

@pytest.mark.skipif(
    not (_RESEARCH / "results.csv").exists(), reason="research results.csv absent"
)
def test_checklist_real_results_csv():
    df = pd.read_csv(_RESEARCH / "results.csv")
    res = stats.koester_checklist(df, k_sigma=2.0)
    assert res["schema"] == "per_seed"
    assert res["techniques"] == ["raw", "raw_plus", "zne", "cdr", "rem"]
    assert res["n_rows"] == 1620
    c = res["checks"]
    assert c["overshoot_beyond_physical_max"] == {
        "raw": 0, "raw_plus": 0, "zne": 5, "cdr": 159, "rem": 53,
    }
    assert c["error_beyond_physical_max"]["rem"] == 5
    assert c["nan_rate"]["cdr"] == pytest.approx(415 / 1620)
    assert c["nan_rate"]["rem"] == pytest.approx(156 / 1620)
    assert c["label_argmin_consistent"] == {"n_checked": 1620, "n_mismatch": 0}
    assert c["winner_margin_below_k_sigma"]["n_flagged"] == 229
    assert c["partial_coverage_winners"] is None
    assert res["passed"] is True
    json.dumps(res)


@pytest.mark.skipif(
    not (_RESEARCH / "aggregated.csv").exists(), reason="research aggregated.csv absent"
)
def test_checklist_real_aggregated_csv():
    df = pd.read_csv(_RESEARCH / "aggregated.csv")
    res = stats.koester_checklist(df)
    assert res["schema"] == "aggregated"
    c = res["checks"]
    assert c["label_argmin_consistent"] == {"n_checked": 540, "n_mismatch": 0}
    assert c["partial_coverage_winners"] == 0
    assert c["winner_margin_below_k_sigma"]["n_flagged"] is None
    assert res["passed"] is True


@pytest.mark.skipif(
    not (_RESEARCH / "results.csv").exists(), reason="research results.csv absent"
)
def test_win_shares_and_permutation_on_real_labels():
    df = pd.read_csv(_RESEARCH / "results.csv")
    ws = stats.win_shares(df["best_technique"])
    assert sum(ws.values()) == pytest.approx(1.0)
    assert all(0.0 <= v <= 1.0 for v in ws.values())
    # raw_plus vs raw is a meaningful paired comparison (both never refuse).
    out = stats.paired_permutation_test(
        df["raw_plus_abs_error"], df["raw_abs_error"], n_perm=2000, seed=0,
        alternative="less",
    )
    assert 0.0 < out["p_value"] <= 1.0
    assert out["n_pairs"] == 1620
