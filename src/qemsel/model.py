"""Train and evaluate the QEM-technique recommender model.

Given the experiment results DataFrame (``qemsel.experiment.run_experiment``
schema), this module trains classifiers that predict the winner label from
the ``feat_*`` circuit/backend features, compares them against a
majority-class baseline via GROUP-AWARE stratified cross-validation, refits
the winner on all data, and persists a joblib bundle that
``qemsel.recommend`` can load.

Input schemas accepted
----------------------
* raw ``results.csv`` (one row per (circuit, seed, backend) unit, ``seed``
  column present), and
* seed-averaged ``aggregated.csv`` (one row per (circuit-config, backend),
  ``n_seeds`` column present, no ``seed`` column).
Nothing here reads ``seed``/``n_seeds`` — only the ``feat_*`` columns, the
grouping columns (family, n_qubits, depth), the optional ``backend`` column
and the label column are required.

Evaluation honesty (stats review 2026-07-21; research pass 2026-07-21)
----------------------------------------------------------------------
The features are angle-blind: different seeds of the same
(family, n_qubits, depth) configuration produce byte-identical feature
vectors, and the same circuit appears once per backend differing only in the
2 backend-noise features. Plain row-level CV therefore leaks near-duplicate
rows between train and test and overstates "new circuit" accuracy by ~0.2.
Fixes implemented here:

* CV folds are grouped by (family, n_qubits, depth) via
  ``StratifiedGroupKFold`` (``GroupKFold`` fallback), so all seeds and all
  backend rows of one circuit configuration land in the same fold. The
  grouping actually used is recorded in metrics key ``cv_grouping``.
* Singleton classes (fewer members than any feasible n_splits >= 2) are
  DROPPED from the CV evaluation with an explicit warning and recorded in
  metrics key ``dropped_classes`` — one lone 'zne' win no longer collapses
  the whole evaluation to the cv_folds=0 training-set fallback. The final
  refit-on-all-rows model still sees EVERY row, so dropped classes remain
  predictable at recommendation time.
* A leave-one-family-out (LOFO) evaluation is reported as the honest proxy
  for "recommend for a NEW circuit" (metrics key ``lofo``).
* A leave-one-backend-out (LOBO) evaluation holds out each distinct
  backend STRING (noise-scaled variants like ``FakeManilaV2@x1.5`` are
  separate strings; metrics key ``lobo``). CAUTION (fixer pass
  2026-07-21): when other scales of the SAME device stay in training, a
  LOBO fold measures noise-level INTERPOLATION on a known device — its
  backend features are bracketed by the training siblings — NOT
  generalization to a new environment. Quote LOBO as the interpolation
  number only.
* A leave-one-DEVICE-out (LODO) evaluation holds out ALL noise scales of
  one base device together (metrics key ``lodo``) — the honest headline
  for "recommend under a NEW noise environment": the model has never seen
  the held-out device at any scale. On datasets without scaled variants
  LODO and LOBO coincide.
* Permutation importances are computed on HELD-OUT fold data (mean over CV
  folds), not on the training data of the refit model — training-data
  importances of a memorizing forest are artifacts.

Two winner labels
-----------------
``train_and_eval`` trains on one label column. ``train_and_eval_all`` is the
one-call pipeline entry point: it always trains the accuracy-at-any-cost
``best_technique`` model (``model.joblib`` + ``metrics.json``) and, when the
``best_technique_cost_aware`` column is present with usable rows,
additionally trains the equal-shot-budget model
(``model_cost_aware.joblib`` + ``metrics_cost_aware.json``); it also embeds
the cost-aware metrics into ``metrics.json`` under the ``'cost_aware'`` key
so ``qemsel.report`` can render both label variants side by side.

All randomness is seeded (``random_state=0`` everywhere); no global state.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

import qemsel
from qemsel import stats
from qemsel.features import FEATURE_NAMES, FEATURE_NAMES_BY_VERSION

try:  # sklearn >= 1.6: cv='prefit' was removed in favour of FrozenEstimator
    from sklearn.frozen import FrozenEstimator
except ImportError:  # pragma: no cover - older sklearn fallback
    FrozenEstimator = None

#: The V2 shots-axis column (qemsel.experiment.BASE_SHOTS_COLUMN). Its
#: presence (>= 2 distinct values) enables the leave-one-shot-budget-out
#: ('loso') generalization metric — imported by name to avoid a model ->
#: experiment dependency edge.
_BASE_SHOTS_COLUMN = "base_shots"

#: Candidate (non-baseline) model factories, in tie-break priority order.
#: Fresh estimator per call so no fitted state leaks between uses.
_MODEL_FACTORIES: dict[str, Callable[[], object]] = {
    "random_forest": lambda: RandomForestClassifier(random_state=0),
    "gradient_boosting": lambda: GradientBoostingClassifier(random_state=0),
}

#: Name used for the majority-class baseline in the per_model metrics.
_BASELINE_NAME = "dummy_majority"

#: Rows whose cdr_abs_error is below this are pre-fix CDR artifacts
#: (mitiq's fully-Clifford short-circuit / degenerate constant fit returned
#: the classical simulator value with ~1e-16 error). Post-fix runs record
#: NaN for those cases instead, so this filter only bites legacy CSVs; a
#: genuine shot-noise-limited CDR result can never be this small.
_CDR_DEGENERATE_TOL: float = 1e-12

#: The cost-aware winner column written by qemsel.experiment; presence of
#: this column (with usable rows) makes train_and_eval_all train a second
#: model bundle.
COST_AWARE_LABEL = "best_technique_cost_aware"

#: V2 (INTERFACES.md section V2; builder-model / B7 implements): the
#: significance-aware label column DERIVED at training time by
#: ``derive_significant_label`` (never written by the experiment — the CSVs
#: stay untouched). Its classes are the technique names plus 'tie'.
SIGNIFICANT_LABEL = "best_technique_significant"

#: V2: class name assigned when no technique beats the runner-up by the
#: required shot-noise margin.
TIE_CLASS = "tie"

#: Backend names may carry a noise-scale suffix '<BaseName>@x<scale>'
#: (grammar of qemsel.backends.parse_backend_name / qemsel.report). Used to
#: pool all scales of one device for the leave-one-device-out evaluation.
_NOISE_SCALE_RE = re.compile(r"^(?P<base>.+)@x(?P<scale>\d+(?:\.\d+)?)$")


def _base_device(backend: str) -> str:
    """Base device name of a backend string ('FakeManilaV2@x1.5' -> 'FakeManilaV2')."""
    m = _NOISE_SCALE_RE.match(str(backend))
    return m.group("base") if m else str(backend)


#: A class must have at least this many members to take part in CV — a
#: singleton can never appear in both a train and a test fold, so it cannot
#: be cross-validated under ANY n_splits >= 2. Classes below the floor are
#: dropped from the CV evaluation (recorded in metrics['dropped_classes'])
#: instead of collapsing the whole evaluation to the cv_folds=0 fallback.
_MIN_CLASS_MEMBERS_FOR_CV = 2


def _feature_columns(feature_version: int = 1) -> list[str]:
    """The exact feature-matrix column names, in canonical order.

    ``feature_version`` (V2) selects the feature set from
    ``qemsel.features.FEATURE_NAMES_BY_VERSION``: version 1 (default) is the
    frozen 10-feature V1 vector (byte-identical), version 2 the 15-feature
    shots-aware vector. Unknown versions raise ValueError.
    """
    names = FEATURE_NAMES_BY_VERSION.get(feature_version)
    if names is None:
        raise ValueError(
            f"unknown feature_version {feature_version!r}; known: "
            f"{sorted(FEATURE_NAMES_BY_VERSION)}"
        )
    return ["feat_" + name for name in names]


def _score(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    """(accuracy, macro-F1) for a prediction vector."""
    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    return acc, f1


def _oof_from_splits(
    estimator: object,
    X: pd.DataFrame,
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, list[float]]:
    """Out-of-fold predictions plus per-fold accuracies for one estimator."""
    oof = np.empty(y.shape[0], dtype=object)
    fold_accuracies: list[float] = []
    for train_idx, test_idx in splits:
        model = clone(estimator)
        model.fit(X.iloc[train_idx], y[train_idx])
        pred = model.predict(X.iloc[test_idx])
        oof[test_idx] = pred
        fold_accuracies.append(float(accuracy_score(y[test_idx], pred)))
    return oof, fold_accuracies


def _fit_all_predictions(
    estimator: object, X: pd.DataFrame, y: np.ndarray
) -> np.ndarray:
    """Degenerate-data path: fit on ALL rows and predict the training set."""
    model = clone(estimator)
    model.fit(X, y)
    return np.asarray(model.predict(X), dtype=object)


def _std(fold_accuracies: list[float]) -> float:
    """Sample std (ddof=1, the conventional choice for 2-5 folds); 0.0 for
    a single value."""
    if len(fold_accuracies) < 2:
        return 0.0
    return float(np.std(fold_accuracies, ddof=1))


def _grouped_splits(
    X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, n_splits: int
) -> tuple[list[tuple[np.ndarray, np.ndarray]], str]:
    """Group-aware CV splits; returns (splits, grouping_mode).

    Tries StratifiedGroupKFold first (keeps class balance across folds while
    never splitting a group); falls back to plain GroupKFold when the
    stratification constraints cannot be satisfied (raises, or yields an
    empty train/test fold).
    """
    try:
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=0)
        splits = list(cv.split(X, y, groups))
        if all(len(tr) > 0 and len(te) > 0 for tr, te in splits):
            return splits, "stratified_group"
    except ValueError:
        pass
    cv = GroupKFold(n_splits=n_splits)
    return list(cv.split(X, y, groups)), "group"


def _leave_one_group_out(
    estimator_factory: Callable[[], object],
    X: pd.DataFrame,
    y: np.ndarray,
    holdout: np.ndarray,
) -> tuple[float, float, dict[str, float], dict[str, float], int] | None:
    """Leave-one-<group>-out evaluation of the chosen model family.

    For every distinct value in ``holdout`` (circuit family, or backend
    name), fit on all OTHER rows and predict the held-out rows. Returns
    (pooled accuracy, pooled macro-F1, per-group accuracy, per-group
    macro-F1, number of groups), or None when fewer than 2 distinct values
    exist. Uses ALL usable rows — including rows of classes dropped from CV
    (an honest evaluation counts them; a model that never saw the class
    simply gets those rows wrong).
    """
    values = sorted(set(holdout))
    if len(values) < 2:
        return None
    oof = np.empty(y.shape[0], dtype=object)
    per_accuracy: dict[str, float] = {}
    per_f1: dict[str, float] = {}
    for value in values:
        test_mask = holdout == value
        model = clone(estimator_factory())
        model.fit(X.loc[~test_mask], y[~test_mask])
        pred = model.predict(X.loc[test_mask])
        oof[test_mask] = pred
        acc_g, f1_g = _score(y[test_mask], pred)
        per_accuracy[value] = acc_g
        per_f1[value] = f1_g
    acc, f1 = _score(y, oof)
    return acc, f1, per_accuracy, per_f1, len(values)


# ==========================================================================
# V2 helpers (builder-model / B7): calibration, held-out probabilities,
# multiclass Brier, and the shot-noise sigma used by the significance label.
# All are private and only exercised when a V2 keyword flag is set, so the
# default (V1) training path never touches them.
# ==========================================================================


def _proba_matrix(model: object, X: pd.DataFrame, labels: list[str]) -> np.ndarray:
    """``predict_proba`` reindexed to ``labels`` column order.

    Columns for classes the fitted model never saw are filled with 0.0, so
    the matrix is comparable across folds/models regardless of each fit's
    ``classes_`` subset.
    """
    proba = np.asarray(model.predict_proba(X), dtype=float)
    class_index = {str(c): i for i, c in enumerate(model.classes_)}
    out = np.zeros((proba.shape[0], len(labels)), dtype=float)
    for j, lab in enumerate(labels):
        src = class_index.get(str(lab))
        if src is not None:
            out[:, j] = proba[:, src]
    return out


def _multiclass_brier(
    proba: np.ndarray, y_true: np.ndarray, labels: list[str]
) -> float:
    """Mean multiclass Brier score sum_c (p_ic - [y_i==c])**2 over samples."""
    pos = {str(lab): j for j, lab in enumerate(labels)}
    onehot = np.zeros_like(proba)
    for i, yt in enumerate(y_true):
        onehot[i, pos[str(yt)]] = 1.0
    return float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))


def _fit_calibrated(
    factory: Callable[[], object],
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
) -> object:
    """Sigmoid-calibrate ``factory()`` on (X, y), fitted and returned.

    Uses GROUP-AWARE cross-fitted calibration (the same leakage argument as
    the main CV: seed/backend duplicates of one circuit config must not span
    the calibration train/holdout boundary — never plain KFold). Falls back
    to prefit calibration (via ``FrozenEstimator``) when the class/group
    structure cannot support >= 2 grouped folds (e.g. a singleton class that
    stays in the refit model), so calibrate=True always yields a working
    predict_proba wrapper that knows every class in ``y``.
    """
    counts = pd.Series(y).value_counts()
    smallest = int(counts.min()) if len(counts) else 0
    n_groups = len(set(groups))
    k = min(n_splits, smallest, n_groups)
    Xr = X.reset_index(drop=True)
    if k >= 2 and len(set(y)) >= 2:
        inner, _ = _grouped_splits(Xr, y, groups, k)
        try:
            cal = CalibratedClassifierCV(factory(), method="sigmoid", cv=inner)
            cal.fit(Xr, y)
            return cal
        except (ValueError, IndexError):  # pragma: no cover - degenerate folds
            pass
    base = factory()
    base.fit(Xr, y)
    if FrozenEstimator is None:  # pragma: no cover - sklearn < 1.6 legacy API
        cal = CalibratedClassifierCV(base, method="sigmoid", cv="prefit")
        cal.fit(Xr, y)
        return cal
    # sklearn >= 1.6 prefit replacement. CalibratedClassifierCV still
    # cross-validates the calibration data even over a FrozenEstimator, so its
    # cv must not exceed the smallest class count: the default cv=5 crashes
    # whenever a class has < 5 members (rare 'raw'/'raw_plus' winners, or any
    # small training fold). Clamp it; when even 2 folds are impossible, degrade
    # honestly to the uncalibrated base (still exposes predict_proba over every
    # class in y) rather than raising and sinking the whole calibrated run.
    cv_fallback = min(5, smallest)
    if cv_fallback >= 2:
        try:
            cal = CalibratedClassifierCV(
                FrozenEstimator(base), method="sigmoid", cv=cv_fallback
            )
            cal.fit(Xr, y)
            return cal
        except (ValueError, IndexError):  # pragma: no cover - degenerate folds
            pass
    return base


def _oof_proba(
    factory: Callable[[], object],
    X: pd.DataFrame,
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    labels: list[str],
    *,
    groups: np.ndarray | None = None,
    n_splits: int = 0,
    calibrated: bool = False,
) -> np.ndarray:
    """Out-of-fold predicted probabilities of one model over the CV splits.

    ``calibrated`` fits a group-cross-fitted sigmoid calibration on each
    training fold (via :func:`_fit_calibrated`) before predicting the
    held-out fold, so the returned probabilities are honestly held-out.
    """
    proba = np.zeros((y.shape[0], len(labels)), dtype=float)
    for train_idx, test_idx in splits:
        X_tr = X.iloc[train_idx].reset_index(drop=True)
        if calibrated:
            g_tr = groups[train_idx] if groups is not None else np.zeros(len(train_idx))
            model = _fit_calibrated(factory, X_tr, y[train_idx], g_tr, n_splits)
        else:
            model = factory()
            model.fit(X_tr, y[train_idx])
        proba[test_idx] = _proba_matrix(model, X.iloc[test_idx], labels)
    return proba


def _safe_estimator_sigma(technique: str, value: float, base_shots: float) -> float:
    """Per-technique estimator sigma, guarded to +inf on invalid inputs.

    Wraps ``qemsel.mitigation.estimator_sigma`` (the single source of truth
    for each technique's variance model — Richardson coefficients included).
    Returns +inf (forcing a conservative 'tie') when the shot count or value
    is missing/non-positive, so a row with incomplete sigma inputs never
    claims a significant winner it cannot support. Unknown technique names
    (possible in synthetic dfs) fall back to the single-execution binomial
    sigma at ``base_shots`` — with the same |value| > 1 -> +inf overshoot
    rule (an unphysical estimate flags variance blow-up; a zero sigma there
    would make ANY margin 'significant', the anti-conservative direction).
    """
    try:
        v = float(value)
        b = float(base_shots)
    except (TypeError, ValueError):
        return math.inf
    if math.isnan(v) or math.isnan(b) or b <= 0:
        return math.inf
    try:
        from qemsel.mitigation import estimator_sigma

        return float(estimator_sigma(technique, v, b))
    except ValueError:
        # unknown technique name: conservative single-execution fallback
        if abs(v) > 1.0:
            return math.inf
        return float(math.sqrt((1.0 - v * v) / b))
    except (TypeError, ImportError):  # pragma: no cover - defensive
        return math.inf


def derive_significant_label(
    df: pd.DataFrame,
    k_sigma: float = 2.0,
    *,
    techniques: list[str] | None = None,
) -> pd.Series:
    """Significance-aware labels (V2; sigma model fixed 2026-07-23).

    Per row: winner = argmin of the non-NaN technique errors, runner-up =
    second smallest. The label is the winner's name ONLY when
    ``runner_err - winner_err >= k_sigma * sigma_combined``; otherwise
    ``TIE_CLASS``; ``''`` (empty string, matching the experiment's all-failed
    convention) when fewer than 1 technique produced a value; the winner's
    name outright when exactly 1 did (no runner-up to be confused with).

    ``sigma_combined = sqrt(sigma_w**2 + sigma_r**2)`` with PER-TECHNIQUE
    ESTIMATOR sigmas via ``qemsel.mitigation.estimator_sigma`` (the single
    source of truth also feeding the Angle-3 boundary's variance side, so
    the learned labels and the analytic boundary share one noise model):

    * per-seed results.csv schema (``<tech>_value`` + ``<tech>_shots``
      columns): ``estimator_sigma(tech, value, base)`` with ``base =
      <tech>_shots / SHOT_MULTIPLIER_V2[tech]`` (the shots column is the
      consumed-budget LEDGER, not pooled shots — the pre-fix
      ``sigma_shot(value, <tech>_shots)`` treated 3x/11x ledgers as pooled
      averaging and understated sigma 2.2x (zne_fr), 7.5x (zne), >= 3.3x
      (cdr), >= 1.7x (rem); ~10% of the research labels flipped back to
      'tie' under the corrected zne sigma alone). Overshoot values
      (|value| > 1) get sigma = +inf -> 'tie', never the sigma = 0 the old
      clamping produced.
    * aggregated schema (``<tech>_mean_abs_error`` + ``<tech>_n_seeds`` +
      ``base_shots``; no value columns): the true expectation value that
      sets the binomial variance is NOT recoverable from
      ``1 - mean_abs_error`` (that proxy was ANTI-conservative: winner
      sigmas understated up to 55x on the smoke data, 21% label flips), so
      the route now uses the worst-case variance term (v = 0) with the
      per-technique amplification factor and seed pooling:
      ``estimator_sigma(tech, 0.0, n_seeds * base_shots)``. This is a
      deliberately CONSERVATIVE upper bound on sigma — expect more ties
      than the per-seed route; a loud note recommends deriving the
      significant label from per-seed results.csv, which this function
      prints on the aggregated route.

    Args:
        df: either experiment schema (auto-detected).
        k_sigma: required margin in combined-sigma units (default 2.0 =
            qemsel.stats.DEFAULT_K_SIGMA; scripts/train_model.py exposes
            ``--k-sigma``).
        techniques: restrict to these technique columns (None =
            auto-detect from the df's error columns).

    Returns:
        pd.Series aligned to ``df.index``, dtype object, values in
        {technique names} | {TIE_CLASS, ''}. Callers assign it to
        ``df[SIGNIFICANT_LABEL]`` and train via
        ``train_and_eval(df, out, label_column=SIGNIFICANT_LABEL,
        bundle_filename='model_significant.joblib',
        metrics_filename='metrics_significant.json')``.

    Raises:
        ValueError: schema lacks the columns the sigma route needs.
    """
    cols = list(df.columns)

    # ---- detect schema + technique error columns -------------------------
    # NB: '<t>_mean_abs_error' also endswith '_abs_error' — exclude it from
    # the per-seed detection so 'raw_mean' is never mistaken for a technique.
    per_seed_techs = [
        c[: -len("_abs_error")]
        for c in cols
        if c.endswith("_abs_error") and not c.endswith("_mean_abs_error")
    ]
    # Per-seed route needs value + shots companions for the analytic sigma.
    per_seed_techs = [
        t
        for t in per_seed_techs
        if f"{t}_value" in cols and f"{t}_shots" in cols
    ]
    agg_techs = [c[: -len("_mean_abs_error")] for c in cols if c.endswith("_mean_abs_error")]

    if per_seed_techs:
        route = "per_seed"
        techs = per_seed_techs
    elif agg_techs:
        route = "aggregated"
        techs = agg_techs
        if _BASE_SHOTS_COLUMN not in cols:
            raise ValueError(
                "aggregated schema needs a 'base_shots' column for the "
                "significance sigma route (V2 aggregated.csv); found "
                f"mean-abs-error columns for {agg_techs} but no base_shots"
            )
        missing_ns = [t for t in techs if f"{t}_n_seeds" not in cols]
        if missing_ns:
            raise ValueError(
                f"aggregated schema missing per-technique n_seeds columns: "
                f"{[t + '_n_seeds' for t in missing_ns]}"
            )
    else:
        raise ValueError(
            "df has no technique error columns ('<tech>_abs_error' or "
            "'<tech>_mean_abs_error') to derive a significance label from"
        )

    if techniques is not None:
        keep = [t for t in techs if t in set(techniques)]
        if not keep:
            raise ValueError(
                f"none of techniques={list(techniques)} present among the "
                f"df's error columns {techs}"
            )
        techs = keep

    n = len(df)
    err = np.column_stack(
        [
            pd.to_numeric(
                df[f"{t}_abs_error" if route == "per_seed" else f"{t}_mean_abs_error"],
                errors="coerce",
            ).to_numpy(dtype=float)
            for t in techs
        ]
    ) if n else np.empty((0, len(techs)))

    if route == "per_seed":
        val = np.column_stack(
            [pd.to_numeric(df[f"{t}_value"], errors="coerce").to_numpy(float) for t in techs]
        ) if n else np.empty((0, len(techs)))
        sht = np.column_stack(
            [pd.to_numeric(df[f"{t}_shots"], errors="coerce").to_numpy(float) for t in techs]
        ) if n else np.empty((0, len(techs)))
    else:
        # Aggregated proxy (documented, conservative): value ~ 1 - mean_error,
        # shots ~ n_seeds * shots_consumed(tech, base_shots). Deferred import
        # keeps model.py loadable without mitiq for every other code path.
        from qemsel.mitigation import shots_consumed

        base = pd.to_numeric(df[_BASE_SHOTS_COLUMN], errors="coerce").to_numpy(float) if n else np.empty(0)
        nseeds = np.column_stack(
            [pd.to_numeric(df[f"{t}_n_seeds"], errors="coerce").to_numpy(float) for t in techs]
        ) if n else np.empty((0, len(techs)))
        val = 1.0 - err
        mult = np.array([shots_consumed(t, 1) for t in techs], dtype=float)
        sht = nseeds * base[:, None] * mult[None, :]

    # Per-technique shot multiplier: `sht` above is the CONSUMED-budget ledger
    # (per_seed: <tech>_shots = mult x base; aggregated: n_seeds x base x mult).
    # `estimator_sigma` wants the per-execution BASE shots B (it applies each
    # technique's variance amplification internally), so divide the ledger back
    # out by `shots_consumed(t, 1)`. For the aggregated route this leaves
    # n_seeds x base — the correct averaged-estimate shot count (Var ~ 1/B).
    from qemsel.mitigation import shots_consumed as _shots_consumed
    _mult = np.array([max(_shots_consumed(t, 1), 1) for t in techs], dtype=float)

    labels_out: list[str] = []
    for i in range(n):
        row_err = err[i]
        valid = ~np.isnan(row_err)
        n_valid = int(valid.sum())
        if n_valid == 0:
            labels_out.append("")
            continue
        valid_idx = np.flatnonzero(valid)
        # argmin (winner) then second smallest (runner-up); stable order =
        # column-appearance order, matching the experiment's argmin tie-break.
        order = valid_idx[np.argsort(row_err[valid_idx], kind="stable")]
        w = int(order[0])
        if n_valid == 1:
            labels_out.append(techs[w])
            continue
        r = int(order[1])
        margin = float(row_err[r] - row_err[w])
        sigma_w = _safe_estimator_sigma(techs[w], val[i, w], sht[i, w] / _mult[w])
        sigma_r = _safe_estimator_sigma(techs[r], val[i, r], sht[i, r] / _mult[r])
        sigma_comb = math.sqrt(sigma_w ** 2 + sigma_r ** 2)
        if margin >= k_sigma * sigma_comb:
            labels_out.append(techs[w])
        else:
            labels_out.append(TIE_CLASS)

    return pd.Series(labels_out, index=df.index, dtype=object)


def train_and_eval(
    df: pd.DataFrame,
    out_dir: Path,
    label_column: str = "best_technique",
    *,
    bundle_filename: str = "model.joblib",
    metrics_filename: str = "metrics.json",
    feature_version: int = 1,
    calibrate: bool = False,
    abstain_threshold: float | None = None,
    extended_stats: bool = False,
) -> dict:
    """Train classifiers predicting the winner label from circuit features.

    Args:
        df: experiment results DataFrame — either the raw
            ``qemsel.experiment.run_experiment`` row schema or the
            seed-averaged aggregated schema (``n_seeds`` column, no ``seed``
            column). Must contain all ``feat_*`` columns,
            ``family``/``n_qubits``/``depth`` for fold grouping, and the
            label column. A ``backend`` column additionally enables the
            leave-one-backend-out evaluation.
        out_dir: output directory; created if missing.
        label_column: which winner column to train on —
            ``'best_technique'`` (accuracy-at-any-shot-cost, default) or
            ``'best_technique_cost_aware'`` (winner at an equal shot
            budget; restores 'raw' as a reachable class).
        bundle_filename: filename (inside out_dir) for the joblib bundle
            (keyword-only; default 'model.joblib' — train_and_eval_all
            passes 'model_cost_aware.joblib' for the second label).
        metrics_filename: filename (inside out_dir) for the metrics JSON
            (keyword-only; default 'metrics.json').
        feature_version: (V2, keyword-only, default 1) feature set to train
            on — selects ``qemsel.features.FEATURE_NAMES_BY_VERSION
            [feature_version]`` as the feat_* matrix columns (df must carry
            them; version 2 dfs come from feature_version-2 experiment
            runs). The bundle records it as 'feature_version'.
        calibrate: (V2, keyword-only, default False) wrap the refit best
            model in ``sklearn.calibration.CalibratedClassifierCV``
            (method='sigmoid', cv=grouped folds reusing THIS function's
            group definition — never plain KFold, the leakage argument is
            the same as for CV) so recommend's probabilities are honest
            enough to threshold. Bundle records 'calibrated': True and the
            'model' entry IS the calibrated wrapper (predict_proba
            preserved). Metrics gain a 'calibration' dict (at least
            {'method', 'brier_before', 'brier_after'} on held-out folds).
        abstain_threshold: (V2, keyword-only, default None) when set (in
            (0, 1)), stored VERBATIM in the bundle as 'abstain_threshold'
            for qemsel.recommend to enforce (max predicted probability
            below it => 'abstain'); combine with calibrate=True or the
            threshold is on uncalibrated scores (allowed but the metrics
            note must say so). Metrics gain 'abstain_threshold' +
            'abstain_rate_cv' (fraction of CV out-of-fold predictions that
            would have abstained).
        extended_stats: (V2, keyword-only, default False) when True, each
            per_model entry additionally carries 'fold_accuracies'
            (list[float]) and the metrics dict gains 'fold_summary' (via
            ``qemsel.stats.summarize_folds``) — kept behind a flag so
            default-path metrics.json stays byte-identical.

    V2 additions to the metrics dict (beyond the exact V1 keys below —
    present ONLY when triggered, so V1-shaped runs stay byte-identical):
        'loso' — leave-one-shot-budget-out holdout (same shape as 'lobo'
            with 'per_budget_accuracy'/'per_budget_macro_f1'/'n_budgets'),
            computed automatically whenever the df carries a ``base_shots``
            column with >= 2 distinct values (the shots-generalization
            number the Angle 3 overlay quotes);
        'feature_version' — when feature_version != 1;
        'calibration', 'abstain_threshold', 'abstain_rate_cv',
        'fold_summary' — per the flags above.

    Behaviour contract:
    * Rows with label == '' / NaN, or with any NaN feature, are dropped
      before training. Rows with ``cdr_abs_error`` < 1e-12 (pre-fix CDR
      classical-simulation artifacts) are also dropped — their labels
      measure classical simulability, not mitigation quality.
    * Feature matrix X: columns ``['feat_' + n for n in features.FEATURE_NAMES]``
      in exactly that order. Target y: the label column.
    * Models compared (fixed seeds): RandomForestClassifier(random_state=0)
      and GradientBoostingClassifier(random_state=0).
    * Cross-validation is GROUP-AWARE: folds never split a
      (family, n_qubits, depth) group, because different seeds/backends of
      the same configuration share (near-)identical feature vectors and
      row-level CV would leak them between train and test.
      Classes with < 2 members are DROPPED from the CV evaluation (warned,
      and recorded in metrics['dropped_classes']) — they cannot straddle a
      train/test boundary under any n_splits, and one singleton must not
      disable the whole evaluation. n_splits = min(5,
      smallest-KEPT-class-count, number-of-kept-groups) with
      StratifiedGroupKFold(shuffle=True, random_state=0), GroupKFold
      fallback (metrics['cv_grouping'] records which ran). If even after
      dropping n_splits < 2 OR fewer than 2 classes remain (single-class
      "CV" would report a meaningless perfect score), skip CV, fit on all
      rows, evaluate on the training set, and set 'cv_folds': 0 with
      dropped_classes == [] (the fallback evaluates every row).
    * Metrics per model: CV-aggregated accuracy and macro-F1 (out-of-fold
      predictions over the KEPT rows, so the confusion matrix is honest);
      each per_model entry additionally carries 'accuracy_std' (ddof=1 std
      of per-fold accuracies).
    * Baseline: majority-class classifier accuracy on the same folds.
    * The model with the higher macro-F1 is refit on ALL usable rows —
      including rows of dropped classes — and saved to
      ``out_dir / bundle_filename`` via joblib.dump as a BUNDLE dict:
      {'model': <fitted sklearn estimator>,
       'feature_names': <list[str], the feat_* column order>,
       'classes': <list[str]>,
       'model_name': <str>,
       'label_column': <str>,
       'qemsel_version': <str>}.
      recommend.py depends on this exact bundle shape.
    * Feature importances: permutation importance on HELD-OUT fold data
      (n_repeats=10, random_state=0, averaged over the CV folds of the best
      model). In the cv_folds=0 fallback they come from the training set
      and are flagged as unreliable via 'feature_importances_note'.
    * Leave-one-family-out generalization ('lofo' key),
      leave-one-backend-out ('lobo' key) and leave-one-DEVICE-out
      ('lodo' key) are evaluated for the best model on ALL usable rows
      whenever >= 2 families / backend strings / base devices are present
      (None otherwise; lobo/lodo are also None when df has no 'backend'
      column). LOFO is the honest headline for "generalizes to a NEW
      circuit family". LODO (all '@x<scale>' siblings of one device held
      out together) is the honest headline for "a NEW noise environment";
      LOBO holds out single backend STRINGS, so when scale-siblings of the
      held-out backend remain in training it measures noise-level
      INTERPOLATION on a known device — do not quote it as new-environment
      generalization (fixer pass 2026-07-21).
    * Full metrics dict also written to ``out_dir / metrics_filename``.

    Returns:
        metrics dict with EXACTLY these keys:
            'best_model_name'     (str)
            'accuracy'            (float, best model, CV out-of-fold)
            'macro_f1'            (float, best model, CV out-of-fold)
            'baseline_accuracy'   (float, majority class)
            'labels'              (list[str], sorted class labels over ALL
                                   usable rows; row/col order of the
                                   confusion matrix)
            'confusion_matrix'    (list[list[int]]; rows of dropped classes
                                   are all-zero — those rows were not in CV)
            'feature_importances' (dict[str, float], feat_* name -> importance
                                   of the best model, held-out folds)
            'feature_importances_note' (str, provenance of the importances)
            'per_model'           (dict[str, {'accuracy': float,
                                   'accuracy_std': float, 'macro_f1': float}]
                                   for both models plus the dummy baseline)
            'n_samples'           (int, usable rows == refit-model rows)
            'cv_n_samples'        (int, rows actually cross-validated;
                                   == n_samples unless classes were dropped)
            'cv_folds'            (int, 0 = degenerate fallback used)
            'cv_grouping'         (str: 'stratified_group' | 'group' |
                                   'none (degenerate)')
            'dropped_classes'     (list[str], classes excluded from CV for
                                   having < 2 members; [] if none or if the
                                   cv_folds=0 fallback ran)
            'label_column'        (str, the label trained on)
            'lofo'                (dict | None: 'accuracy', 'macro_f1',
                                   'per_family_accuracy',
                                   'per_family_macro_f1', 'n_families')
            'lobo'                (dict | None: 'accuracy', 'macro_f1',
                                   'per_backend_accuracy',
                                   'per_backend_macro_f1', 'n_backends')
            'lodo'                (dict | None: 'accuracy', 'macro_f1',
                                   'per_device_accuracy',
                                   'per_device_macro_f1', 'n_devices')

    Raises:
        ValueError: if df lacks required columns or has zero usable rows.
    """
    if abstain_threshold is not None and not (0.0 < abstain_threshold < 1.0):
        raise ValueError(
            f"abstain_threshold must be in (0, 1); got {abstain_threshold!r}"
        )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # feature_version selects the feat_* matrix columns (V2). Invalid
    # versions raise here via _feature_columns.
    feat_cols = _feature_columns(feature_version)
    group_cols = ["family", "n_qubits", "depth"]
    required = feat_cols + group_cols + [label_column]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"df lacks required columns: {missing}")

    # ---- drop pre-fix CDR classical-simulation artifact rows -------------
    if "cdr_abs_error" in df.columns:
        cdr_err = pd.to_numeric(df["cdr_abs_error"], errors="coerce")
        degenerate = cdr_err.notna() & (cdr_err < _CDR_DEGENERATE_TOL)
        n_degenerate = int(degenerate.sum())
        if n_degenerate:
            print(
                f"[qemsel.model] dropping {n_degenerate} row(s) with "
                f"cdr_abs_error < {_CDR_DEGENERATE_TOL} — pre-fix CDR "
                "classical-simulation artifacts (labels untrustworthy)"
            )
            df = df.loc[~degenerate]

    # ---- drop unusable rows (NaN/empty winner, NaN feature) --------------
    winner = df[label_column]
    usable = winner.notna() & (winner.astype(str).str.strip() != "")
    usable &= ~df[feat_cols].isna().any(axis=1)
    work = df.loc[usable]
    if len(work) == 0:
        raise ValueError(
            f"zero usable rows: every row has a NaN/empty {label_column} "
            "or a NaN feature"
        )

    # Named DataFrame (not ndarray) so the fitted model carries feature names:
    # sklearn then VALIDATES column names/order when recommend.py predicts
    # with its named row (and no "fitted without feature names" warning).
    X = work[feat_cols].astype(float).reset_index(drop=True)
    y = work[label_column].astype(str).to_numpy()
    labels = sorted(set(y))
    # Fold groups: one group per circuit configuration — its seeds and
    # backend rows share (near-)identical features and must not straddle
    # a train/test boundary. (Works for both the raw results.csv schema and
    # the seed-averaged aggregated.csv schema: no 'seed' column is read.)
    groups = (
        work["family"].astype(str)
        + "_q" + work["n_qubits"].astype(str)
        + "_d" + work["depth"].astype(str)
    ).to_numpy()
    families = work["family"].astype(str).to_numpy()
    backends = (
        work["backend"].astype(str).to_numpy()
        if "backend" in work.columns
        else None
    )
    # V2: leave-one-shot-budget-out holdout, active only when the df carries
    # the base_shots column (list-mode / shots-axis runs) with >= 2 budgets.
    budgets = (
        work[_BASE_SHOTS_COLUMN].astype(str).to_numpy()
        if _BASE_SHOTS_COLUMN in work.columns
        and work[_BASE_SHOTS_COLUMN].nunique() >= 2
        else None
    )

    # ---- class balance report (honest, always printed) -------------------
    counts = pd.Series(y).value_counts()
    balance = {lab: int(counts[lab]) for lab in labels}
    print(
        "[qemsel.model] label=%s, n_samples=%d, class balance: %s"
        % (label_column, len(y), ", ".join(f"{k}={v}" for k, v in balance.items()))
    )

    # ---- drop singleton classes from CV (NOT from the refit) --------------
    # A class with < 2 members can never sit in both a train and a test
    # fold, so no n_splits >= 2 can evaluate it; previously one lone 'zne'
    # win collapsed the WHOLE evaluation to the cv_folds=0 training-set
    # fallback. Now the singleton rows are excluded from CV only — the
    # refit-on-all model below still trains on every row.
    dropped_classes = sorted(
        lab for lab in labels if counts[lab] < _MIN_CLASS_MEMBERS_FOR_CV
    )
    kept_mask = ~np.isin(y, dropped_classes)
    X_cv = X.loc[kept_mask].reset_index(drop=True)
    y_cv = y[kept_mask]
    groups_cv = groups[kept_mask]

    # CV additionally needs >= 2 KEPT classes: a single-class "CV" would
    # report a meaningless perfect score (and GradientBoosting cannot even
    # fit one class) — the honest cv_folds=0 fallback is better there.
    if len(y_cv) > 0 and len(set(y_cv)) >= 2:
        counts_cv = pd.Series(y_cv).value_counts()
        smallest = int(counts_cv.min())
        n_groups = len(set(groups_cv))
        n_splits = min(5, smallest, n_groups)
    else:
        n_splits = 0

    candidates: dict[str, object] = {
        name: factory() for name, factory in _MODEL_FACTORIES.items()
    }
    candidates[_BASELINE_NAME] = DummyClassifier(strategy="most_frequent")

    per_model: dict[str, dict[str, float]] = {}
    oof_predictions: dict[str, np.ndarray] = {}
    fold_accs_by_model: dict[str, list[float]] = {}
    splits: list[tuple[np.ndarray, np.ndarray]] = []

    if n_splits >= 2:
        cv_folds = n_splits
        if dropped_classes:
            print(
                "[qemsel.model] WARNING: class(es) "
                f"{dropped_classes} have < {_MIN_CLASS_MEMBERS_FOR_CV} "
                "members and are DROPPED from the CV evaluation "
                f"({int((~kept_mask).sum())} row(s)); they remain in the "
                "final refit model and in LOFO/LOBO. Recorded in "
                "metrics['dropped_classes']."
            )
        if n_splits < 5:
            print(
                f"[qemsel.model] smallest CV class has {smallest} member(s), "
                f"{n_groups} feature groups -> {n_splits}-fold grouped CV"
            )
        splits, cv_grouping = _grouped_splits(X_cv, y_cv, groups_cv, n_splits)
        for name, estimator in candidates.items():
            oof, fold_accs = _oof_from_splits(estimator, X_cv, y_cv, splits)
            acc, f1 = _score(y_cv, oof)
            per_model[name] = {
                "accuracy": acc,
                "accuracy_std": _std(fold_accs),
                "macro_f1": f1,
            }
            oof_predictions[name] = oof
            fold_accs_by_model[name] = fold_accs
        eval_X, eval_y = X_cv, y_cv
    else:
        cv_folds = 0
        cv_grouping = "none (degenerate)"
        # Nothing was excluded from this evaluation — every row is scored
        # (on the training set), so dropped_classes is [] by definition.
        dropped_classes = []
        print(
            "[qemsel.model] WARNING: even after excluding singleton classes "
            "there are < 2 usable CV classes/groups — too little data for "
            "grouped cross-validation. Fitting on ALL rows and evaluating "
            "on the training set; these metrics are optimistic. cv_folds=0 "
            "flags this."
        )
        for name, estimator in candidates.items():
            pred = _fit_all_predictions(estimator, X, y)
            acc, f1 = _score(y, pred)
            per_model[name] = {
                "accuracy": acc,
                "accuracy_std": 0.0,
                "macro_f1": f1,
            }
            oof_predictions[name] = pred
            fold_accs_by_model[name] = [acc]
        eval_X, eval_y = X, y

    # ---- pick best NON-dummy model by macro-F1 (ties -> first in order) --
    best_model_name = max(
        _MODEL_FACTORIES, key=lambda name: per_model[name]["macro_f1"]
    )
    best_pred = oof_predictions[best_model_name]
    cm = confusion_matrix(eval_y, best_pred, labels=labels)

    # ---- leave-one-family-out / leave-one-backend-out (honest headlines) --
    # Evaluated on ALL usable rows (dropped-class rows included): these are
    # independent of CV feasibility and answer "NEW circuit family" / "NEW
    # noise environment" for the paper.
    lofo = None
    lofo_result = _leave_one_group_out(
        _MODEL_FACTORIES[best_model_name], X, y, families
    )
    if lofo_result is not None:
        acc, f1, per_acc, per_f1, n_values = lofo_result
        lofo = {
            "accuracy": acc,
            "macro_f1": f1,
            "per_family_accuracy": per_acc,
            "per_family_macro_f1": per_f1,
            "n_families": n_values,
        }
        print(
            "[qemsel.model] leave-one-family-out: accuracy "
            f"{lofo['accuracy']:.3f}, macro-F1 {lofo['macro_f1']:.3f} "
            f"({lofo['n_families']} families)"
        )

    lobo = None
    lodo = None
    if backends is not None:
        lobo_result = _leave_one_group_out(
            _MODEL_FACTORIES[best_model_name], X, y, backends
        )
        if lobo_result is not None:
            acc, f1, per_acc, per_f1, n_values = lobo_result
            lobo = {
                "accuracy": acc,
                "macro_f1": f1,
                "per_backend_accuracy": per_acc,
                "per_backend_macro_f1": per_f1,
                "n_backends": n_values,
            }
            print(
                "[qemsel.model] leave-one-backend-out: accuracy "
                f"{lobo['accuracy']:.3f}, macro-F1 {lobo['macro_f1']:.3f} "
                f"({lobo['n_backends']} backends; scale-siblings may remain "
                "in training -> read as noise-level interpolation)"
            )
        # Leave-one-DEVICE-out: all '@x<scale>' siblings of one base device
        # are held out together — the honest "NEW noise environment" number
        # (a LOBO fold whose device stays in training at other scales only
        # measures interpolation; fixer pass 2026-07-21).
        devices = np.array([_base_device(b) for b in backends])
        lodo_result = _leave_one_group_out(
            _MODEL_FACTORIES[best_model_name], X, y, devices
        )
        if lodo_result is not None:
            acc, f1, per_acc, per_f1, n_values = lodo_result
            lodo = {
                "accuracy": acc,
                "macro_f1": f1,
                "per_device_accuracy": per_acc,
                "per_device_macro_f1": per_f1,
                "n_devices": n_values,
            }
            print(
                "[qemsel.model] leave-one-device-out: accuracy "
                f"{lodo['accuracy']:.3f}, macro-F1 {lodo['macro_f1']:.3f} "
                f"({lodo['n_devices']} devices; headline 'new noise "
                "environment' number)"
            )

    # ---- leave-one-shot-budget-out (V2, Angle-3 shots generalization) -----
    # Holds out each distinct base_shots budget and predicts it from the
    # others — the shots-axis extrapolation number the boundary overlay
    # quotes. lobo-shaped (per_budget_* + n_budgets). Only meaningful with
    # feature_version=2 (log2_shots present) but computed whenever the column
    # is there, so it is None on scalar-shots runs and never perturbs V1.
    loso = None
    if budgets is not None:
        loso_result = _leave_one_group_out(
            _MODEL_FACTORIES[best_model_name], X, y, budgets
        )
        if loso_result is not None:
            acc, f1, per_acc, per_f1, n_values = loso_result
            loso = {
                "accuracy": acc,
                "macro_f1": f1,
                "per_budget_accuracy": {str(k): v for k, v in per_acc.items()},
                "per_budget_macro_f1": {str(k): v for k, v in per_f1.items()},
                "n_budgets": n_values,
            }
            print(
                "[qemsel.model] leave-one-shot-budget-out: accuracy "
                f"{loso['accuracy']:.3f}, macro-F1 {loso['macro_f1']:.3f} "
                f"({loso['n_budgets']} budgets)"
            )

    # ---- permutation importances on HELD-OUT folds ------------------------
    if cv_folds >= 2:
        fold_importances: list[np.ndarray] = []
        for train_idx, test_idx in splits:
            fold_model = _MODEL_FACTORIES[best_model_name]()
            fold_model.fit(X_cv.iloc[train_idx], y_cv[train_idx])
            perm = permutation_importance(
                fold_model,
                X_cv.iloc[test_idx],
                y_cv[test_idx],
                n_repeats=10,
                random_state=0,
            )
            fold_importances.append(perm.importances_mean)
        importances_mean = np.mean(fold_importances, axis=0)
        importances_note = "held-out (mean over CV folds)"
    else:
        # Degenerate path: training-set importances of a memorizing model —
        # flagged so nobody mistakes them for science.
        train_model = _MODEL_FACTORIES[best_model_name]()
        train_model.fit(X, y)
        perm = permutation_importance(
            train_model, X, y, n_repeats=10, random_state=0
        )
        importances_mean = perm.importances_mean
        importances_note = "training-set (UNRELIABLE: cv_folds=0)"
    feature_importances = {
        col: float(val) for col, val in zip(feat_cols, importances_mean)
    }

    # ---- V2: held-out probabilities for calibration / abstain metrics -----
    # Only computed when a probability-consuming flag is set, so the default
    # path does no extra model fitting and stays byte-identical.
    best_factory = _MODEL_FACTORIES[best_model_name]
    labels_eval = sorted(set(eval_y))
    calibration = None
    abstain_rate_cv = None
    if calibrate or abstain_threshold is not None:
        if cv_folds >= 2:
            uncal_proba = _oof_proba(best_factory, X_cv, y_cv, splits, labels_eval)
            cal_proba = (
                _oof_proba(
                    best_factory, X_cv, y_cv, splits, labels_eval,
                    groups=groups_cv, n_splits=n_splits, calibrated=True,
                )
                if calibrate
                else None
            )
        else:
            _bm = best_factory()
            _bm.fit(X, y)
            uncal_proba = _proba_matrix(_bm, X, labels_eval)
            cal_proba = (
                _proba_matrix(
                    _fit_calibrated(best_factory, X, y, groups, n_splits),
                    X, labels_eval,
                )
                if calibrate
                else None
            )
        if calibrate:
            calibration = {
                "method": "sigmoid",
                "brier_before": _multiclass_brier(uncal_proba, eval_y, labels_eval),
                "brier_after": _multiclass_brier(cal_proba, eval_y, labels_eval),
            }
        if abstain_threshold is not None:
            src = cal_proba if calibrate else uncal_proba
            abstain_rate_cv = float(np.mean(src.max(axis=1) < abstain_threshold))

    # ---- refit best model on ALL data, persist bundle ---------------------
    # NOTE: X, y — every usable row, INCLUDING rows of dropped classes.
    if calibrate:
        bundle_model = _fit_calibrated(best_factory, X, y, groups, n_splits)
    else:
        bundle_model = best_factory()
        bundle_model.fit(X, y)
    bundle = {
        "model": bundle_model,
        "feature_names": feat_cols,
        "classes": labels,
        "model_name": best_model_name,
        "label_column": label_column,
        "qemsel_version": qemsel.__version__,
    }
    # V2 bundle keys are ADDITIVE and appear only on a V2 path (any of
    # feature_version != 1 / calibrate / abstain_threshold). A pure default
    # run keeps the exact 6-key V1 bundle so recommend.py's V1 branch and the
    # frozen bundle-shape test stay valid. extended_stats is metrics-only.
    _v2_bundle = (
        feature_version != 1 or calibrate or abstain_threshold is not None
    )
    if _v2_bundle:
        bundle["feature_version"] = int(feature_version)
        bundle["calibrated"] = bool(calibrate)
        bundle["abstain_threshold"] = (
            float(abstain_threshold) if abstain_threshold is not None else None
        )
    joblib.dump(bundle, out_dir / bundle_filename)

    # extended_stats: per-fold accuracies on each per_model entry (behind the
    # flag, so the default per_model schema stays byte-identical).
    if extended_stats:
        for name in per_model:
            per_model[name]["fold_accuracies"] = [
                float(a) for a in fold_accs_by_model.get(name, [])
            ]

    metrics = {
        "best_model_name": best_model_name,
        "accuracy": per_model[best_model_name]["accuracy"],
        "macro_f1": per_model[best_model_name]["macro_f1"],
        "baseline_accuracy": per_model[_BASELINE_NAME]["accuracy"],
        "labels": labels,
        "confusion_matrix": [[int(v) for v in row] for row in cm],
        "feature_importances": feature_importances,
        "feature_importances_note": importances_note,
        "per_model": per_model,
        "n_samples": int(len(y)),
        "cv_n_samples": int(len(eval_y)),
        "cv_folds": int(cv_folds),
        "cv_grouping": cv_grouping,
        "dropped_classes": list(dropped_classes),
        "label_column": label_column,
        "lofo": lofo,
        "lobo": lobo,
        "lodo": lodo,
    }
    # ---- V2 additive metrics (present ONLY when triggered) ----------------
    if loso is not None:
        metrics["loso"] = loso
    if feature_version != 1:
        metrics["feature_version"] = int(feature_version)
    if calibrate:
        metrics["calibration"] = calibration
    if abstain_threshold is not None:
        metrics["abstain_threshold"] = float(abstain_threshold)
        metrics["abstain_rate_cv"] = abstain_rate_cv
    if extended_stats:
        metrics["fold_summary"] = stats.summarize_folds(
            fold_accs_by_model[best_model_name]
        )

    with (out_dir / metrics_filename).open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    return metrics


def train_and_eval_all(
    df: pd.DataFrame,
    out_dir: Path,
    *,
    feature_version: int = 1,
    calibrate: bool = False,
    abstain_threshold: float | None = None,
    extended_stats: bool = False,
) -> dict:
    """Train BOTH winner-label models when the data supports them.

    V2 (builder-model / B7): the keyword-only options are forwarded
    verbatim to every underlying ``train_and_eval`` call; defaults keep the
    V1 behavior byte-identical (the options are stub-guarded there).

    One-call pipeline entry point:

    1. Always trains the accuracy-at-any-shot-cost ``best_technique`` model
       -> ``model.joblib`` + ``metrics.json`` (identical to calling
       ``train_and_eval(df, out_dir)``).
    2. When the ``best_technique_cost_aware`` column exists AND has usable
       rows, additionally trains the equal-shot-budget model ->
       ``model_cost_aware.joblib`` + ``metrics_cost_aware.json``. A legacy
       CSV without the column (or with an all-empty column) skips this step
       with a notice instead of crashing — full backward compatibility.
    3. When both trainings succeed, ``metrics.json`` is rewritten with the
       cost-aware metrics embedded under the extra key ``'cost_aware'`` so
       downstream report generation (``qemsel.report.generate_report``)
       renders both label variants side by side without extra plumbing.
       The dict RETURNED for 'best_technique' keeps the exact
       ``train_and_eval`` schema (no 'cost_aware' key).

    Args:
        df: experiment results DataFrame (raw or seed-aggregated schema).
        out_dir: output directory; created if missing.

    Returns:
        {'best_technique': <train_and_eval metrics dict>,
         'best_technique_cost_aware': <metrics dict> | None}
    """
    out_dir = Path(out_dir)
    v2_kwargs = dict(
        feature_version=feature_version,
        calibrate=calibrate,
        abstain_threshold=abstain_threshold,
        extended_stats=extended_stats,
    )
    primary = train_and_eval(df, out_dir, "best_technique", **v2_kwargs)

    cost_metrics: dict | None = None
    if COST_AWARE_LABEL in df.columns:
        try:
            cost_metrics = train_and_eval(
                df,
                out_dir,
                COST_AWARE_LABEL,
                bundle_filename="model_cost_aware.joblib",
                metrics_filename="metrics_cost_aware.json",
                **v2_kwargs,
            )
        except ValueError as exc:
            print(
                f"[qemsel.model] NOTE: cost-aware model skipped — "
                f"'{COST_AWARE_LABEL}' column present but unusable ({exc})"
            )
    else:
        print(
            f"[qemsel.model] NOTE: no '{COST_AWARE_LABEL}' column — "
            "training the accuracy-only model (legacy schema)."
        )

    if cost_metrics is not None:
        combined = dict(primary)
        combined["cost_aware"] = cost_metrics
        with (out_dir / "metrics.json").open("w", encoding="utf-8") as fh:
            json.dump(combined, fh, indent=2)

    return {
        "best_technique": primary,
        COST_AWARE_LABEL: cost_metrics,
    }
