"""Statistical hygiene helpers (Koester-Mauerer checklist, CIs, effect sizes).

V2 module (INTERFACES.md section V2; builder-stats / B6 implements every
``NotImplementedError`` body). Pure numpy/pandas/scipy-free statistics on the
experiment DataFrames — NO quantum imports, NO qemsel cross-module imports
(this module sits at the bottom of the dependency DAG so model.py and the
stats CLI can both import it freely).

Design rules (binding):
* Every function is deterministic given its ``seed`` parameter
  (``numpy.random.default_rng(seed)``, never global state).
* Every return value is JSON-serializable (plain float/int/str/bool/
  list/dict — no numpy scalars; cast with float()/int()).
* NaN policy is stated per function; silently producing NaN output from
  NaN input is forbidden.
* Both experiment schemas are accepted where a DataFrame is taken:
  per-seed ``results.csv`` (``<tech>_abs_error`` columns) and seed-averaged
  ``aggregated.csv`` (``<tech>_mean_abs_error`` columns); detect by column
  presence like report.py does.

B6 also owns ``scripts/compute_stats.py`` (CLI:
``--data <results.csv|aggregated.csv> --out <dir>``) writing ``stats.json``:
``{'win_share_ci': {label_col: {tech: <bootstrap dict>}},
'paired_tests': {'raw_plus_vs_raw': <permutation dict>,
'top2_<a>_vs_<b>': <permutation dict>}, 'effect_sizes': {name: <cliffs
delta float>}, 'checklist': <koester_checklist dict>, 'n_rows': int,
'data_path': str}`` — exactly the dict ``qemsel.report.generate_report``
accepts as ``stats_results``.
"""

from __future__ import annotations

import math
from statistics import NormalDist
from typing import Callable, Sequence

import numpy as np
import pandas as pd

#: Default number of bootstrap resamples / permutations (paper-grade; tests
#: pass smaller values through the keyword).
DEFAULT_N_BOOT: int = 10_000
DEFAULT_N_PERM: int = 10_000

#: Default significance multiplier for shot-noise margins (winner must beat
#: runner-up by k * sigma to count as significant; shared default with
#: qemsel.model.derive_significant_label).
DEFAULT_K_SIGMA: float = 2.0

#: Tolerance for the "beyond physical range" checklist comparisons — floating
#: slack so an estimate of exactly 1.0 is never flagged.
_PHYSICAL_TOL: float = 1e-9


def sigma_shot(value: float, shots: float) -> float:
    """Shot-noise standard deviation of a Pauli-expectation estimate.

    ``sqrt((1 - min(value**2, 1)) / shots)`` — the binomial std of an
    n-shot estimate of an expectation with true value ``value`` (values
    outside [-1, 1], possible for mitigated estimates, are clamped to 1 in
    the variance term, giving sigma 0 — the honest lower bound there is 0
    because the analytic formula does not cover unphysical estimates).

    Raises:
        ValueError: shots <= 0 or value is NaN.
    """
    v = float(value)
    s = float(shots)
    if math.isnan(v):
        raise ValueError("sigma_shot: value is NaN")
    if math.isnan(s) or s <= 0:
        raise ValueError(f"sigma_shot: shots must be > 0 (got {shots!r})")
    var_term = 1.0 - min(v * v, 1.0)
    # var_term is in [0, 1]; negative is impossible after the clamp.
    return float(math.sqrt(var_term / s))


def win_shares(
    labels: Sequence[str] | pd.Series,
    techniques: Sequence[str] | None = None,
) -> dict[str, float]:
    """Fraction of rows each technique wins.

    Empty-string / NaN labels are EXCLUDED from the denominator (they mean
    "all techniques failed", not a win for anyone). ``techniques`` None =>
    the distinct labels present, sorted; explicitly passed techniques absent
    from ``labels`` get share 0.0. Shares over the returned keys sum to 1.0
    (up to float error) when techniques is None.

    Returns:
        dict technique -> share in [0, 1].
    """
    s = pd.Series(list(labels), dtype="object")
    s = s[s.notna()]
    s = s[s.astype(str) != ""]
    denom = int(s.shape[0])

    if techniques is None:
        keys = sorted(str(x) for x in s.unique())
    else:
        keys = [str(t) for t in techniques]

    if denom == 0:
        return {k: 0.0 for k in keys}

    counts = s.astype(str).value_counts()
    return {k: float(int(counts.get(k, 0)) / denom) for k in keys}


def _clean_1d(values: Sequence[float] | np.ndarray | pd.Series) -> tuple[np.ndarray, int]:
    """Return (finite-valued float array, count of dropped NaNs)."""
    arr = np.asarray(values, dtype=float).ravel()
    mask = ~np.isnan(arr)
    return arr[mask], int((~mask).sum())


def bootstrap_ci(
    values: Sequence[float] | np.ndarray | pd.Series,
    statistic: Callable[[np.ndarray], float] | None = None,
    *,
    n_boot: int = DEFAULT_N_BOOT,
    ci: float = 0.95,
    seed: int = 0,
    groups: Sequence | np.ndarray | pd.Series | None = None,
) -> dict:
    """Percentile bootstrap confidence interval for one statistic.

    NaNs are dropped first (count reported). ``statistic`` None => mean.
    Resampling via ``numpy.random.default_rng(seed)`` (deterministic).

    ``groups`` (ADDITIVE, findings-applier 2026-07-23): when given (one label
    per value, same length), the bootstrap resamples whole CLUSTERS (groups)
    with replacement instead of rows. Rows sharing a circuit are strongly
    correlated (measured ICC up to 0.64 on the research sweep), so the
    default row-level resampling understates the CI width — this is exactly
    the reduced-effective-N artefact the Koester-Mauerer paper warns about.
    Default None preserves the previous behavior byte-identically.

    Returns:
        dict with keys EXACTLY: 'estimate' (statistic on the full data),
        'lo', 'hi' (percentile bounds), 'ci' (echo), 'n' (non-NaN count),
        'n_dropped_nan' (int), 'n_boot' (echo), 'seed' (echo). When
        ``groups`` is given, additionally 'n_groups' (int) and
        'resample': 'cluster'.

    Raises:
        ValueError: fewer than 2 non-NaN values, ci not in (0, 1), a groups
            length mismatch, or < 2 distinct groups after NaN dropping.
    """
    if not (0.0 < ci < 1.0):
        raise ValueError(f"ci must be in (0, 1); got {ci!r}")
    arr = np.asarray(values, dtype=float).ravel()
    mask = ~np.isnan(arr)
    clean = arr[mask]
    n_dropped = int((~mask).sum())
    n = int(clean.shape[0])
    if n < 2:
        raise ValueError(f"bootstrap_ci needs >= 2 non-NaN values; got {n}")

    stat_fn = (lambda a: float(np.mean(a))) if statistic is None else statistic
    estimate = float(stat_fn(clean))

    rng = np.random.default_rng(seed)

    if groups is None:
        idx = rng.integers(0, n, size=(int(n_boot), n))
        resamples = clean[idx]
        if statistic is None:
            boot = resamples.mean(axis=1)
        else:
            boot = np.array(
                [float(statistic(row)) for row in resamples], dtype=float
            )
        extra: dict = {}
    else:
        glabels = np.asarray(list(groups), dtype=object).ravel()
        if glabels.shape[0] != arr.shape[0]:
            raise ValueError(
                f"groups length {glabels.shape[0]} != values length {arr.shape[0]}"
            )
        glabels = glabels[mask]
        uniq = sorted(set(str(g) for g in glabels), key=str)
        n_groups = len(uniq)
        if n_groups < 2:
            raise ValueError(
                f"cluster bootstrap needs >= 2 distinct groups; got {n_groups}"
            )
        members = [clean[np.asarray([str(g) for g in glabels]) == u] for u in uniq]
        boot = np.empty(int(n_boot), dtype=float)
        for b in range(int(n_boot)):
            pick = rng.integers(0, n_groups, size=n_groups)
            sample = np.concatenate([members[int(k)] for k in pick])
            boot[b] = float(stat_fn(sample))
        extra = {"n_groups": int(n_groups), "resample": "cluster"}

    alpha = (1.0 - ci) / 2.0
    lo = float(np.quantile(boot, alpha))
    hi = float(np.quantile(boot, 1.0 - alpha))
    return {
        "estimate": estimate,
        "lo": lo,
        "hi": hi,
        "ci": float(ci),
        "n": n,
        "n_dropped_nan": n_dropped,
        "n_boot": int(n_boot),
        "seed": int(seed),
        **extra,
    }


def win_share_ci(
    labels: Sequence[str] | pd.Series,
    technique: str,
    *,
    n_boot: int = DEFAULT_N_BOOT,
    ci: float = 0.95,
    seed: int = 0,
    groups: Sequence | np.ndarray | pd.Series | None = None,
) -> dict:
    """Bootstrap CI of one technique's win share.

    Convenience wrapper: resamples the label vector (empty/NaN labels
    excluded as in :func:`win_shares`) and applies the share-of-``technique``
    statistic through :func:`bootstrap_ci`'s machinery. ``groups`` (ADDITIVE)
    switches to the cluster bootstrap — see :func:`bootstrap_ci`.

    KNOWN LIMIT (findings-applier 2026-07-23): the percentile bootstrap of a
    0/n (or n/n) proportion collapses to a zero-width interval with 0%
    actual coverage — exactly where the Angle-3 claim lives (ZNE win share
    ~0 in low-noise/low-shot slices). Report :func:`wilson_interval`
    alongside (or instead) for proportions near the boundary; the stats CLI
    does this automatically.

    Returns:
        the :func:`bootstrap_ci` dict shape plus 'technique' (echo).
    """
    s = pd.Series(list(labels), dtype="object")
    keep = s.notna() & (s.astype(str) != "")
    s = s[keep]
    indicator = (s.astype(str) == str(technique)).to_numpy(dtype=float)
    g = None
    if groups is not None:
        garr = pd.Series(list(groups), dtype="object")
        if len(garr) != len(keep):
            raise ValueError(
                f"groups length {len(garr)} != labels length {len(keep)}"
            )
        g = garr[keep.to_numpy()].to_numpy()
    out = bootstrap_ci(
        indicator, statistic=None, n_boot=n_boot, ci=ci, seed=seed, groups=g
    )
    out["technique"] = str(technique)
    return out


def wilson_interval(k: int, n: int, *, ci: float = 0.95) -> dict:
    """Wilson score interval for a binomial proportion (ADDITIVE, 2026-07-23).

    The percentile bootstrap of a proportion returns a degenerate [p, p]
    interval at p = 0 or 1 (0% coverage); the Wilson interval stays honest
    at the boundary (e.g. 0/24 successes -> 95% CI [0, 0.138]). Deterministic,
    closed-form, no resampling.

    Args:
        k: number of successes (0 <= k <= n).
        n: number of trials (> 0).
        ci: confidence level in (0, 1).

    Returns:
        dict with keys EXACTLY: 'estimate' (k/n), 'lo', 'hi', 'ci', 'k',
        'n', 'method' ('wilson').

    Raises:
        ValueError: n <= 0, k out of range, or ci not in (0, 1).
    """
    if not (0.0 < ci < 1.0):
        raise ValueError(f"ci must be in (0, 1); got {ci!r}")
    n_i = int(n)
    k_i = int(k)
    if n_i <= 0:
        raise ValueError(f"n must be > 0; got {n!r}")
    if not (0 <= k_i <= n_i):
        raise ValueError(f"k must be in [0, n]; got k={k!r}, n={n!r}")
    z = float(NormalDist().inv_cdf(0.5 + ci / 2.0))
    p = k_i / n_i
    denom = 1.0 + z * z / n_i
    centre = (p + z * z / (2 * n_i)) / denom
    half = (z / denom) * math.sqrt(p * (1.0 - p) / n_i + z * z / (4.0 * n_i * n_i))
    return {
        "estimate": float(p),
        "lo": float(max(0.0, centre - half)),
        "hi": float(min(1.0, centre + half)),
        "ci": float(ci),
        "k": k_i,
        "n": n_i,
        "method": "wilson",
    }


def adjust_pvalues(pvalues: dict[str, float], method: str = "holm") -> dict[str, float]:
    """Multiple-comparison adjustment over a family of p-values (ADDITIVE).

    ``method``: 'holm' (step-down Bonferroni, controls FWER — the default
    for the report's confirmatory contrasts) or 'bh' (Benjamini-Hochberg,
    controls FDR). Adjusted values are clipped to [0, 1] and keep the input
    keys. Deterministic; ties broken by insertion order.

    Raises:
        ValueError: empty mapping, NaN p-value, p outside [0, 1], or an
            unknown method.
    """
    if method not in ("holm", "bh"):
        raise ValueError(f"method must be 'holm' or 'bh'; got {method!r}")
    items = list(pvalues.items())
    if not items:
        raise ValueError("adjust_pvalues: empty p-value mapping")
    for name, p in items:
        pf = float(p)
        if math.isnan(pf) or not (0.0 <= pf <= 1.0):
            raise ValueError(f"invalid p-value for {name!r}: {p!r}")
    m = len(items)
    order = sorted(range(m), key=lambda i: float(items[i][1]))
    adjusted = [0.0] * m
    if method == "holm":
        running = 0.0
        for rank, idx in enumerate(order):
            val = min(1.0, (m - rank) * float(items[idx][1]))
            running = max(running, val)
            adjusted[idx] = running
    else:  # bh: step-up
        running = 1.0
        for rev_rank, idx in enumerate(reversed(order)):
            rank = m - rev_rank  # 1-based rank from smallest
            val = min(1.0, m / rank * float(items[idx][1]))
            running = min(running, val)
            adjusted[idx] = running
    return {items[i][0]: float(adjusted[i]) for i in range(m)}


def paired_permutation_test(
    err_a: Sequence[float] | np.ndarray | pd.Series,
    err_b: Sequence[float] | np.ndarray | pd.Series,
    *,
    n_perm: int = DEFAULT_N_PERM,
    seed: int = 0,
    alternative: str = "two-sided",
    groups: Sequence | np.ndarray | pd.Series | None = None,
) -> dict:
    """Paired sign-flip permutation test on mean(err_a - err_b).

    Pairs where EITHER value is NaN are dropped pairwise (count reported).
    Statistic: mean paired difference. Null: differences are symmetric
    around 0; each permutation flips signs uniformly at random
    (``default_rng(seed)``). ``alternative``: 'two-sided' | 'less'
    (err_a < err_b, i.e. A better) | 'greater'. p-value uses the
    add-one correction: ``(1 + #extreme) / (1 + n_perm)``.

    ``groups`` (ADDITIVE, findings-applier 2026-07-23): one cluster label
    per pair; when given, sign flips are drawn per CLUSTER and applied to
    every pair in the cluster together. Rows sharing a circuit are strongly
    correlated (measured ICC 0.25-0.64 on the research sweep), so row-level
    flips overstate the effective N and understate p (a 24-row/3-circuit
    smoke slice produced p=0.0002 where 3 exchangeable clusters cannot go
    below ~0.125). Note the granularity: with only G clusters there are
    2**G distinct flip patterns, so the achievable p floor is ~2/2**G
    (two-sided) — report G alongside p. Default None preserves the previous
    behavior byte-identically.

    Returns:
        dict with keys EXACTLY: 'mean_diff' (float, mean(a - b)),
        'p_value' (float), 'n_pairs' (int), 'n_dropped_nan' (int),
        'n_perm', 'alternative', 'seed' (echoes). When ``groups`` is given,
        additionally 'n_groups' (int) and 'exchange_unit': 'cluster'.

    Raises:
        ValueError: length mismatch, < 2 usable pairs, bad alternative,
            groups length mismatch, or < 2 distinct usable clusters.
    """
    if alternative not in ("two-sided", "less", "greater"):
        raise ValueError(
            f"alternative must be 'two-sided' | 'less' | 'greater'; got {alternative!r}"
        )
    a = np.asarray(err_a, dtype=float).ravel()
    b = np.asarray(err_b, dtype=float).ravel()
    if a.shape != b.shape:
        raise ValueError(f"length mismatch: {a.shape[0]} vs {b.shape[0]}")

    diff = a - b
    mask = ~np.isnan(diff)
    n_dropped = int((~mask).sum())
    d = diff[mask]
    n_pairs = int(d.shape[0])
    if n_pairs < 2:
        raise ValueError(f"need >= 2 usable pairs; got {n_pairs}")

    observed = float(d.mean())

    rng = np.random.default_rng(seed)
    extra: dict = {}
    if groups is None:
        # +/-1 sign flips, one draw per (permutation, pair).
        signs = rng.integers(0, 2, size=(int(n_perm), n_pairs)) * 2 - 1
        perm_means = (signs * d).mean(axis=1)
    else:
        glabels = np.asarray(list(groups), dtype=object).ravel()
        if glabels.shape[0] != a.shape[0]:
            raise ValueError(
                f"groups length {glabels.shape[0]} != pairs length {a.shape[0]}"
            )
        gkept = np.asarray([str(g) for g in glabels[mask]])
        uniq = sorted(set(gkept))
        n_groups = len(uniq)
        if n_groups < 2:
            raise ValueError(
                f"cluster permutation needs >= 2 distinct groups; got {n_groups}"
            )
        gidx = np.asarray([uniq.index(g) for g in gkept], dtype=int)
        # One +/-1 draw per (permutation, CLUSTER), broadcast to members.
        gsigns = rng.integers(0, 2, size=(int(n_perm), n_groups)) * 2 - 1
        perm_means = (gsigns[:, gidx] * d).mean(axis=1)
        extra = {"n_groups": int(n_groups), "exchange_unit": "cluster"}

    if alternative == "two-sided":
        extreme = int(np.sum(np.abs(perm_means) >= abs(observed) - 1e-15))
    elif alternative == "less":
        extreme = int(np.sum(perm_means <= observed + 1e-15))
    else:  # greater
        extreme = int(np.sum(perm_means >= observed - 1e-15))

    p_value = float((1 + extreme) / (1 + int(n_perm)))
    return {
        "mean_diff": observed,
        "p_value": p_value,
        "n_pairs": n_pairs,
        "n_dropped_nan": n_dropped,
        "n_perm": int(n_perm),
        "alternative": alternative,
        "seed": int(seed),
        **extra,
    }


def cliffs_delta(
    a: Sequence[float] | np.ndarray | pd.Series,
    b: Sequence[float] | np.ndarray | pd.Series,
    *,
    paired: bool = False,
) -> float:
    """Cliff's delta effect size: P(a > b) - P(a < b) over all pairs.

    NaNs dropped per-array (listwise within each array, not pairwise — the
    two samples need not be the same length). Returns a float in [-1, 1];
    negative means a tends SMALLER than b (for error columns: a better).

    ``paired=True`` (ADDITIVE, findings-applier 2026-07-23): requires equal
    lengths and drops PAIRS where either value is NaN before comparing. For
    paired technique-error columns with refusals (e.g. CDR NaN on 415/1620
    research rows) the default per-array drop is asymmetric: the rival keeps
    its (easier) rows where CDR refused, flattering the refusing technique.
    Pairwise dropping conditions BOTH samples on the same row set. Default
    False preserves the previous behavior byte-identically.

    Raises:
        ValueError: either array has no non-NaN values (or, with
            paired=True, a length mismatch / no usable pairs).
    """
    if paired:
        a_arr = np.asarray(a, dtype=float).ravel()
        b_arr = np.asarray(b, dtype=float).ravel()
        if a_arr.shape != b_arr.shape:
            raise ValueError(
                f"cliffs_delta(paired=True): length mismatch "
                f"{a_arr.shape[0]} vs {b_arr.shape[0]}"
            )
        keep = ~(np.isnan(a_arr) | np.isnan(b_arr))
        a_clean, b_clean = a_arr[keep], b_arr[keep]
    else:
        a_clean, _ = _clean_1d(a)
        b_clean, _ = _clean_1d(b)
    na = int(a_clean.shape[0])
    nb = int(b_clean.shape[0])
    if na == 0 or nb == 0:
        raise ValueError("cliffs_delta: both arrays need >= 1 non-NaN value")

    # Broadcast comparison; na*nb pairs. (1620^2 ~ 2.6M — comfortable.)
    diff = a_clean[:, None] - b_clean[None, :]
    greater = int(np.count_nonzero(diff > 0))
    less = int(np.count_nonzero(diff < 0))
    return float((greater - less) / (na * nb))


def summarize_folds(fold_scores: Sequence[float]) -> dict:
    """Per-fold summary used by model/report for CV-score tables.

    Returns:
        dict with keys EXACTLY: 'mean', 'std' (ddof=1; 0.0 when < 2 folds),
        'min', 'max', 'n_folds'. All plain floats/ints.

    Raises:
        ValueError: empty input or any NaN score.
    """
    arr = np.asarray(list(fold_scores), dtype=float).ravel()
    if arr.shape[0] == 0:
        raise ValueError("summarize_folds: empty input")
    if np.isnan(arr).any():
        raise ValueError("summarize_folds: NaN score(s) present")
    n = int(arr.shape[0])
    std = float(np.std(arr, ddof=1)) if n >= 2 else 0.0
    return {
        "mean": float(np.mean(arr)),
        "std": std,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "n_folds": n,
    }


# --------------------------------------------------------------------------- #
# koester_checklist and its schema helpers
# --------------------------------------------------------------------------- #

def _detect_schema(df: pd.DataFrame) -> tuple[str, list[str], str]:
    """Return (schema, techniques, err_suffix).

    schema is 'per_seed' | 'aggregated'; techniques are in COLUMN-APPEARANCE
    order (== the experiment's config order, which fixes the argmin
    tie-break); err_suffix is the per-technique error-column suffix.
    """
    mean_suffix = "_mean_abs_error"
    err_suffix = "_abs_error"
    mean_cols = [c for c in df.columns if c.endswith(mean_suffix)]
    plain_cols = [
        c for c in df.columns if c.endswith(err_suffix) and not c.endswith(mean_suffix)
    ]
    if plain_cols:
        techs = [c[: -len(err_suffix)] for c in plain_cols]
        return "per_seed", list(dict.fromkeys(techs)), err_suffix
    if mean_cols:
        techs = [c[: -len(mean_suffix)] for c in mean_cols]
        return "aggregated", list(dict.fromkeys(techs)), mean_suffix
    raise ValueError(
        "koester_checklist: DataFrame has no '<tech>_abs_error' / "
        "'<tech>_mean_abs_error' columns — not an experiment results frame"
    )


def koester_checklist(
    df: pd.DataFrame,
    *,
    k_sigma: float = DEFAULT_K_SIGMA,
    sigma_fn: Callable[[str, float, float], float] | None = None,
) -> dict:
    """INTERNAL data-integrity checklist over an experiment DataFrame.

    NAMING CAVEAT (findings-applier 2026-07-23): this is qemsel's own
    six-item integrity gate MOTIVATED BY the Koester-Mauerer review
    framework (arXiv:2605.29872), NOT an implementation of that paper's
    eight criteria (their items are parameter documentation, sensitivity/
    knob-robustness sweeps, longitudinal drift assessment, and inferential
    testing with effect sizes — of which qemsel covers the inferential/
    effect-size/CI part in this module's other functions and parameter
    logging via run_meta.json; drift is N/A sim-side; knob-robustness and
    clustered effective-N remain open items the report discloses). The
    function name is kept for API stability; report.py renders the section
    under the honest heading.

    Accepts BOTH schemas (per-seed results.csv / aggregated.csv;
    detected by column presence). Techniques are auto-detected from the
    ``<tech>_abs_error`` / ``<tech>_mean_abs_error`` columns.

    ``sigma_fn`` (ADDITIVE): optional per-technique sigma model
    ``sigma_fn(technique, value, shots) -> float`` used by the
    winner-margin check instead of the pooled-shots binomial formula. The
    default binomial ``sqrt((1-v^2)/shots)`` treats the consumed-shots
    ledger as pooled shots, which understates sigma ~2-8x for
    extrapolation/correction estimators (zne 7.6x with Richardson
    coefficients (3,-3,1)); pass ``qemsel.mitigation.estimator_sigma``
    (adapted to the ledger convention) for estimator-aware margins — the
    stats CLI does. A returned inf/NaN flags the row as a tie
    (conservative). Default None is byte-identical to the old behavior.

    Checks (each entry JSON-serializable; counts are ints, fractions floats):

    * 'overshoot_beyond_physical_max': per technique, number of rows with
      ``|<tech>_value| > 1 + tol`` (tol 1e-9) — a Pauli expectation beyond
      the physical range flags variance blow-up (mitigated values MAY
      legitimately overshoot; the checklist reports, it does not censor).
      Only for schemas carrying ``<tech>_value`` columns; None otherwise.
    * 'error_beyond_physical_max': per technique, rows with
      ``abs_error > 1 + |ideal| + tol`` (worse than maximally wrong).
      None when no 'ideal' column (aggregated schema).
    * 'nan_rate': per technique, fraction of rows with NaN error (refusals
      + failures; cross-check against errors.log is the caller's job).
    * 'label_argmin_consistent': {'n_checked', 'n_mismatch'} — recompute
      argmin over the error columns and compare to 'best_technique'
      (rows with all-NaN errors skipped).
    * 'winner_margin_below_k_sigma': {'k_sigma', 'n_flagged', 'fraction'} —
      rows where (runner-up error - winner error) < k_sigma * combined
      shot-noise sigma of the two estimates (via :func:`sigma_shot` when
      value/shots columns exist; None n_flagged/fraction on the aggregated
      schema, which carries neither per-shot value/shots nor per-seed
      scatter — see the note below).
    * 'partial_coverage_winners': int — aggregated schema only (rows whose
      winner's ``<tech>_n_seeds`` < ``n_seeds``); None on per-seed schema.
      (Expected 0 — the experiment's coverage rule forbids it.)

    Note (winner_margin on aggregated): a defensible shot-noise margin needs
    either ``<tech>_value`` + ``<tech>_shots`` (per-seed) or a per-seed
    scatter column. The V1 aggregated.csv stores only ``<tech>_mean_abs_error``
    and ``<tech>_n_seeds`` — no shots, no scatter — and this module may not
    import ``qemsel`` to reconstruct shots from ``base_shots``, so the margin
    is reported as None rather than a fabricated 0. It IS computed whenever
    value+shots columns are present (the graded ``results.csv`` path).

    Returns:
        dict with keys EXACTLY: 'schema' ('per_seed' | 'aggregated'),
        'n_rows' (int), 'techniques' (list[str]), 'checks' (dict of the six
        entries above), 'passed' (bool: True iff
        label_argmin_consistent.n_mismatch == 0 and
        (partial_coverage_winners in (None, 0))).

    Raises:
        ValueError: no technique error columns / no best_technique column.
    """
    schema, techniques, err_suffix = _detect_schema(df)
    if "best_technique" not in df.columns:
        raise ValueError("koester_checklist: DataFrame lacks a 'best_technique' column")

    n_rows = int(df.shape[0])
    tol = _PHYSICAL_TOL

    def err_col(t: str) -> str:
        return f"{t}{err_suffix}"

    # --- overshoot_beyond_physical_max -------------------------------------
    has_value = any(f"{t}_value" in df.columns for t in techniques)
    if has_value:
        overshoot: dict[str, int] | None = {}
        for t in techniques:
            vcol = f"{t}_value"
            if vcol in df.columns:
                vals = pd.to_numeric(df[vcol], errors="coerce")
                overshoot[t] = int((vals.abs() > 1.0 + tol).sum())
    else:
        overshoot = None

    # --- error_beyond_physical_max -----------------------------------------
    if "ideal" in df.columns:
        ideal_abs = pd.to_numeric(df["ideal"], errors="coerce").abs()
        error_beyond: dict[str, int] | None = {}
        for t in techniques:
            errs = pd.to_numeric(df[err_col(t)], errors="coerce")
            error_beyond[t] = int((errs > 1.0 + ideal_abs + tol).sum())
    else:
        error_beyond = None

    # --- nan_rate ----------------------------------------------------------
    nan_rate: dict[str, float] = {}
    for t in techniques:
        errs = pd.to_numeric(df[err_col(t)], errors="coerce")
        nan_rate[t] = float(errs.isna().mean()) if n_rows else 0.0

    # --- label_argmin_consistent -------------------------------------------
    err_mat = pd.DataFrame(
        {t: pd.to_numeric(df[err_col(t)], errors="coerce") for t in techniques}
    )
    # The aggregated winner is argmin RESTRICTED to techniques with MAXIMUM
    # seed coverage in the group (experiment.py _aggregate coverage rule):
    # a partial-coverage mean is not comparable, so those techniques are
    # ineligible to win. Mirror that here or every partial-coverage row reads
    # as a spurious mismatch. The per-seed schema has no coverage rule.
    argmin_mat = err_mat
    if schema == "aggregated" and all(f"{t}_n_seeds" in df.columns for t in techniques):
        cov = pd.DataFrame(
            {t: pd.to_numeric(df[f"{t}_n_seeds"], errors="coerce") for t in techniques}
        )
        max_cov = cov.max(axis=1)
        eligible = cov.eq(max_cov, axis=0)
        argmin_mat = err_mat.where(eligible)
    valid_mask = argmin_mat.notna().any(axis=1)
    n_checked = int(valid_mask.sum())
    best = df["best_technique"].where(df["best_technique"].notna(), "").astype(str)
    if n_checked:
        recomputed = argmin_mat.loc[valid_mask].idxmin(axis=1).astype(str)
        n_mismatch = int((recomputed != best.loc[valid_mask]).sum())
    else:
        n_mismatch = 0
    label_argmin_consistent = {"n_checked": n_checked, "n_mismatch": n_mismatch}

    # --- winner_margin_below_k_sigma ---------------------------------------
    has_value_shots = all(
        f"{t}_value" in df.columns and f"{t}_shots" in df.columns for t in techniques
    )
    if has_value_shots and techniques:
        E = err_mat.to_numpy(dtype=float)
        V = np.column_stack(
            [pd.to_numeric(df[f"{t}_value"], errors="coerce").to_numpy(float) for t in techniques]
        )
        S = np.column_stack(
            [pd.to_numeric(df[f"{t}_shots"], errors="coerce").to_numpy(float) for t in techniques]
        )
        n_flagged = 0
        n_margin_checked = 0
        for i in range(E.shape[0]):
            errs = E[i]
            finite = int(np.count_nonzero(~np.isnan(errs)))
            if finite < 2:
                continue
            order = np.argsort(errs, kind="stable")  # NaNs sort to the end
            w, r = int(order[0]), int(order[1])
            vw, sw = V[i, w], S[i, w]
            vr, sr = V[i, r], S[i, r]
            if (
                math.isnan(vw)
                or math.isnan(vr)
                or math.isnan(sw)
                or math.isnan(sr)
                or sw <= 0
                or sr <= 0
            ):
                continue
            if sigma_fn is not None:
                try:
                    sig_w = float(sigma_fn(techniques[w], vw, sw))
                    sig_r = float(sigma_fn(techniques[r], vr, sr))
                except (ValueError, TypeError):
                    sig_w = sig_r = math.inf
                if math.isnan(sig_w) or math.isnan(sig_r):
                    sig_w = sig_r = math.inf
            else:
                sig_w = math.sqrt((1.0 - min(vw * vw, 1.0)) / sw)
                sig_r = math.sqrt((1.0 - min(vr * vr, 1.0)) / sr)
            combined = math.sqrt(sig_w * sig_w + sig_r * sig_r)
            margin = float(errs[r] - errs[w])
            n_margin_checked += 1
            if margin < k_sigma * combined:
                n_flagged += 1
        winner_margin = {
            "k_sigma": float(k_sigma),
            "n_flagged": int(n_flagged),
            "fraction": float(n_flagged / n_margin_checked) if n_margin_checked else 0.0,
        }
        if sigma_fn is not None:
            winner_margin["sigma_mode"] = "estimator_aware"
    else:
        winner_margin = {
            "k_sigma": float(k_sigma),
            "n_flagged": None,
            "fraction": None,
        }

    # --- partial_coverage_winners ------------------------------------------
    if schema == "aggregated" and "n_seeds" in df.columns:
        n_seeds_col = pd.to_numeric(df["n_seeds"], errors="coerce")
        partial = 0
        for t in techniques:
            wcol = f"{t}_n_seeds"
            if wcol not in df.columns:
                continue
            is_winner = best == t
            tn = pd.to_numeric(df[wcol], errors="coerce")
            partial += int(((tn < n_seeds_col) & is_winner).sum())
        partial_coverage_winners: int | None = int(partial)
    else:
        partial_coverage_winners = None

    checks = {
        "overshoot_beyond_physical_max": overshoot,
        "error_beyond_physical_max": error_beyond,
        "nan_rate": nan_rate,
        "label_argmin_consistent": label_argmin_consistent,
        "winner_margin_below_k_sigma": winner_margin,
        "partial_coverage_winners": partial_coverage_winners,
    }
    passed = bool(
        label_argmin_consistent["n_mismatch"] == 0
        and (partial_coverage_winners is None or partial_coverage_winners == 0)
    )
    return {
        "schema": schema,
        "n_rows": n_rows,
        "techniques": list(techniques),
        "checks": checks,
        "passed": passed,
    }
