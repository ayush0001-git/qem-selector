"""Generate the final human-readable benchmark + model report.

Accepts both the raw ``results.csv`` schema (one row per unit, ``seed``
column) and the seed-averaged ``aggregated.csv`` schema (``n_seeds`` column,
no ``seed`` column) — nothing here reads ``seed``.

Noise-scale convention: a backend name may carry an ``@x<scale>`` suffix
(e.g. ``FakeManilaV2@x1.5`` = the Manila noise model with all error rates
scaled 1.5x). Plain names mean scale 1.0. Section 5 and the
``winner_vs_noise.png`` figure aggregate over that parsed scale — the money
plot for the "does more noise change the best technique" research question.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # must precede pyplot import; headless-safe

import matplotlib.pyplot as plt  # noqa: E402  (after matplotlib.use on purpose)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

#: Canonical technique ordering used when laying out tables/figures. Extra
#: techniques found in the DataFrame are appended alphabetically.
#: 'raw_plus' is the empirical equal-budget baseline (raw at the max shot
#: multiplier) — listed right after 'raw' when present. V2: the additive
#: techniques (mitigation.TECHNIQUES_V2) slot next to their siblings —
#: purely a display order; V1 DataFrames (no such columns) are unaffected.
_CANONICAL_TECHNIQUES: list[str] = [
    "raw",
    "raw_plus",
    "zne",
    "zne_fr",
    "cdr",
    "cdr_ridge",
    "cdr_rf",
    "rem",
]

_REQUIRED_DF_COLUMNS: list[str] = ["circuit_id", "family", "backend", "best_technique"]
_REQUIRED_METRIC_KEYS: list[str] = [
    "best_model_name",
    "accuracy",
    "macro_f1",
    "baseline_accuracy",
    "labels",
    "confusion_matrix",
    "feature_importances",
]

_PNG_ERROR = "error_by_technique.png"
_PNG_WIN = "win_rate.png"
_PNG_CONFUSION = "confusion_matrix.png"
_PNG_IMPORTANCES = "feature_importances.png"
_PNG_NOISE = "winner_vs_noise.png"

#: V2 (builder-recommend / B8): the boundary-overlay figure is PRODUCED by
#: qemsel.boundary.overlay_selector_vs_theory (constant OVERLAY_PNG there);
#: section 9 embeds it by the relative filename recorded in
#: boundary_overlay['plot_path']. Report.py never imports boundary.
_PNG_BOUNDARY = "boundary_overlay.png"

#: Name of the cost-aware winner column written by qemsel.experiment.
_COST_AWARE_COLUMN = "best_technique_cost_aware"

#: Rows with cdr_abs_error below this are pre-fix CDR classical-simulation
#: artifacts (see qemsel.mitigation) — excluded from every aggregate.
_CDR_DEGENERATE_TOL: float = 1e-12

#: Backend names may carry a noise-scale suffix: '<BaseName>@x<scale>'.
_NOISE_SCALE_RE = re.compile(r"^(?P<base>.+)@x(?P<scale>\d+(?:\.\d+)?)$")


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def _fmt(x: object) -> str:
    """Format a number to 3 significant figures; 'n/a' for None/NaN/inf."""
    if x is None:
        return "n/a"
    try:
        val = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not math.isfinite(val):
        return "n/a"
    if val == int(val) and abs(val) < 1e6:
        return str(int(val))
    return f"{val:.3g}"


def _fmt_pm(value: object, std: object) -> str:
    """Format 'value ± std' when a finite std is available, else just value."""
    base = _fmt(value)
    if std is None:
        return base
    try:
        std_val = float(std)
    except (TypeError, ValueError):
        return base
    if not math.isfinite(std_val):
        return base
    return f"{base} ± {std_val:.2g}"


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    """Render a GitHub-flavoured markdown table."""
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)


def _detect_techniques(df: pd.DataFrame) -> list[str]:
    """Techniques present in df, inferred from ``<tech>_abs_error`` columns.

    The seed-averaged aggregated.csv schema stores ``<tech>_mean_abs_error``
    instead — the ``_mean`` infix is stripped so 'cdr_mean' is never
    mis-detected as a technique (fixer pass 2026-07-21; previously an
    aggregated.csv rendered with zero valid winners).
    """
    suffix = "_abs_error"
    found = [c[: -len(suffix)] for c in df.columns if c.endswith(suffix)]
    found = list(
        dict.fromkeys(t[: -len("_mean")] if t.endswith("_mean") else t for t in found)
    )
    ordered = [t for t in _CANONICAL_TECHNIQUES if t in found]
    ordered += sorted(t for t in found if t not in _CANONICAL_TECHNIQUES)
    return ordered


def _parse_backend(backend: str) -> tuple[str, float]:
    """Split a backend name into (base_name, noise_scale).

    ``'FakeManilaV2@x1.5' -> ('FakeManilaV2', 1.5)``;
    plain names -> scale 1.0.
    """
    m = _NOISE_SCALE_RE.match(str(backend))
    if m:
        return m.group("base"), float(m.group("scale"))
    return str(backend), 1.0


def _noise_scales(df: pd.DataFrame) -> pd.Series:
    """Per-row noise scale parsed from the backend column."""
    return df["backend"].map(lambda b: _parse_backend(b)[1])


def _validate(df: pd.DataFrame, model_metrics: dict) -> list[str]:
    """Raise ValueError on schema violations; return detected techniques."""
    missing_cols = [c for c in _REQUIRED_DF_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"results DataFrame missing required columns: {missing_cols}")
    techniques = _detect_techniques(df)
    if not techniques:
        raise ValueError(
            "results DataFrame has no '<technique>_abs_error' columns — not an "
            "experiment results table"
        )
    if len(df) == 0:
        raise ValueError("results DataFrame is empty")
    if not isinstance(model_metrics, dict):
        raise ValueError("model_metrics must be a dict (train_and_eval return schema)")
    missing_keys = [k for k in _REQUIRED_METRIC_KEYS if k not in model_metrics]
    if missing_keys:
        raise ValueError(f"model_metrics missing required keys: {missing_keys}")
    return techniques


# ---------------------------------------------------------------------------
# figures (Agg backend; every figure closed after save)
# ---------------------------------------------------------------------------

def _save_error_by_technique(
    df: pd.DataFrame, techniques: list[str], backends: list[str], path: Path
) -> None:
    """Bar chart: mean abs_error per technique, grouped by backend."""
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    x = np.arange(len(techniques), dtype=float)
    width = 0.8 / max(len(backends), 1)
    for i, backend in enumerate(backends):
        sub = df[df["backend"] == backend]
        means = [
            float(sub[f"{t}_abs_error"].mean())
            if f"{t}_abs_error" in sub.columns
            else float("nan")
            for t in techniques
        ]
        ax.bar(x + i * width, means, width, label=backend)
    ax.set_xticks(x + (len(backends) - 1) * width / 2.0)
    ax.set_xticklabels(techniques)
    ax.set_xlabel("technique")
    ax.set_ylabel("mean |error|")
    ax.set_title("Mean absolute expectation error by technique")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _save_win_rate(
    df: pd.DataFrame, techniques: list[str], backends: list[str], path: Path
) -> None:
    """Bar chart: fraction of rows where each technique wins, per backend."""
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    x = np.arange(len(techniques), dtype=float)
    width = 0.8 / max(len(backends), 1)
    best = df["best_technique"].fillna("")
    for i, backend in enumerate(backends):
        mask = df["backend"] == backend
        n = int(mask.sum())
        rates = [
            float(((best == t) & mask).sum()) / n if n else 0.0 for t in techniques
        ]
        ax.bar(x + i * width, rates, width, label=backend)
    ax.set_xticks(x + (len(backends) - 1) * width / 2.0)
    ax.set_xticklabels(techniques)
    ax.set_xlabel("technique")
    ax.set_ylabel("win rate (fraction of rows best)")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("How often each technique gives the lowest error")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _save_winner_vs_noise(
    df: pd.DataFrame, techniques: list[str], path: Path
) -> bool:
    """THE money plot: win rate + mean |error| per technique vs noise scale.

    Parses the '@x<scale>' suffix from the backend column (plain names =
    scale 1.0). Two stacked panels sharing the x axis:
      top    — win rate of each technique at each noise scale (line+marker)
      bottom — mean abs_error of each technique at each noise scale.
    Returns True when the figure was written; False (nothing written) when
    fewer than 2 distinct scales exist — a single-scale dataset has no
    sweep to plot.
    """
    scales = _noise_scales(df)
    unique_scales = sorted(scales.unique())
    if len(unique_scales) < 2:
        return False

    best = df["best_technique"].fillna("")
    fig, (ax_win, ax_err) = plt.subplots(
        2, 1, figsize=(7.5, 7.5), sharex=True
    )
    for tech in techniques:
        win_rates: list[float] = []
        mean_errs: list[float] = []
        for s in unique_scales:
            mask = scales == s
            valid = best[mask].isin(techniques)
            n_valid = int(valid.sum())
            wins = int(((best[mask] == tech) & valid).sum())
            win_rates.append(wins / n_valid if n_valid else float("nan"))
            col = f"{tech}_abs_error"
            if col in df.columns:
                vals = df.loc[mask, col].dropna()
                mean_errs.append(float(vals.mean()) if len(vals) else float("nan"))
            else:
                mean_errs.append(float("nan"))
        ax_win.plot(unique_scales, win_rates, marker="o", label=tech)
        ax_err.plot(unique_scales, mean_errs, marker="o", label=tech)
    ax_win.set_ylabel("win rate")
    ax_win.set_ylim(-0.02, 1.02)
    ax_win.set_title("Does more noise change the best technique?")
    ax_win.legend(fontsize=8)
    ax_win.grid(True, alpha=0.3)
    ax_err.set_xlabel(
        "NOMINAL noise scale (backend '@x<scale>'; plain = 1.0; caps can "
        "compress realized rates - see report section 5)"
    )
    ax_err.set_ylabel("mean |error|")
    ax_err.legend(fontsize=8)
    ax_err.grid(True, alpha=0.3)
    ax_err.set_xticks(unique_scales)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def _save_confusion_matrix(model_metrics: dict, path: Path) -> None:
    """Heatmap of model_metrics['confusion_matrix'] with ['labels']."""
    cm = np.asarray(model_metrics["confusion_matrix"], dtype=float)
    labels = [str(label) for label in model_metrics["labels"]]
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    if cm.size == 0 or cm.ndim != 2:
        ax.text(0.5, 0.5, "no confusion-matrix data", ha="center", va="center")
        ax.set_axis_off()
    else:
        im = ax.imshow(cm, cmap="Blues")
        fig.colorbar(im, ax=ax, fraction=0.046)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
        thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
        for r in range(cm.shape[0]):
            for c in range(cm.shape[1]):
                ax.text(
                    c,
                    r,
                    _fmt(cm[r, c]),
                    ha="center",
                    va="center",
                    color="white" if cm[r, c] > thresh else "black",
                    fontsize=9,
                )
        ax.set_xlabel("predicted technique")
        ax.set_ylabel("true best technique")
    ax.set_title("Recommender confusion matrix (out-of-fold)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _save_feature_importances(model_metrics: dict, path: Path) -> None:
    """Sorted horizontal bar chart of feature importances."""
    importances = dict(model_metrics["feature_importances"])
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    if not importances:
        ax.text(0.5, 0.5, "no feature-importance data", ha="center", va="center")
        ax.set_axis_off()
    else:
        items = sorted(importances.items(), key=lambda kv: kv[1], reverse=True)
        names = [k for k, _ in items][::-1]  # biggest on top of barh
        values = [float(v) for _, v in items][::-1]
        ax.barh(range(len(names)), values, color="#4c72b0")
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlabel("importance")
    note = model_metrics.get("feature_importances_note")
    title = "Feature importances (best model, permutation)"
    if note:
        title += f"\n{note}"
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# markdown sections
# ---------------------------------------------------------------------------

def _conditioning_disclosure(
    df: pd.DataFrame, run_config: dict | None
) -> str | None:
    """Circuit-selection conditioning disclosure line for section 1.

    Returns the disclosure bullet when the run config sets
    ``min_abs_ideal`` > 0 (checked at both the experiment level and inside
    ``config['circuits']``), or — config unavailable — when the data
    carries bumped rejection-sampling seeds (seed >= SUB_SEED_STRIDE);
    None otherwise. This is the user-facing transfer caveat for the
    ``generate_suite`` rejection sampling (fixer pass 2026-07-21): the
    accepted circuits are a systematically atypical high-|<Z^n>| subset.
    """
    threshold = 0.0
    if isinstance(run_config, dict):
        sources = [run_config]
        circuits_cfg = run_config.get("circuits")
        if isinstance(circuits_cfg, dict):
            sources.append(circuits_cfg)
        for source in sources:
            value = source.get("min_abs_ideal", 0.0)
            try:
                threshold = max(threshold, float(value))
            except (TypeError, ValueError):
                pass
    bumped = False
    if "seed" in df.columns:
        # Bumped seeds prove rejection sampling ran even without the config
        # (circuits.SUB_SEED_STRIDE — kept in sync by a regression test).
        from qemsel.circuits import SUB_SEED_STRIDE

        seeds = pd.to_numeric(df["seed"], errors="coerce")
        bumped = bool((seeds >= SUB_SEED_STRIDE).any())
    if threshold <= 0.0 and not bumped:
        return None
    thr_txt = _fmt(threshold) if threshold > 0.0 else "min_abs_ideal (see run_meta.json)"
    return (
        f"- **Circuit-selection conditioning:** random-family circuits were "
        f"rejection-sampled at generation until |<Z^n>| >= {thr_txt} "
        "(bumped seeds appear as seed + k*1000003). The accepted circuits "
        "are therefore an ATYPICAL, high-|ideal| subset of each random "
        "family — the least-scrambling tail for hw_efficient_ansatz / "
        "layered_random (acceptance shrinks with n and depth), and for "
        "near_clifford the minority whose Clifford backbone keeps Z^n "
        "near a stabilizer (the unconditioned median |ideal| there is 0). "
        "Post-conditioning |ideal| magnitudes also differ systematically "
        "across families, which every abs-error comparison inherits. Read "
        "all results as claims about *random circuits conditioned on "
        f"|<Z^n>| >= {thr_txt}*, not about typical random circuits."
    )


def _section_overview(
    df: pd.DataFrame,
    techniques: list[str],
    backends: list[str],
    n_degenerate_cdr: int = 0,
    run_config: dict | None = None,
) -> str:
    families = sorted(df["family"].dropna().unique())
    n_circuits = df["circuit_id"].nunique()
    n_failed = int((df["best_technique"].fillna("") == "").sum())
    lines = [
        "## 1. Overview",
        "",
        f"- Result rows (circuit x backend units): **{len(df)}**",
        f"- Distinct circuits: **{n_circuits}**",
        f"- Circuit families ({len(families)}): {', '.join(families)}",
        f"- Backends ({len(backends)}): {', '.join(backends)}",
        f"- Techniques compared ({len(techniques)}): {', '.join(techniques)}",
        f"- Rows where every technique failed (no best): **{n_failed}**",
    ]
    if "n_seeds" in df.columns:
        n_seeds = pd.to_numeric(df["n_seeds"], errors="coerce")
        lines.append(
            f"- Rows are SEED-AVERAGED (aggregated schema): each row pools "
            f"{_fmt(n_seeds.min())}-{_fmt(n_seeds.max())} seeds "
            f"(mean {_fmt(n_seeds.mean())}) — winner labels are majority/mean "
            "winners, not single-shot lottery outcomes."
        )
    scales = sorted(_noise_scales(df).unique())
    if len(scales) > 1:
        lines.append(
            f"- Noise scales present ({len(scales)}): "
            f"{', '.join('x' + _fmt(s) for s in scales)} "
            "(parsed from the backend '@x<scale>' suffix; see section 5)"
        )
    if n_degenerate_cdr:
        lines.append(
            f"- Rows EXCLUDED as pre-fix CDR classical-simulation artifacts "
            f"(cdr_abs_error < {_CDR_DEGENERATE_TOL:g}): **{n_degenerate_cdr}** "
            "— their 'cdr wins' labels measured classical simulability, not "
            "mitigation quality."
        )
    disclosure = _conditioning_disclosure(df, run_config)
    if disclosure:
        lines.append(disclosure)
    if "n_qubits" in df.columns:
        lines.append(
            f"- Qubit counts: {_fmt(df['n_qubits'].min())}"
            f"-{_fmt(df['n_qubits'].max())}; "
            f"depths: {_fmt(df['depth'].min())}-{_fmt(df['depth'].max())}"
            if "depth" in df.columns
            else f"- Qubit counts: {_fmt(df['n_qubits'].min())}-{_fmt(df['n_qubits'].max())}"
        )
    return "\n".join(lines)


def _section_technique_comparison(
    df: pd.DataFrame, techniques: list[str], backends: list[str]
) -> str:
    rows: list[list[str]] = []
    for backend in backends:
        sub = df[df["backend"] == backend]
        for tech in techniques:
            col = f"{tech}_abs_error"
            series = sub[col] if col in sub.columns else pd.Series(dtype=float)
            n_nan = int(series.isna().sum())
            rows.append(
                [
                    backend,
                    tech,
                    _fmt(series.mean()),
                    _fmt(series.median()),
                    str(n_nan),
                    str(len(sub)),
                ]
            )
    overall_rows: list[list[str]] = []
    for tech in techniques:
        col = f"{tech}_abs_error"
        series = df[col] if col in df.columns else pd.Series(dtype=float)
        overall_rows.append(
            [
                tech,
                _fmt(series.mean()),
                _fmt(series.median()),
                str(int(series.isna().sum())),
            ]
        )
    return "\n".join(
        [
            "## 2. Technique comparison",
            "",
            "Mean and median absolute error of the mitigated (or raw) expectation "
            "value versus the exact ideal value. NaN counts are technique "
            "failures caught by the experiment runner.",
            "",
            "### Per backend",
            "",
            _md_table(
                ["Backend", "Technique", "Mean abs error", "Median abs error", "NaN (failures)", "Rows"],
                rows,
            ),
            "",
            "### Overall (all backends pooled)",
            "",
            _md_table(
                ["Technique", "Mean abs error", "Median abs error", "NaN (failures)"],
                overall_rows,
            ),
            "",
            f"![Mean abs error by technique]({_PNG_ERROR})",
        ]
    )


def _section_cost_normalized(
    df: pd.DataFrame, techniques: list[str]
) -> str:
    mean_shots: dict[str, float] = {}
    for tech in techniques:
        col = f"{tech}_shots"
        mean_shots[tech] = float(df[col].mean()) if col in df.columns else float("nan")
    finite = [v for v in mean_shots.values() if not math.isnan(v) and v > 0]
    base = min(finite) if finite else float("nan")
    rows: list[list[str]] = []
    for tech in techniques:
        err_col = f"{tech}_abs_error"
        err = float(df[err_col].mean()) if err_col in df.columns else float("nan")
        shots = mean_shots[tech]
        rel_cost = shots / base if finite and not math.isnan(shots) else float("nan")
        # SAME cost model as experiment._pick_winners: error * sqrt(relative
        # cost). Extra averaging alone buys a ~1/sqrt(shots) noise reduction,
        # so sqrt is the first-order fair penalty (a linear penalty would
        # over-charge shot-hungry techniques; the report and the CSV column
        # must not disagree — review finding, 2026-07-21).
        cost_weighted = (
            err * math.sqrt(rel_cost)
            if not (math.isnan(err) or math.isnan(rel_cost))
            else float("nan")
        )
        total = (
            float(df[f"{tech}_shots"].sum()) if f"{tech}_shots" in df.columns else float("nan")
        )
        rows.append(
            [tech, _fmt(err), _fmt(shots), _fmt(rel_cost), _fmt(cost_weighted), _fmt(total)]
        )
    lines = [
        "## 3. Cost-normalized view",
        "",
        "Mitigation is not free: each technique consumes a multiple of the "
        "base shot budget (`<tech>_shots` columns). 'Relative cost' is mean "
        "shots divided by the cheapest technique's mean shots; "
        "'cost-weighted error' multiplies mean error by **sqrt(relative "
        "cost)** — the same model used for the per-row "
        "`best_technique_cost_aware` column — so a technique only looks "
        "good here if its accuracy gain beats the ~sqrt(k) shot-noise "
        "reduction the same budget would buy as plain extra averaging.",
        "",
        _md_table(
            [
                "Technique",
                "Mean abs error",
                "Mean shots/row",
                "Relative cost",
                "Cost-weighted error (err x sqrt(rel cost))",
                "Total shots",
            ],
            rows,
        ),
    ]
    if "raw_plus" in techniques:
        lines += [
            "",
            "`raw_plus` is the EMPIRICAL equal-budget baseline: the raw "
            "(unmitigated) circuit run at the maximum shot multiplier's "
            "budget. Where a mitigation technique beats `raw_plus`, its "
            "advantage survives giving the null strategy the same number "
            "of shots — a stronger claim than the analytic sqrt proxy. "
            "Note: in the COST-AWARE label `raw_plus` is a comparison "
            "column, not a realistically reachable class — it pays the "
            "same sqrt(11) shot penalty as CDR while raw's error is bias- "
            "(not shot-noise-) dominated, so extra shots buy it almost "
            "nothing and it (correctly) almost never wins the cost-aware "
            "argmin.",
        ]
    else:
        lines += [
            "",
            "Caveat: the sqrt penalty is an analytic proxy; when errors are "
            "bias-dominated (mean error >> shot noise) extra shots buy almost "
            "nothing and the penalty over-charges mitigation. This dataset "
            "predates the empirical equal-budget baseline (`raw_plus` "
            "column absent).",
        ]

    # Per-row cost-aware winner tables from the stored column (the fair
    # per-row comparison — previously computed but never analyzed).
    if _COST_AWARE_COLUMN in df.columns:
        winner = df[_COST_AWARE_COLUMN].fillna("")
        valid = df[winner.isin(techniques)]
        n_valid = len(valid)
        overall_rows = []
        for tech in techniques:
            wins = int((valid[_COST_AWARE_COLUMN] == tech).sum())
            rate = wins / n_valid if n_valid else 0.0
            overall_rows.append([tech, str(wins), _fmt(rate)])
        family_rows = []
        for fam in sorted(df["family"].dropna().unique()):
            sub = valid[valid["family"] == fam]
            family_rows.append(
                [str(fam)]
                + [
                    str(int((sub[_COST_AWARE_COLUMN] == t).sum()))
                    for t in techniques
                ]
                + [str(len(sub))]
            )
        lines += [
            "",
            "### Cost-aware win rates (equal shot budget, per-row "
            "`best_technique_cost_aware`)",
            "",
            f"A technique 'wins' a row when it minimizes "
            f"abs_error x sqrt(shots / base_shots). Rows counted: {n_valid} "
            f"of {len(df)}.",
            "",
            _md_table(["Technique", "Wins", "Win rate"], overall_rows),
            "",
            "#### Cost-aware winner counts per circuit family",
            "",
            _md_table(["Family"] + techniques + ["Rows"], family_rows),
        ]
    return "\n".join(lines)


def _section_win_rates(
    df: pd.DataFrame, techniques: list[str], backends: list[str]
) -> str:
    best = df["best_technique"].fillna("")
    valid = df[best.isin(techniques)]
    n_valid = len(valid)

    overall_rows = []
    for tech in techniques:
        wins = int((valid["best_technique"] == tech).sum())
        rate = wins / n_valid if n_valid else 0.0
        overall_rows.append([tech, str(wins), _fmt(rate)])

    def _crosstab_rows(group_col: str, groups: list[str]) -> list[list[str]]:
        out = []
        for g in groups:
            sub = valid[valid[group_col] == g]
            out.append(
                [str(g)]
                + [str(int((sub["best_technique"] == t).sum())) for t in techniques]
                + [str(len(sub))]
            )
        return out

    families = sorted(df["family"].dropna().unique())
    return "\n".join(
        [
            "## 4. Win rates",
            "",
            f"A technique 'wins' a row when it has the smallest non-NaN absolute "
            f"error. Rows counted: {n_valid} of {len(df)}.",
            "",
            "### Overall",
            "",
            _md_table(["Technique", "Wins", "Win rate"], overall_rows),
            "",
            "### Winner counts per circuit family",
            "",
            _md_table(
                ["Family"] + techniques + ["Rows"], _crosstab_rows("family", families)
            ),
            "",
            "### Winner counts per backend",
            "",
            _md_table(
                ["Backend"] + techniques + ["Rows"], _crosstab_rows("backend", backends)
            ),
            "",
            f"![Win rate per backend]({_PNG_WIN})",
        ]
    )


def _section_noise_sweep(
    df: pd.DataFrame, techniques: list[str], noise_png_written: bool
) -> str:
    """Section 5: win rate + mean error per technique vs parsed noise scale."""
    scales = _noise_scales(df)
    unique_scales = sorted(scales.unique())
    header = [
        "## 5. Noise-scale sweep",
        "",
        "Backend names may carry an `@x<scale>` suffix (e.g. "
        "`FakeManilaV2@x1.5` = the Manila noise model with all error rates "
        "scaled by 1.5); plain names mean scale 1.0. This section pools rows "
        "by that parsed scale to answer the study's core research question: "
        "**does increasing the noise level change WHICH mitigation technique "
        "is best?**",
        "",
    ]
    if len(unique_scales) < 2:
        return "\n".join(
            header
            + [
                f"All {len(df)} rows share a single noise scale "
                f"(x{_fmt(unique_scales[0]) if unique_scales else '1'}) — no "
                "`@x<scale>`-scaled backends in this dataset, so the sweep is "
                "not applicable. Re-run the experiment with `noise_scales` "
                "configured to populate this section.",
            ]
        )

    best = df["best_technique"].fillna("")
    base_names = sorted({_parse_backend(b)[0] for b in df["backend"].dropna()})

    win_rows: list[list[str]] = []
    err_rows: list[list[str]] = []
    for s in unique_scales:
        mask = scales == s
        sub = df[mask]
        valid = sub[best[mask].isin(techniques)]
        n_valid = len(valid)
        rates: dict[str, float] = {}
        win_cells: list[str] = []
        for tech in techniques:
            wins = int((valid["best_technique"] == tech).sum())
            rate = wins / n_valid if n_valid else float("nan")
            rates[tech] = rate
            win_cells.append(f"{wins} ({_fmt(rate)})")
        top = max(rates, key=lambda t: (rates[t] if not math.isnan(rates[t]) else -1))
        win_rows.append(
            [f"x{_fmt(s)}", str(n_valid)] + win_cells + [top]
        )
        err_cells = []
        for tech in techniques:
            col = f"{tech}_abs_error"
            if col in sub.columns:
                vals = sub[col].dropna()
                err_cells.append(_fmt(vals.mean()) if len(vals) else "n/a")
            else:
                err_cells.append("n/a")
        err_rows.append([f"x{_fmt(s)}"] + err_cells)

    lines = header + [
        f"Noise scales found ({len(unique_scales)}): "
        f"{', '.join('x' + _fmt(s) for s in unique_scales)}; base backends "
        f"pooled per scale: {', '.join(base_names)}.",
        "",
        "### Win rate per technique vs noise scale",
        "",
        "Cells are `wins (win rate)` among rows with a valid winner at that "
        "scale.",
        "",
        _md_table(
            ["Noise scale", "Rows"] + techniques + ["Top technique"], win_rows
        ),
        "",
        "### Mean abs error per technique vs noise scale",
        "",
        _md_table(["Noise scale"] + techniques, err_rows),
    ]

    # ---- per-scale device composition + realized noise levels -------------
    # (fixer pass 2026-07-21) The '@x<scale>' suffix is a NOMINAL dial: the
    # scaled model caps each error rate (gate 0.9, readout 0.45), so on a
    # cap-saturated device the realized average error grows by LESS than
    # <scale>. The features carry the realized (scaled-and-capped) values —
    # print them so winners can be read against realized noise, not the
    # nominal dial. Also print which devices make up each scale pool:
    # unequal compositions confound scale with device.
    comp_rows: list[list[str]] = []
    scale_devices: dict[float, set[str]] = {}
    have_feats = (
        "feat_backend_avg_2q_error" in df.columns
        and "feat_backend_avg_readout_error" in df.columns
    )
    for backend in sorted(
        df["backend"].dropna().unique(),
        key=lambda b: (_parse_backend(b)[1], _parse_backend(b)[0]),
    ):
        base, scale = _parse_backend(backend)
        scale_devices.setdefault(scale, set()).add(base)
        sub = df[df["backend"] == backend]
        comp_rows.append(
            [
                str(backend),
                str(base),
                f"x{_fmt(scale)}",
                _fmt(sub["feat_backend_avg_2q_error"].mean()) if have_feats else "n/a",
                _fmt(sub["feat_backend_avg_readout_error"].mean())
                if have_feats
                else "n/a",
                str(len(sub)),
            ]
        )
    lines += [
        "",
        "### Per-scale device composition and realized noise levels",
        "",
        _md_table(
            [
                "Backend",
                "Base device",
                "Nominal scale",
                "Realized avg 2q error",
                "Realized avg readout error",
                "Rows",
            ],
            comp_rows,
        ),
    ]
    compositions = {frozenset(devs) for devs in scale_devices.values()}
    if len(compositions) > 1:
        comp_txt = "; ".join(
            f"x{_fmt(s)}: {', '.join(sorted(devs))}"
            for s, devs in sorted(scale_devices.items())
        )
        lines += [
            "",
            "**Warning — unequal device composition across scales** "
            f"({comp_txt}): in the pooled per-scale tables above, scale "
            "effects are CONFOUNDED with device effects. Compare scales "
            "within one device, or use a config where every device "
            "contributes every scale.",
        ]
    lines += [
        "",
        "Two caveats when reading this sweep (fixer pass 2026-07-21):",
        "",
        "1. **Nominal vs realized scale.** `@x<scale>` multiplies each "
        "calibrated error rate but CAPS the result (gate 0.9, readout "
        "0.45). On a device whose calibration already sits at/above a cap "
        "the dial compresses and can locally invert: FakeLagosV2 stores "
        "46.4% readout error on q2 (above the 45% cap), so its realized "
        "average readout error is only ~1.28x plain at nominal x1.5 and "
        "~1.44x at x2.0, q2's error DECREASES from x1.0 (0.464) to x1.5 "
        "(0.45), and max_readout_error is non-monotone in scale. "
        "REM-vs-scale trends on such devices are partly cap artifacts — "
        "read winners against the realized columns above, not the nominal "
        "dial.",
        "2. **Scale 1.0 differs in KIND, not just strength.** Plain (x1.0) "
        "rows run the full `AerSimulator.from_backend` noise model "
        "(composite thermal-relaxation channels); scaled rows run a "
        "synthetic depolarizing + symmetric-readout model built from the "
        "same calibration. The x1.0 -> x1.5 step therefore changes noise "
        "CHARACTER as well as level (ZNE is the technique most sensitive "
        "to this); readout stays symmetric on both paths, so REM is "
        "comparable throughout.",
    ]
    if noise_png_written:
        lines += [
            "",
            f"![Win rate and mean error vs noise scale]({_PNG_NOISE})",
        ]
    return "\n".join(lines)


def _label_summary_rows(metrics: dict) -> dict[str, str]:
    """Flatten one train_and_eval metrics dict into display strings."""
    per_model = metrics.get("per_model", {})
    best = metrics.get("best_model_name", "n/a")
    best_std = per_model.get(best, {}).get("accuracy_std")
    lofo = metrics.get("lofo") if isinstance(metrics.get("lofo"), dict) else {}
    lobo = metrics.get("lobo") if isinstance(metrics.get("lobo"), dict) else {}
    lodo = metrics.get("lodo") if isinstance(metrics.get("lodo"), dict) else {}
    dropped = metrics.get("dropped_classes") or []
    return {
        "Best model": str(best),
        "CV accuracy (± fold std)": _fmt_pm(metrics.get("accuracy"), best_std),
        "CV macro F1": _fmt(metrics.get("macro_f1")),
        "Baseline accuracy": _fmt(metrics.get("baseline_accuracy")),
        "CV folds": _fmt(metrics.get("cv_folds")),
        "Samples (refit)": _fmt(metrics.get("n_samples")),
        "Samples in CV": _fmt(metrics.get("cv_n_samples", metrics.get("n_samples"))),
        "Classes dropped from CV": ", ".join(dropped) if dropped else "none",
        "LOFO accuracy (new family)": _fmt(lofo.get("accuracy")),
        "LOFO macro F1": _fmt(lofo.get("macro_f1")),
        "LOBO accuracy (scale interpolation)": _fmt(lobo.get("accuracy")),
        "LOBO macro F1": _fmt(lobo.get("macro_f1")),
        "LODO accuracy (new device)": _fmt(lodo.get("accuracy")),
        "LODO macro F1": _fmt(lodo.get("macro_f1")),
        "Classes": ", ".join(str(c) for c in metrics.get("labels", [])) or "n/a",
    }


def _holdout_table(holdout: dict, unit: str) -> str:
    """Markdown table for a lofo/lobo dict: per-group accuracy + macro-F1."""
    acc_key = f"per_{unit}_accuracy"
    f1_key = f"per_{unit}_macro_f1"
    per_acc = holdout.get(acc_key, {}) or {}
    per_f1 = holdout.get(f1_key, {}) or {}
    rows = [
        [str(g), _fmt(a), _fmt(per_f1.get(g)) if per_f1 else "n/a"]
        for g, a in sorted(per_acc.items())
    ]
    return _md_table([f"Held-out {unit}", "Accuracy", "Macro F1"], rows)


def _section_model_eval(
    model_metrics: dict, cost_aware_metrics: dict | None = None
) -> str:
    per_model = model_metrics.get("per_model", {})
    per_model_rows = [
        [
            name,
            _fmt_pm(m.get("accuracy"), m.get("accuracy_std")),
            _fmt(m.get("macro_f1")),
        ]
        for name, m in per_model.items()
    ]
    cv_folds = model_metrics.get("cv_folds", "n/a")
    best_name = model_metrics["best_model_name"]
    best_std = per_model.get(best_name, {}).get("accuracy_std")
    baseline_std = per_model.get("dummy_majority", {}).get("accuracy_std")
    label_column = model_metrics.get("label_column", "best_technique")
    cv_grouping = model_metrics.get("cv_grouping")
    n_samples = model_metrics.get("n_samples", "n/a")
    cv_n_samples = model_metrics.get("cv_n_samples")
    sample_note = f"n_samples = {n_samples}"
    if cv_n_samples is not None and cv_n_samples != n_samples:
        sample_note += f" ({cv_n_samples} in CV after class drops)"
    lines = [
        "## 6. Model evaluation",
        "",
        f"Best model: **{best_name}** "
        f"({sample_note}, "
        f"cv_folds = {cv_folds}"
        + (f", folds grouped by circuit config: {cv_grouping}" if cv_grouping else "")
        + f", training label: `{label_column}`).",
        "",
        "Accuracies are shown as mean ± std over CV folds (std of per-fold "
        "accuracies, ddof=1).",
        "",
        _md_table(
            ["Metric", "Best model", "Majority-class baseline"],
            [
                [
                    "Accuracy",
                    _fmt_pm(model_metrics["accuracy"], best_std),
                    _fmt_pm(model_metrics["baseline_accuracy"], baseline_std),
                ],
                ["Macro F1", _fmt(model_metrics["macro_f1"]), "n/a"],
            ],
        ),
        "",
    ]
    dropped = model_metrics.get("dropped_classes") or []
    if dropped:
        lines += [
            f"**Note — classes dropped from CV:** {', '.join(dropped)} had "
            "fewer than 2 members, so their rows were excluded from the "
            "cross-validated metrics above (a singleton can never appear in "
            "both a train and a test fold). They ARE included in the final "
            "refit model and in the LOFO/LOBO evaluations below, and remain "
            "recommendable classes.",
            "",
        ]
    lofo = model_metrics.get("lofo")
    if isinstance(lofo, dict):
        lines += [
            "### Generalization to a NEW circuit family (leave-one-family-out)",
            "",
            "The headline 'new circuit' number: each family is predicted by "
            "a model that never saw ANY circuit of that family. This is the "
            "honest proxy for the project's claim of recommending for a new "
            "circuit; the grouped-CV numbers above answer the weaker "
            "'new configuration of a known family' question.",
            "",
            f"- LOFO pooled accuracy: **{_fmt(lofo.get('accuracy'))}**, "
            f"pooled macro F1: **{_fmt(lofo.get('macro_f1'))}** "
            f"({lofo.get('n_families', 'n/a')} families)",
            "",
            _holdout_table(lofo, "family"),
            "",
        ]
    lobo = model_metrics.get("lobo")
    if isinstance(lobo, dict):
        lines += [
            "### Noise-level interpolation (leave-one-backend-out)",
            "",
            "Each backend STRING (noise-scaled `@x<scale>` variants are "
            "separate strings) is predicted by a model that never saw rows "
            "from that exact string. CAUTION: when other scales of the SAME "
            "device stay in training (e.g. `FakeManilaV2@x1.5` held out "
            "while `FakeManilaV2` and `FakeManilaV2@x2.0` remain), the fold "
            "measures noise-level interpolation on a known device — its "
            "backend features are bracketed by the training siblings. Do "
            "NOT quote this as 'generalizes to a new noise environment'; "
            "that claim belongs to the leave-one-device-out number below.",
            "",
            f"- LOBO pooled accuracy: **{_fmt(lobo.get('accuracy'))}**, "
            f"pooled macro F1: **{_fmt(lobo.get('macro_f1'))}** "
            f"({lobo.get('n_backends', 'n/a')} backends)",
            "",
            _holdout_table(lobo, "backend"),
            "",
        ]
    lodo = model_metrics.get("lodo")
    if isinstance(lodo, dict):
        lines += [
            "### Generalization to a NEW noise environment (leave-one-device-out)",
            "",
            "The headline 'new noise environment' number: ALL noise scales "
            "of one base device are held out together, so the model has "
            "never seen the held-out device at any scale.",
            "",
            f"- LODO pooled accuracy: **{_fmt(lodo.get('accuracy'))}**, "
            f"pooled macro F1: **{_fmt(lodo.get('macro_f1'))}** "
            f"({lodo.get('n_devices', 'n/a')} devices)",
            "",
            _holdout_table(lodo, "device"),
            "",
        ]
    if per_model_rows:
        lines += [
            "### Per-model cross-validated metrics",
            "",
            _md_table(["Model", "Accuracy (± fold std)", "Macro F1"], per_model_rows),
            "",
        ]
    if isinstance(cost_aware_metrics, dict):
        primary_col = _label_summary_rows(model_metrics)
        cost_col = _label_summary_rows(cost_aware_metrics)
        side_rows = [
            [metric, primary_col[metric], cost_col.get(metric, "n/a")]
            for metric in primary_col
        ]
        lines += [
            "### Both winner labels side by side",
            "",
            "`best_technique` optimizes pure accuracy at ANY shot cost; "
            "`best_technique_cost_aware` charges each technique "
            "sqrt(relative shot cost) first — the winner an equal-budget "
            "user should pick. Bundles: `model.joblib` (accuracy-only) and "
            "`model_cost_aware.joblib` (cost-aware).",
            "",
            _md_table(
                [
                    "Metric",
                    "`best_technique` (accuracy-only)",
                    "`best_technique_cost_aware` (equal budget)",
                ],
                side_rows,
            ),
            "",
        ]
    if cv_folds == 0:
        lines += [
            "**Warning:** cv_folds = 0 — even after excluding singleton "
            "classes there were < 2 usable classes/groups, so metrics were "
            "computed on the training set (optimistic).",
            "",
        ]
    importances_note = model_metrics.get("feature_importances_note")
    if importances_note:
        lines += [
            f"Feature importances: permutation importance, {importances_note}.",
            "",
        ]
    lines += [
        f"![Confusion matrix]({_PNG_CONFUSION})",
        "",
        f"![Feature importances]({_PNG_IMPORTANCES})",
    ]
    return "\n".join(lines)


def _section_reproducibility(model_metrics: dict) -> str:
    import qemsel

    packages = [
        "qiskit",
        "qiskit-aer",
        "qiskit-ibm-runtime",
        "mitiq",
        "numpy",
        "scipy",
        "scikit-learn",
        "pandas",
        "matplotlib",
        "joblib",
    ]
    version_rows: list[list[str]] = [["qemsel", str(qemsel.__version__)]]
    from importlib.metadata import PackageNotFoundError, version

    for pkg in packages:
        try:
            version_rows.append([pkg, version(pkg)])
        except PackageNotFoundError:
            version_rows.append([pkg, "not installed"])
    return "\n".join(
        [
            "## 7. Reproducibility",
            "",
            "- The exact experiment configuration, package versions and "
            "timestamp of the data-generating run are stored in "
            "`run_meta.json` next to `results.csv` in the experiment output "
            "directory.",
            "- All circuit generators, executors and mitigation calls are "
            "seeded; rerunning with the same config reproduces the dataset.",
            f"- Model bundle version: qemsel "
            f"{model_metrics.get('qemsel_version', qemsel.__version__)}.",
            "",
            "### Package versions (report environment)",
            "",
            _md_table(["Package", "Version"], version_rows),
        ]
    )


# ---------------------------------------------------------------------------
# V2 sections (builder-recommend / B8): statistical hygiene + boundary overlay
# ---------------------------------------------------------------------------

def _humanize_comparison(name: str) -> str:
    """Turn a stats.json comparison key into readable prose.

    ``'raw_plus_vs_raw' -> 'raw_plus vs raw'``;
    ``'top2_zne_vs_cdr' -> 'top-2: zne vs cdr'``.
    """
    if name.startswith("top2_"):
        return "top-2: " + name[len("top2_"):].replace("_vs_", " vs ")
    return name.replace("_vs_", " vs ")


def _fmt_per_tech(mapping: object) -> str:
    """Compact 'tech: value, ...' rendering of a per-technique dict.

    None (a check that does not apply to this schema) renders 'n/a (schema)'.
    """
    if mapping is None:
        return "n/a (schema)"
    if not isinstance(mapping, dict) or not mapping:
        return "n/a"
    ordered = [t for t in _CANONICAL_TECHNIQUES if t in mapping]
    ordered += [t for t in mapping if t not in _CANONICAL_TECHNIQUES]
    return ", ".join(f"{t}: {_fmt(mapping[t])}" for t in ordered)


def _koester_lines(checklist: dict) -> list[str]:
    """Render the Koester-Mauerer checklist dict as a pass/flag block.

    Any failing gate (overall ``passed`` False, an argmin mismatch, or a
    non-zero partial-coverage count) is rendered in **bold**.
    """
    checks = checklist.get("checks", {}) if isinstance(checklist, dict) else {}
    passed = checklist.get("passed")
    schema = checklist.get("schema", "?")
    n_rows = checklist.get("n_rows", "?")
    overall = "PASS" if passed else "FAIL"

    argmin = checks.get("label_argmin_consistent") or {}
    n_mismatch = argmin.get("n_mismatch")
    n_checked = argmin.get("n_checked")
    argmin_flag = "**FLAG**" if (n_mismatch or 0) else "ok"

    margin = checks.get("winner_margin_below_k_sigma") or {}
    partial = checks.get("partial_coverage_winners")
    partial_flag = (
        "n/a (per-seed)"
        if partial is None
        else ("**FLAG**" if partial else "ok")
    )

    rows = [
        [
            "label_argmin_consistent",
            f"n_mismatch = {_fmt(n_mismatch)} / n_checked = {_fmt(n_checked)}",
            argmin_flag,
        ],
        [
            "winner_margin_below_k_sigma",
            f"n_flagged = {_fmt(margin.get('n_flagged'))} "
            f"(fraction {_fmt(margin.get('fraction'))}, "
            f"k_sigma = {_fmt(margin.get('k_sigma'))})",
            "info (statistical ties)",
        ],
        [
            "partial_coverage_winners",
            _fmt(partial) if partial is not None else "n/a (per-seed schema)",
            partial_flag,
        ],
        [
            "nan_rate",
            _fmt_per_tech(checks.get("nan_rate")),
            "info",
        ],
        [
            "overshoot_beyond_physical_max",
            _fmt_per_tech(checks.get("overshoot_beyond_physical_max")),
            "info",
        ],
        [
            "error_beyond_physical_max",
            _fmt_per_tech(checks.get("error_beyond_physical_max")),
            "info",
        ],
    ]
    return [
        "### Koester-Mauerer statistical checklist",
        "",
        f"Schema: `{schema}`; rows checked: {n_rows}. Overall verdict: "
        f"**{overall}**. "
        "The two GATING checks are `label_argmin_consistent` (the stored "
        "`best_technique` must equal the per-row argmin over the error "
        "columns) and `partial_coverage_winners` (aggregated schema only — a "
        "winner must not be decided on fewer seeds than the row pools); a "
        "flagged gate fails the run. The remaining rows are informational "
        "(they report, they do not censor): `winner_margin_below_k_sigma` "
        "counts labels that are statistical ties, and the physical-range "
        "checks count variance blow-ups.",
        "",
        _md_table(["Check", "Result", "Flag"], rows),
        "",
    ]


def _section_stats(stats_results: dict) -> str:
    """Section 8: statistical hygiene from the ``compute_stats`` dict.

    Renders per-technique win-share bootstrap CIs (each label column
    present), the paired permutation tests (raw_plus-vs-raw + the top-2
    contest), Cliff's-delta effect sizes, and the Koester checklist as a
    pass/flag table. Every field is read with ``.get`` so a partial dict
    degrades gracefully instead of raising.
    """
    n_rows = stats_results.get("n_rows")
    data_path = stats_results.get("data_path")
    lines = [
        "## 8. Statistical hygiene",
        "",
        "Reviewer-bar statistics (the Koester-Mauerer checklist) computed by "
        "`scripts/compute_stats.py` over "
        f"{('`' + str(data_path) + '`') if data_path else 'the results table'}"
        f"{f' ({n_rows} rows)' if n_rows is not None else ''}. Confidence "
        "intervals are percentile bootstraps; the paired tests are "
        "sign-flip permutation tests on the per-row error differences. These "
        "quantify how much of each headline win-share and each head-to-head "
        "gap survives sampling noise.",
        "",
    ]

    win_ci = stats_results.get("win_share_ci") or {}
    if win_ci:
        lines += [
            "### Win-share bootstrap confidence intervals",
            "",
            "Fraction of rows each technique wins, with a bootstrap CI over "
            "the resampled row set. A CI that straddles a rival's estimate "
            "means the win-share ordering is not resolved at this sample "
            "size.",
            "",
        ]
        for label_col, per_tech in win_ci.items():
            lines += [f"**Label column `{label_col}`**", ""]
            rows = []
            for tech, d in (per_tech or {}).items():
                d = d or {}
                rows.append(
                    [
                        tech,
                        _fmt(d.get("estimate")),
                        f"[{_fmt(d.get('lo'))}, {_fmt(d.get('hi'))}]",
                        _fmt(d.get("ci")),
                        _fmt(d.get("n")),
                    ]
                )
            lines += [
                _md_table(
                    ["Technique", "Win share", "Bootstrap CI", "CI level", "n"],
                    rows,
                ),
                "",
            ]

    paired = stats_results.get("paired_tests") or {}
    if paired:
        lines += [
            "### Paired permutation tests",
            "",
            "Sign-flip permutation test on the mean paired error difference "
            "(negative mean diff = the first-named technique has the smaller "
            "error). `raw_plus_vs_raw` checks whether equal-budget extra "
            "averaging alone (`raw_plus`) beats plain `raw`; the `top2` row "
            "tests the two techniques with the highest win shares against "
            "each other.",
            "",
        ]
        rows = []
        for name, d in paired.items():
            d = d or {}
            rows.append(
                [
                    _humanize_comparison(name),
                    _fmt(d.get("mean_diff")),
                    _fmt(d.get("p_value")),
                    _fmt(d.get("n_pairs")),
                    str(d.get("alternative", "n/a")),
                ]
            )
        lines += [
            _md_table(
                ["Comparison", "Mean paired diff", "p-value", "n pairs", "alternative"],
                rows,
            ),
            "",
        ]

    effect = stats_results.get("effect_sizes") or {}
    if effect:
        lines += [
            "### Effect sizes (Cliff's delta)",
            "",
            "Cliff's delta in [-1, 1] over all error pairs; NEGATIVE means the "
            "first-named technique tends to the SMALLER error (better). A "
            "significant p-value with a near-zero delta is a real but tiny "
            "effect.",
            "",
            _md_table(
                ["Comparison", "Cliff's delta"],
                [
                    [_humanize_comparison(name), _fmt(val)]
                    for name, val in effect.items()
                ],
            ),
            "",
        ]

    checklist = stats_results.get("checklist") or {}
    if checklist:
        lines += _koester_lines(checklist)

    return "\n".join(lines).rstrip("\n")


def _resolve_overlay_figure(plot_path: object, out_dir: Path) -> str:
    """Validate the overlay figure sits inside ``out_dir`` and return its
    out_dir-relative posix name for embedding.

    Raises:
        ValueError: plot_path missing, resolving outside out_dir, or the file
            does not exist there.
    """
    if not plot_path:
        raise ValueError(
            "boundary_overlay is missing 'plot_path' — cannot embed the "
            "overlay figure"
        )
    p = Path(str(plot_path))
    out_res = out_dir.resolve()
    candidate = (p if p.is_absolute() else (out_dir / p)).resolve()
    try:
        rel = candidate.relative_to(out_res).as_posix()
    except ValueError as exc:
        raise ValueError(
            f"boundary overlay figure must sit inside the report out_dir "
            f"({out_dir}); plot_path={plot_path!r} resolves to {candidate}, "
            "outside it. Run overlay_selector_vs_theory with out_dir set to "
            "the report directory so the PNG lands next to report.md."
        ) from exc
    if not candidate.exists():
        raise ValueError(
            f"boundary overlay figure referenced by plot_path ({plot_path}) "
            f"does not exist at {candidate}; run "
            "qemsel.boundary.overlay_selector_vs_theory first"
        )
    return rel


def _section_boundary_overlay(boundary_overlay: dict, out_dir: Path) -> str:
    """Section 9: the Angle-3 selector-vs-theory ZNE help-harm overlay.

    Embeds the figure (validated to live inside ``out_dir``), the agreement
    /IoU/region-share numbers, and the mandatory caveats block.
    """
    rel = _resolve_overlay_figure(boundary_overlay.get("plot_path"), out_dir)
    zne_labels = boundary_overlay.get("zne_labels") or []
    metric_rows = [
        ["Grid points evaluated", _fmt(boundary_overlay.get("n_points"))],
        [
            "Selector-vs-theory agreement",
            f"{_fmt(boundary_overlay.get('agreement_pct'))}%",
        ],
        ["IoU of the two 'help' regions", _fmt(boundary_overlay.get("iou_help"))],
        [
            "Selector 'use ZNE' share",
            _fmt(boundary_overlay.get("selector_help_share")),
        ],
        ["Theory 'help' share", _fmt(boundary_overlay.get("theory_help_share"))],
        ["Noise (eps) feature", str(boundary_overlay.get("eps_feature"))],
        [
            "Predicted classes counted as 'use ZNE'",
            ", ".join(str(x) for x in zne_labels) or "n/a",
        ],
    ]
    return "\n".join(
        [
            "## 9. ZNE help-harm boundary overlay",
            "",
            "The paper's Angle-3 centerpiece: the recommender's LEARNED "
            "ZNE-refusal region (feature-only, never sees the ideal value) "
            "overlaid on the ANALYTIC finite-shot help-harm boundary of "
            "Scavino (arXiv:2605.08251), in the (noise strength x shot "
            "budget) plane. Agreement is the fraction of grid points where "
            "the selector's 'use ZNE' decision matches the theory's 'help' "
            "verdict; IoU is the overlap of the two help regions.",
            "",
            _md_table(["Metric", "Value"], metric_rows),
            "",
            f"![Selector ZNE-refusal region vs analytic boundary]({rel})",
            "",
            "**Reading this overlay (mandatory caveats):**",
            "",
            "1. **Realized, not nominal, noise axis.** The eps axis uses the "
            "REALIZED backend error rates from `get_backend_info` "
            f"(`{boundary_overlay.get('eps_feature')}`), never the nominal "
            "`@x<scale>` dial. On cap-saturated devices the nominal dial "
            "compresses (PROJECT_STATUS section 4.10), so a point's position "
            "on this axis is its true error rate.",
            "2. **`zne_fr` alignment.** The theory curve is derived for the "
            "FIXED-Richardson ZNE variant (`zne_fr`: scale factors and "
            "Lagrange-at-zero coefficients shared with "
            "`qemsel.mitigation.richardson_coefficients`, equal-split shot "
            "budget). The selector may also count the legacy `zne` label "
            "(random-fold, 3x budget) as 'use ZNE' when configured; those "
            "two ZNE flavors do NOT sit on the same boundary, so a mismatch "
            "against `zne`-trained bundles is partly a variant mismatch, not "
            "a modeling error.",
            "3. **Simulation only.** Every grid point is a simulated Fake "
            "backend; no `ibm_*` hardware is touched. The noise amplification "
            "the theory assumes (scaling the noise parameter) matches our "
            "`@x` dial more closely than gate folding does — on real "
            "hardware only folding exists.",
        ]
    )


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

def generate_report(
    df: pd.DataFrame,
    model_metrics: dict,
    out_dir: Path,
    cost_aware_metrics: dict | None = None,
    run_config: dict | None = None,
    *,
    stats_results: dict | None = None,
    boundary_overlay: dict | None = None,
) -> Path:
    """Write report.md plus PNG figures summarizing the whole study.

    Args:
        df: experiment results DataFrame — either the raw
            ``qemsel.experiment.run_experiment`` row schema or the
            seed-averaged aggregated schema (``n_seeds`` column, no
            ``seed`` column; its ``<tech>_mean_abs_error`` columns are
            aliased to ``<tech>_abs_error`` internally).
        model_metrics: metrics dict with the exact
            ``qemsel.model.train_and_eval`` return schema (older dicts
            without the 2026-07-21 research-pass keys — ``lobo``,
            ``lodo``, ``dropped_classes``, ``cv_n_samples`` — still
            render).
        out_dir: output directory; created if missing. All PNGs are written
            here and referenced from report.md by RELATIVE filename.
        cost_aware_metrics: optional second metrics dict trained on
            ``best_technique_cost_aware`` (from
            ``qemsel.model.train_and_eval_all``); when given — or when
            ``model_metrics`` embeds it under the ``'cost_aware'`` key,
            as ``train_and_eval_all`` writes into metrics.json — section 6
            renders both label variants side by side.
        run_config: optional experiment config dict (the ``config`` key of
            ``run_meta.json``). When it sets ``min_abs_ideal`` > 0, section
            1 discloses the circuit-selection conditioning (random circuits
            rejection-sampled on ``|<Z^n>| >= threshold``) — a user-facing
            transfer caveat that previously existed nowhere (fixer pass
            2026-07-21). Without the config, bumped rejection-sampling
            seeds in the data still trigger the disclosure.
        stats_results: (V2, keyword-only; builder-recommend / B8
            implements) the ``stats.json`` dict written by
            ``scripts/compute_stats.py`` (schema in qemsel.stats module
            docstring). When given, a NEW section 8 "Statistical hygiene"
            renders: per-technique win-share bootstrap CIs (both label
            columns when present), the raw_plus-vs-raw paired permutation
            test, top-2-technique permutation p-value + Cliff's delta, and
            the Koester checklist as a pass/flag table (any 'passed': False
            printed in bold with the counts). Omitted (None, default) =>
            report is BYTE-IDENTICAL to V1.
        boundary_overlay: (V2, keyword-only; B8) the dict returned by
            ``qemsel.boundary.overlay_selector_vs_theory`` (or loaded from
            its JSON). When given, a NEW section 9 "ZNE help-harm boundary
            overlay" renders: the figure (relative filename from
            'plot_path'; must already sit inside out_dir — ValueError
            otherwise), agreement_pct + iou_help + region shares, n_points,
            and the mandatory caveats block (realized-not-nominal eps axis;
            zne_fr alignment note; sim-only). Omitted => byte-identical.
            CLI: ``scripts/make_report.py --stats-json PATH
            --boundary-json PATH`` (both optional).

    Behaviour contract:
    * matplotlib must use the 'Agg' backend (``matplotlib.use('Agg')`` before
      pyplot import); never call plt.show(); every figure closed after save.
    * PNG files with EXACTLY these names:
        'error_by_technique.png'    mean abs_error per technique, grouped by
                                    backend (bar chart)
        'win_rate.png'              fraction of rows where each technique is
                                    best_technique, per backend
        'winner_vs_noise.png'       win rate AND mean abs_error per technique
                                    vs the noise scale parsed from the
                                    backend column's '@x<scale>' suffix
                                    (scale 1.0 for plain names) — written
                                    ONLY when >= 2 distinct scales exist
        'confusion_matrix.png'      heatmap from model_metrics
                                    ['confusion_matrix'] with ['labels']
        'feature_importances.png'   sorted bar chart from
                                    model_metrics['feature_importances']
    * report.md sections (in order):
        1. Overview — dataset size, circuit families, backends, techniques,
           seed-averaging + noise scales when present.
        2. Technique comparison — table of mean/median abs_error per
           technique per backend; NaN counts (technique failures) reported.
        3. Cost-normalized view — same comparison at an equal shot budget
           (error x sqrt(relative cost), matching the experiment's
           best_technique_cost_aware model) plus cost-aware win-rate and
           per-family winner tables from that stored column; notes the
           empirical `raw_plus` equal-budget baseline when present.
        4. Win rates — how often each technique is best, overall and per
           circuit family / backend.
        5. Noise-scale sweep — win rate and mean abs error per technique vs
           parsed noise scale (always present; says 'not applicable' for
           single-scale datasets); multi-scale datasets additionally get a
           per-scale device-composition + realized-error table (nominal
           scale != realized scale on cap-saturated devices), a confounding
           warning when compositions differ, and the nominal-vs-realized /
           noise-character caveats.
        6. Model evaluation — accuracy ± std + macro-F1 vs majority
           baseline, dropped-classes note, LOFO + LOBO (relabeled as
           noise-level interpolation) + LODO (new-device headline) tables,
           both winner labels side by side when available, confusion
           matrix figure, feature importances figure.
        7. Reproducibility — pointer to run_meta.json, package versions.
    * Numbers in the text formatted to 3 significant figures.

    Returns:
        Path to the written ``out_dir / 'report.md'``.

    Raises:
        ValueError: if df or model_metrics lack required columns/keys.
    """
    techniques = _validate(df, model_metrics)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Aggregated schema: alias '<tech>_mean_abs_error' -> '<tech>_abs_error'
    # so every section reads one column name (copy first — never mutate the
    # caller's frame).
    alias_cols = {
        f"{t}_abs_error": f"{t}_mean_abs_error"
        for t in techniques
        if f"{t}_abs_error" not in df.columns
        and f"{t}_mean_abs_error" in df.columns
    }
    if alias_cols:
        df = df.copy()
        for target, source in alias_cols.items():
            df[target] = df[source]

    if cost_aware_metrics is None:
        embedded = model_metrics.get("cost_aware")
        if isinstance(embedded, dict):
            cost_aware_metrics = embedded

    # Exclude pre-fix CDR classical-simulation artifact rows from EVERY
    # aggregate (post-fix runs record NaN for those cases instead; a genuine
    # shot-noise-limited CDR error can never be < 1e-12).
    n_degenerate_cdr = 0
    if "cdr_abs_error" in df.columns:
        cdr_err = pd.to_numeric(df["cdr_abs_error"], errors="coerce")
        degenerate = cdr_err.notna() & (cdr_err < _CDR_DEGENERATE_TOL)
        n_degenerate_cdr = int(degenerate.sum())
        if n_degenerate_cdr:
            df = df.loc[~degenerate]
            if len(df) == 0:
                raise ValueError(
                    "every row was a pre-fix CDR classical-simulation "
                    "artifact (cdr_abs_error < 1e-12); re-run the experiment "
                    "with the fixed qemsel.mitigation"
                )

    backends = sorted(df["backend"].dropna().unique())

    _save_error_by_technique(df, techniques, backends, out_dir / _PNG_ERROR)
    _save_win_rate(df, techniques, backends, out_dir / _PNG_WIN)
    noise_png_written = _save_winner_vs_noise(df, techniques, out_dir / _PNG_NOISE)
    _save_confusion_matrix(model_metrics, out_dir / _PNG_CONFUSION)
    _save_feature_importances(model_metrics, out_dir / _PNG_IMPORTANCES)

    sections = [
        "# QEM-Selector — benchmark and recommender report",
        "",
        "Comparison of quantum error mitigation techniques on noisy simulated "
        "backends, plus an ML model that recommends the best technique per "
        "circuit.",
        "",
        _section_overview(df, techniques, backends, n_degenerate_cdr, run_config),
        "",
        _section_technique_comparison(df, techniques, backends),
        "",
        _section_cost_normalized(df, techniques),
        "",
        _section_win_rates(df, techniques, backends),
        "",
        _section_noise_sweep(df, techniques, noise_png_written),
        "",
        _section_model_eval(model_metrics, cost_aware_metrics),
        "",
        _section_reproducibility(model_metrics),
        "",
    ]

    # ---- V2 additive sections (byte-identical to V1 when both args None) ---
    # These EXTEND the section list only when their input dict is supplied,
    # so a V1 call (both None) produces the exact original report bytes.
    if stats_results is not None:
        sections.extend([_section_stats(stats_results), ""])
    if boundary_overlay is not None:
        sections.extend([_section_boundary_overlay(boundary_overlay, out_dir), ""])

    report_path = out_dir / "report.md"
    report_path.write_text("\n".join(sections), encoding="utf-8")
    return report_path
