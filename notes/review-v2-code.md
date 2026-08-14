# V2 code review — staff-engineer bug hunt (2026-07-23)

Scope: the V2 additions (shots axis, zne_fr/cdr_ridge/cdr_rf, boundary module,
stats module, calibrated/abstaining model, significance labels, report §8/§9).
Every finding below was REPRODUCED by running code with the project venv;
verification scripts lived in the session scratchpad. Areas checked and found
CLEAN are listed at the end.

---

## F1 (HIGH) — `cdr_ridge` is crippled by an alpha-scale artifact; Angle 2 comparison invalid as implemented

`mitigation.CDR_RIDGE_ALPHA = 1.0` ("sklearn default; matches Korolev") is
applied to a SINGLE UNSTANDARDIZED feature (the noisy expectation, range
[-1, 1]). Over the 10 training points, Sxx = sum((x-mean)^2) is O(0.01–1), so
the Ridge slope is `Sxy/(Sxx + alpha)` — shrunk by 43–99% toward 0, i.e. the
prediction collapses toward the mean of the training ideals.

Measured (boundary_smoke, layered_random_q2_d4_s0 @ FakeManilaV2@x0.25, 4096
shots — near-noiseless, true map ~identity):

- Sxx = 0.7645 -> slope shrink factor 0.433
- OLS (what mitiq `cdr` fits): slope +0.995, pred -0.7135 (ideal -0.7172, err 0.004)
- Ridge(alpha=1): slope +0.431, pred -0.4505 (err 0.267 — pure regularization bias)

Across the 24 smoke rows: mean |err| cdr 0.021 vs cdr_ridge **0.142** (6.7x
worse; worse on 21/24 rows). On mirror circuits cdr_ridge predicts ~0.953 at
EVERY noise level — a near-constant output. A CDR variant with nonzero error
at zero noise is unphysical; this is not "regularized linear CDR", it is
regularization bias from an alpha that is huge relative to the data scale.
The Angle 2 map anchored on "Ridge usually wins" (Korolev 2606.02697) would
find the opposite as a pure artifact.

Fix: standardize x before Ridge (Pipeline(StandardScaler, Ridge)) or use
RidgeCV over alphas ~ {1e-4..1}, or scale alpha by Sxx. Keep the refusal
guards identical. File: `src/qemsel/mitigation.py` (CDR_RIDGE_ALPHA ~line 241,
`_apply_cdr_sklearn` step 4 ~line 904).

## F2 (HIGH) — calibrate=True silently ships an UNCALIBRATED model stamped `calibrated: True`

`model._fit_calibrated` computes `k = min(n_splits, smallest_class, n_groups)`.
The refit call passes the FULL y (including CV-dropped singleton classes), so
any singleton class (a lone 'raw'/'cdr_ridge' win — common) forces
smallest_class = 1 -> k = 1 -> grouped path skipped; then
`cv_fallback = min(5, 1) = 1` -> FrozenEstimator path skipped too -> the bare
uncalibrated estimator is returned. `train_and_eval` then stamps
`bundle["calibrated"] = bool(calibrate)` unconditionally and writes a
`metrics['calibration']` dict whose brier_before/after come from the fold-level
calibrated OOF probabilities — an improvement the SHIPPED artifact does not
have.

Reproduced two ways:

- Shipped smoke artifacts: `results/boundary_smoke/model.joblib` and
  `model_cost_aware.joblib` both carry `calibrated: True` but store a plain
  `RandomForestClassifier` / `GradientBoostingClassifier` (dropped_classes
  ['cdr_ridge','raw'] in metrics.json — the singletons that triggered it).
  metrics.json still reports brier_before 0.989 -> brier_after 0.755.
- Synthetic 3-class df with one singleton: same result
  (calibrated flag True, model type GradientBoostingClassifier).

Impact: the paper's "calibrated, abstaining selector" claim; abstain_rate_cv
is measured on calibrated fold probabilities while recommend-time abstain
thresholds act on uncalibrated probabilities — inconsistent abstain behavior.
`tests/test_model_v2.py` only covers the balanced-class happy path.

Fix: after `_fit_calibrated`, set the flag from the actual object (isinstance
CalibratedClassifierCV) or record `calibration_degraded: True` + warn loudly.
Secondary note: the FrozenEstimator fallback uses a plain int cv (ungrouped
KFold), contradicting the module's own never-plain-KFold leakage rule.
File: `src/qemsel/model.py` (~lines 309–362, 1084, 1100–1108).

## F3 (HIGH) — significance-label sigma model is wrong for multi-execution techniques; ~10% of research labels flip

`derive_significant_label` (per-seed route) uses
`sigma_shot(value, <tech>_shots)` with `<tech>_shots` = TOTAL consumed shots,
i.e. it pretends all executions averaged one estimator. For extrapolation/
correction estimators the variance is AMPLIFIED, not pooled:

- `zne` (mitiq Richardson, nodes (1,2,3), coeffs (3,-3,1), each level at B
  shots): Var = (9+9+1)·v/B = 19·v/B. Implemented: v/(3B). Sigma
  underestimated **7.5x**.
- `zne_fr` (equal split, coeffs (1.5,-0.5), B/2 per level): Var = 5·v/B.
  Implemented: v/B (shots column = B). Underestimated **2.2x**.
- `cdr`/`rem`: shots=11B/3B pretends training/calibration shots reduce the
  target-estimate variance; they do not (regression/inversion noise adds).

Measured on `results/research/results.csv` (1620 rows, k_sigma=2): labels =
{cdr 924, rem 443, tie 229, zne 18, raw_plus 6}. Of the 513 rows where zne is
winner or runner-up and the implemented sigma says "significant", **168 flip
to 'tie'** with the variance-correct zne sigma alone — 10.4% of ALL labels,
before even correcting cdr/rem/zne_fr. The tie class is systematically
understated, which defeats the label's purpose (separating statistical ties
from real wins) and inflates `model_significant`.

Compounding edge (verified): `stats.sigma_shot` clamps |value| > 1 to
variance 0, so any overshooting estimate (research run: zne 5 / cdr 159 /
rem 53 overshoot rows) gets sigma = 0 and ANY margin is declared significant
— exactly on the variance-blowup rows. A synthetic aggregated row with
mean_abs_error 2.5 vs 2.5000001 labels 'raw' significant at margin 1e-7.

Same sigma is used by `stats.koester_checklist.winner_margin_below_k_sigma`
(info-only there, but the reported tie fraction is likewise understated).

Fix: per-technique variance models — zne/zne_fr: sum c_j^2·(1-v_j^2)/n_j from
`richardson_coefficients` (single source of truth already exists); rem:
var_raw/damping^2; cdr: at least var of the single target execution at B
shots (conservative); treat |v|>1 as v=1-epsilon or sigma=+inf (tie), never 0.
Files: `src/qemsel/model.py` (~527–571), `src/qemsel/stats.py` (54, 494–499).

## F4 (MEDIUM) — `scripts/compute_stats.py` is contracted, documented, and missing

INTERFACES.md V2.8 (line ~243) assigns B6 "`scripts/compute_stats.py` — NEW
CLI `--data <csv> --out <dir>` writing stats.json"; `make_report.py --stats-json`
help says "written by scripts/compute_stats.py"; the stats module docstring
documents its schema. The file does not exist (`notes/B6-stats.md` explicitly
declared it out of scope and no one picked it up). The report §8 pipeline is
therefore unrunnable from the documented CLIs — `results/boundary_smoke/
stats.json` was produced ad hoc by the integrator. There is also no driver
script for `boundary.overlay_selector_vs_theory` (boundary.json equally ad
hoc), though that one was never promised as a script. The boundary.yaml
paper run will need both; ~40 lines of CLI to write.

## F5 (LOW) — a single-class holdout complement crashes the whole training run

`model._leave_one_group_out` fits with no guard; if removing one family/
backend/device/budget leaves single-class training data,
GradientBoostingClassifier raises ("y contains 1 class...") and the exception
propagates out of `train_and_eval` — no bundle, no metrics (reproduced with a
synthetic df where one backend's complement is single-class). Pre-existing for
LOFO/LOBO/LODO, but the new LOSO axis and small boundary-slice runs add
opportunities. Fix: try/except per held-out value, record None + note.
File: `src/qemsel/model.py` (~263).

## F6 (LOW) — `estimate_params` ignores a grid_spec `eps_feature` override

The overlay grid honors `grid_spec['eps_feature']`, but `estimate_params`
hardcodes `DEFAULT_EPS_FEATURE` for the secant slope (line ~167), so with an
override the theory params are derived on the 2q-error axis while delta_mse
is evaluated on a different axis — inconsistent boundary. Latent (nothing
overrides it today). File: `src/qemsel/boundary.py`.

---

## Verified clean (ran, not just read)

- **Shots-axis resume**: fake-fast list-mode run (16 units, 2 budgets),
  truncated 3 rows + torn partial line, re-run — exact refill, 0 duplicate
  (circuit_id, backend, base_shots) keys, torn tail repaired, aggregated.csv
  regrouped per budget (8 groups), errors.log carries the `s{budget}` field.
  np.int64-vs-int key types hash-compatible; scalar-mode CSV in a list-mode
  out_dir correctly dies on the column-equality check.
- **Determinism / seed plumbing of new techniques**: current boundary_smoke
  results.csv vs boundary_smoke_prev — all 18 overlapping units identical in
  every column (the 6 extra rows are the added @x0.5 backend). cdr_rf seeds
  its RandomForestRegressor from the unit seed; zne_fr rebuilds seeded
  per-level executors; equal-split cost matches SHOT_MULTIPLIER_V2.
- **Boundary-grid feature construction**: re-ran
  `overlay_selector_vs_theory` on the stored fv2 model.joblib + the smoke
  grid — reproduced stored boundary.json EXACTLY (agreement 86.67%, 15/15
  grid points, delta_mse to <1e-12). Feature vectors are built in the
  bundle's `feature_names` order and sklearn validates the named DataFrame.
- **Calibration bundle serialization round-trip**: `model_significant.joblib`
  (true CalibratedClassifierCV + abstain 0.9, fv2) joblib-loads and
  `recommend()` returns abstain with probabilities summing to 1; fv2 bundle
  without `base_shots` raises the documented ValueError.
- **Report edge dfs**: all-NaN technique column, single-class metrics,
  all-'' winners, partial stats dicts — `generate_report` renders without
  crashing; `koester_checklist` reports nan_rate 1.0 and passes.
- `_normalize_shots` validation, low-signal screen with `unit_idx +=
  len(budgets)`, `_write_aggregated` per-budget cost bases, LOSO wiring,
  richardson_coefficients ((1,3) -> (1.5,-0.5); K_0=4, K_1=5 match the spike),
  backends low-noise dial monotonicity (0.046 < 0.067 < 0.103 in smoke).
