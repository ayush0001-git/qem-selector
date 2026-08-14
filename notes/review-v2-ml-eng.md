# Review: V2 model.py training/eval pipeline (principal-ML-engineer pass)

Scope: `src/qemsel/model.py` (derive_significant_label, calibrate, abstain,
V2 features/LOSO), `scripts/train_model.py`, `tests/test_model_v2.py`,
smoke artifacts `results/boundary_smoke/*`. All numeric claims below were
verified by running code with the project venv (scripts in the session
scratchpad: `check_sigma.py`, `check_calibrate.py`).

Verdict up front: the CV scaffolding is genuinely strong — group-aware
folds, honest OOF metrics, held-out permutation importances, LOFO/LODO/LOSO
holdouts, cross-fitted calibration Brier. I would NOT ship the
significance-aware label as-is: its sigma is statistically wrong for every
mitigated technique, in the anti-conservative direction, and the shipped
smoke artifact was produced through the worst of the two routes.

---

## HIGH-1: significant-label sigma treats consumed shots as pooled shots — understates sigma for every mitigated technique

`derive_significant_label` (per-seed route) computes
`sigma_shot(<tech>_value, <tech>_shots)` where `<tech>_shots =
shots_consumed(tech, base_shots)` (model.py lines ~527-531, 565-567).
`sqrt((1-x^2)/N)` is correct ONLY for a single direct estimate from N
pooled shots. The techniques do not pool:

- **zne** (Richardson over scales (1,2,3), each node at `base_shots`):
  Var = sum c_k^2 * (1-x_k^2)/base_shots with c = (3, -3, 1), sum c^2 = 19.
  Code uses (1-x^2)/(3*base). Sigma understated **x sqrt(57) ~ 7.5**.
- **zne_fr** (fixed Richardson (1,3), equal_split of ONE budget):
  Var ~ 2.5 * (1-x^2)/(base/2) = 5*(1-x^2)/base; code uses (1-x^2)/base.
  Understated **x sqrt(5) ~ 2.2**.
- **cdr / cdr_ridge**: the 10 training-circuit executions fit the
  regression map; they do not average the target estimate down. Code
  divides by 11*base. Understated **>= x sqrt(11) ~ 3.3** (lower bound —
  regression adds variance on top).
- **rem**: 2 calibration circuits + inverse response matrix (amplifies).
  Code divides by 3*base. Understated **>= x sqrt(3) ~ 1.7**.
- raw / raw_plus are correct (raw_plus genuinely measures once at 11N).

Measured impact on `results/boundary_smoke/results.csv` (24 rows,
k_sigma=2): **4/24 (17%) labels flip from a significant winner to 'tie'**
under estimator-aware sigmas (using the codebase's own
`mitigation.richardson_coefficients`). Winners in this smoke were mostly
cdr/rem; on runs where zne/zne_fr win near the help-harm boundary the x7.5
/ x2.2 factors bite exactly where margins are smallest.

Why this is a paper-killer for Angle 3: `boundary.py` computes the analytic
boundary WITH the Richardson variance amplification (`variance_k_q` =
sum c_j^2/pi_j - 1, from the same `richardson_coefficients`). So the
learned-boundary side (trained on significance labels that IGNORE the
amplification) and the analytic side use inconsistent noise models — the
selector will "significantly" prefer/refuse ZNE in a band where the theory
says shot noise cannot distinguish it. A reviewer who checks either side's
variance model will find the other one wrong.

Fix: compute per-technique sigma with the estimator's actual variance:
raw/raw_plus as now; zne via sum c^2/(per-node shots); zne_fr via
sum c^2/pi over the split budget (both coefficients already exported by
`mitigation.richardson_coefficients`); cdr/rem at MINIMUM use the target
execution's base_shots (honest lower bound on sigma -> conservative ties),
better: propagate the regression/inverse-matrix factor.

## HIGH-2: aggregated-route value proxy `1 - mean_abs_error` is anti-conservative — and it produced the shipped smoke artifact

Docstring calls the aggregated proxy "conservative"; it is the opposite.
`sigma_shot(1 - err, N)` has variance term `1-(1-err)^2 ~ 2*err`, so a
SMALL error forces a tiny sigma regardless of the TRUE expectation value,
which is what actually sets the binomial variance. Ideal values in the
smoke run span -0.717..1.0 (median 0.29), i.e. variance terms near 1, not
near 0.

Measured on `results/boundary_smoke/aggregated.csv` (24 rows): winner-sigma
understated **median x3.6, max x54.9**; **5/24 (21%) labels flip** purely
from replacing the proxy value with the group's true ideal (holding the
also-wrong shots pooling of HIGH-1 fixed at the shipped behavior).

Compounding: the label counts in the shipped
`results/boundary_smoke/metrics_significant.json` (cdr 10, raw_plus 2,
rem 3, tie 9) match the aggregated route exactly — the shipped
`model_significant.joblib` was trained on these labels. And
`scripts/train_model.py` explicitly RECOMMENDS aggregated.csv for headline
numbers, so the worst route is the default paper path.

Fix options (any is defensible, current proxy is not):
(a) carry `ideal` (and/or per-tech mean values) into aggregated.csv — the
per-seed file already has both; (b) use the empirical seed-scatter
(std of per-seed errors / sqrt(n_seeds)) when n_seeds >= 2; (c) refuse the
aggregated route for the significant label and require per-seed data.

## HIGH-3: `calibrate=True` silently ships an UNCALIBRATED model stamped `calibrated: True` whenever a singleton class exists

Reproduced with the venv (sklearn 1.9.0): a training frame with one
singleton class (exactly the research sweep's "one lone 'zne' win"
situation described in model.py's own docstring) takes this path in
`_fit_calibrated`: smallest class count = 1 -> grouped path skipped ->
`cv_fallback = min(5, 1) = 1 < 2` -> returns the bare fitted
RandomForestClassifier. The bundle still records
`bundle['calibrated'] = True`, `metrics['calibration']` is still emitted
(from the CV folds, where the singleton was dropped), and
`recommend.py` will then apply the stored `abstain_threshold` to RAW
random-forest probabilities the user was told are calibrated. RF raw
probabilities are typically overconfident -> the abstain gate fires far
less often than the CV `abstain_rate_cv` promised.

Control run without the singleton ships a `CalibratedClassifierCV` as
expected, so the degradation is purely data-dependent and invisible.

Fix: record the actual calibration route in bundle+metrics
(`'calibration_route': 'grouped_cv' | 'prefit_frozen' | 'DEGRADED_uncalibrated'`),
set `calibrated: False` on the degraded path, and warn loudly. Better:
exclude singleton classes from the calibration fit (they are already
excluded from CV) instead of abandoning calibration wholesale.

## MEDIUM-1: abstain threshold is honest but unactionable — no selective-accuracy/coverage from OOF

Good news first: the threshold is user-supplied, stored verbatim, never
tuned on any holdout, and `abstain_rate_cv` comes from out-of-fold
(calibrated when calibrate=True) probabilities — no test leakage. But the
metrics give NO way to CHOOSE the threshold honestly: only the abstain
rate is reported, not what accuracy the model achieves on the rows it
keeps. Anyone picking a threshold today must either guess or peek at
held-out evaluations. Emit, from the SAME OOF probabilities already
computed: coverage vs selective-accuracy at a grid of thresholds
(e.g. 0.5..0.95), so the paper can state "threshold chosen from CV
coverage-accuracy curve only". Also: in the cv_folds=0 fallback,
`abstain_rate_cv` comes from training-set probabilities with no flag —
mirror the `feature_importances_note` pattern.

## RULING: `feat_log2_shots` is legitimate signal, NOT leakage

Reasoning, for the record:
1. It is a deployment-time controllable input: `recommend()` REQUIRES
   `base_shots` for feature_version-2 bundles (verified in recommend.py),
   so nothing available at train time is unavailable at predict time.
2. No fold leakage: the CV group key (family, n_qubits, depth)
   deliberately EXCLUDES base_shots, so the same circuit config at 256 and
   4096 shots lands in the SAME fold — near-duplicate rows differing only
   in log2_shots can never straddle train/test.
3. Shots-axis generalization is measured separately and honestly (LOSO).

One caveat the paper MUST state (medium, wording only): for the
SIGNIFICANT label, shots enters the label-generating rule itself
(sigma ~ 1/sqrt(N), so low budgets mechanically produce 'tie'). The
learned tie-region-vs-shots dependence is therefore partly the labeling
rule read back, not discovered physics. For Angle 3 that is the point
(the selector should reproduce the significance geometry), but present it
as such.

## LOW: post-selection winner bias in the significance test

The winner is chosen by argmin over the same noisy errors that feed the
margin test; with 7 techniques the winner's error is selection-biased low,
making the winner-vs-runner-up test slightly anti-conservative even with
correct sigmas. With k_sigma=2 and the HIGH-1/HIGH-2 fixes in place this
is second-order; a one-line acknowledgment in the paper suffices (or use
max-over-pairs / Bonferroni-lite k).

## What is RIGHT (verified, keep)

- `stats.sigma_shot` itself implements sqrt((1 - min(x^2,1))/N) exactly as
  specified; unphysical values clamp to sigma 0 with a documented rationale.
- Group-aware CV with singleton-class dropping, honest cv_folds=0
  fallback, and `cv_grouping`/`dropped_classes` bookkeeping.
- Calibration Brier before/after are computed on OOF folds with
  CROSS-FITTED per-fold calibration (`_oof_proba(calibrated=True)` fits
  `_fit_calibrated` inside each training fold) — this part is honest.
- Permutation importances on held-out folds; training-set fallback flagged.
- LOFO/LOBO/LODO/LOSO family with correct interpolation-vs-generalization
  caveats; LOSO activates only when >= 2 budgets exist.
- V1 byte-identity discipline (default paths add no keys, V2 keys additive).
