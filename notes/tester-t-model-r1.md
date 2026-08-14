# Adversarial Tester T-model — round 1 (2026-07-23)

Scope: `qemsel.model` V2 surface (significant label, calibration, abstain) +
old-label-path END_RESULT reproduction. All verification by RUNNING code
(scripts in the session scratchpad: `test_sig_label.py`, `test_calib_abstain.py`,
`test_recommend_abstain.py`, `test_cli_paths.py`, `repro_research.py`).
Sim-only; no ibm_* touched; no src/tests modified.

## Verdict: PASS (0 critical, 0 major; 2 minor edge notes)

### 1. significant label (`derive_significant_label`) — PASS
- Per-seed route, exact threshold straddling (values=0, shots=1e4 so sigma is
  analytic): margin == k*sigma_comb -> winner (>= inclusive), margin*(1-1e-9)
  -> tie, 200-point sweep over [0.5T, 1.5T]: 0 mismatches. k_sigma=1/2/4 moves
  the boundary correctly; sigma correctly shrinks with |value| (0.99 case).
- Edge cases all correct: all-NaN -> `''`; single valid -> winner outright;
  NaN/0 shots or NaN value -> conservative tie (sigma=inf); |value|>1 -> sigma
  clamped 0 -> any margin significant; empty df OK; techniques filter OK;
  missing base_shots / n_seeds on aggregated schema -> clear ValueError.
- Aggregated route: 400-row randomized df incl. NaNs and near-boundary pairs
  matches an independent reimplementation of the documented rule
  (sigma_shot(1-mean_err, n_seeds*base_shots*shots_consumed)) on all 400 rows.
- Real data: `results/boundary_smoke/aggregated.csv` recomputed independently
  -> 24/24 labels match; 9/24 tie rows (confirms integrator's number).

### 2. calibration — PASS
- `calibrate=True` on noisy synthetic (2 seeds): held-out CV Brier improves
  (0.868->0.662; 0.725->0.642). Fresh-draw holdout: calibrated bundle Brier
  0.679 < uncalibrated 0.787; overconfidence gap (mean top-p − accuracy)
  shrinks +0.228 -> +0.127. metrics['calibration'] = {method: sigmoid,
  brier_before, brier_after}; calibrated bundle carries calibrated=True and
  full class set; default (V1) bundle keeps exactly the 6 V1 keys and default
  metrics gain no V2 keys.

### 3. abstain — PASS
- Threshold validation: 0.0 / 1.0 / -0.2 / 1.5 all raise ValueError.
- `abstain_rate_cv`: exactly 0.0 at threshold 1/3 with 3 classes (max-p can
  never be below 1/n), monotone in threshold (0.0 <= 0.556 <= 1.0), in [0,1].
- `recommend()` boundary is EXACT and strict: stub bundle with p_max=0.6 —
  threshold 0.6 -> NOT abstain; nextafter(0.6) -> abstain. Threshold stored
  verbatim; V2 return keys exact; V1 bundle returns exactly 3 keys; fv2 bundle
  without base_shots raises; log2_shots feeds through (4096->12, 1024->10).
- CLI E2E: shipped `model_significant.joblib` (thr 0.9) -> exit 2 + 'abstain'
  on 5/5 probes (top-p 0.40–0.44); thr-None `model.joblib` and my thr-0.35
  bundle -> exit 0 + argmax on 7/7 probes. Exit code always == 2 iff top-p <
  threshold.

### 4. old label paths reproduce END_RESULT — PASS
- Fresh default-flag `train_and_eval_all` on `results/research/aggregated.csv`:
  metrics.json AND metrics_cost_aware.json FULL-DICT IDENTICAL to shipped.
  Headlines match END_RESULT.md exactly: CV 0.796±0.053 / F1 0.417 / baseline
  0.594 / LOFO 0.787 / LOBO 0.893 / LODO 0.865 (Jakarta 0.967, Lagos 0.694,
  Manila 0.933); cost-aware 0.728±0.133 / 0.583 / 0.437 / 0.702 / 0.783 / 0.704.
- Determinism: retraining the integrator's significant config (`--label
  significant --feature-version 2 --calibrate --abstain-threshold 0.9 --stats`
  on boundary_smoke aggregated) reproduces the shipped
  `metrics_significant.json` full-dict.

## Minor edge notes (no action required for the paper runs)
1. Two techniques with EQUAL errors and both |value| >= 1 (sigma clamped to 0)
   get labeled a "significant" winner by column order (margin 0 >= k*0)
   instead of tie. Reachable only with saturated/unphysical estimates.
   Repro: df with alpha/beta abs_error=0.1, value=1.0, shots=1e4 ->
   `derive_significant_label` returns 'alpha'.
2. Aggregated-schema dfs with technique names outside TECHNIQUES_V2 crash with
   `ValueError: unknown technique 'foo'` from `qemsel.mitigation.shots_consumed`
   — correct to refuse, but the message does not mention the significance-label
   schema. Per-seed route accepts arbitrary names (no shots_consumed needed).
3. Informational: the significant-model CLI can recommend the literal class
   'tie' with exit 0 (it is a real class, distinct from 'abstain'); wrappers
   mapping technique names to executors must handle it.
