# Tester B — round 1 (end-to-end + scientific sanity)

Date: 2026-07-21. Independent black-box test of the integrated pipeline. I did not
write any of this code. All commands run with the project venv python from the
project root. Test artifacts (mine): `results/tb_r1_main` (fresh e2e) and
`results/tb_r1_crash` (kill/resume test). Verification scripts live in my session
scratchpad, not in the repo.

## Verdict: ALL PASSED — zero critical, zero major. Integrator claims reproduced.

## 1. Fresh end-to-end chain (results/tb_r1_main) — PASS

| Step | Command | Result |
| --- | --- | --- |
| run_experiment | `--config configs/tiny.yaml --out results/tb_r1_main` | exit 0, 20/20 units, 121.9 s, no errors.log |
| train_model | `--data .../results.csv --out ...` | exit 0, honest cv_folds=0 fallback, best random_forest, acc 0.800 == majority baseline, macro-F1 0.296 |
| make_report | `--data ... --metrics ... --out ...` | exit 0, report.md + 4 PNGs (31–39 KB each) |
| recommend demo 1 | `--demo ghz_plus --qubits 3 --depth 4 --seed 0 --backend FakeLagosV2` | exit 0, cdr p=0.90, no sklearn warnings |
| recommend demo 2 | `--demo layered_random --qubits 2 --depth 4 --seed 7 --backend FakeManilaV2` | exit 0, cdr p=0.78, no sklearn warnings |

Only warnings seen during the sweep: mitiq's benign "input circuit is very short"
(ZNE) and one scipy OptimizeWarning inside mitiq CDR curve_fit (see 5c) —
matches the integrator's "3 benign mitiq warnings" description.

## 2. Independent pandas verification of results.csv — PASS (all ~40 checks)

Recomputed everything myself from the raw CSV (script: scratchpad/verify_csv.py):

- 20 rows, 32 columns, 0 duplicate (circuit_id, backend) pairs.
- Every `<tech>_abs_error` >= 0 and finite where not NaN; recomputed
  `|value - ideal|` matches the stored abs_error to < 1.7e-16 on every row.
- All `<tech>_value` within [-1.1, 1.1] (max observed |value| = 1.0277, a REM
  inversion slightly past 1 — documented unclipped behavior). All `ideal` in [-1, 1].
- `best_technique` == my own recomputed NaN-excluded argmin on ALL 20 rows.
  `best_technique_cost_aware` == my recomputed argmin of
  `abs_error * sqrt(shots/2000)` on ALL 20 rows.
- NaN rate per technique: 0% for raw/zne/cdr/rem (gate was < 30%).
- Shots columns: raw 2000 (1x), zne 6000 (3x), rem 6000 (3x), cdr 22000 (11x) on
  every row — positive and ordered cdr > zne/rem > raw as the cost model requires.

## 3. Scientific sanity — PASS

- **Mirror ideal**: exactly +1.0 on all 4 mirror rows (integrator's snap fix holds
  on a fresh run, not just on the recomputed rows).
- **raw never wins**: best_technique raw = 0/20; the best mitigated estimate beats
  raw on **20/20 rows**. Mitigation is correctly wired.
- **CDR != raw**: cdr_value differs from raw_value on 20/20 rows. ZNE != raw on
  20/20 rows. REM != raw on 20/20 rows. No silent fallbacks.
- **Physics is sensible**: mean abs_error pooled raw 0.2367 / zne 0.2284 /
  cdr 0.0230 / rem 0.0286. Per backend: Lagos raw 0.384 (heavy ~27% readout error
  degrades raw badly, ZNE barely helps 0.382 — ZNE does not target readout error),
  Manila raw 0.090. REM recovers Lagos to 0.035. Mirror rows on Lagos: raw ~0.47-0.50
  vs ideal 1.0, CDR/REM recover to within 0.03 — all as expected physically.
- **Winner distribution**: cdr 16 / rem 3 / zne 1 / raw 0; cost-aware cdr 14 /
  rem 5 / raw 1 — identical to the integrator's claim.

## 4. Crash-safety (results/tb_r1_crash) — PASS

- Started a fresh run, killed the python process (Stop-Process -Force) after
  exactly 3 rows had been appended to results.csv. CSV intact after kill (3 rows).
- Restarted with the SAME config + out dir: header line said "3 already in
  results.csv", exactly 3 "skipped (already in results.csv)" lines ([1/20]-[3/20]),
  then computed units 4-20 and finished exit 0.
- Final CSV: 20 rows, 0 duplicates, same unit set as the uninterrupted run, and
  **bit-identical values to the uninterrupted run** (max numeric diff < 1e-12 on
  every column, winners identical) — per-unit seeding makes resume exact.

## 5. Reproducibility + observations (no action required, all minor)

a) **My fresh run vs integrator's results/tiny**: identical to < 1e-12 on every
   numeric column except cdr_value/cdr_abs_error (max diff 1.1e-9 — CDR
   regression-solver jitter). Winners and cost-aware winners identical on all
   20 rows. The integrator's reported stats are fully reproduced.

b) **Claim nit (wording only)**: the integration report says "REM strongest on
   readout-heavy FakeLagosV2". In the data (theirs and mine), CDR has the lowest
   mean abs_error on Lagos (0.025 vs REM 0.0347); REM is 2nd. What is true: REM's
   *relative* advantage over raw/zne is largest on Lagos, and REM wins 2 of its 3
   accuracy wins there. No code issue.

c) **Scientific caveat for the small/full runs** (extends the documented CDR
   Clifford-short-circuit quirk): on 8/20 rows (all ghz_plus AND all near_clifford
   rows) `cdr_value == ideal` EXACTLY (abs_error 0.0).
   - ghz_plus (n_non_clifford=0): mitiq's documented fully-Clifford short-circuit.
   - near_clifford (n_non_clifford=2): NOT short-circuited — instead
     `fraction_non_clifford=0.2` of 2 non-Clifford gates rounds to 0 kept, so ALL
     training circuits are fully Clifford with ideal value exactly 1.0; the
     curve_fit then degenerates to the constant y=1.0 (this is the scipy
     OptimizeWarning seen in the log) and predicts the ideal exactly, ignoring the
     noisy target measurement entirely.
   - Contrary to the integration report's parenthetical, mirror_circuit rows are
     NOT short-circuited (n_non_clifford=8 -> genuine CDR regression,
     abs_errors 4e-4..9e-3).
   - Net effect: cdr's 16/20 label share at tiny scale is partly an artifact of
     circuits whose Clifford proxies are trivially exact. At larger qubit counts /
     depths (small/full configs) training circuits will have varied ideal values
     and this degeneracy should fade — but worth re-checking the
     `cdr_abs_error == 0` row count on the small run before trusting the labels.

d) The model at 20 rows predicts the majority class everywhere (confusion matrix
   collapses to the cdr column, permutation importances all 0.0) while
   predict_proba still varies — consistent with the integrator's "data-scale
   property" framing; metrics.json flags it honestly (cv_folds=0, accuracy ==
   baseline).

## Errors found: none critical, none major.
Minor items: 5b (report wording), 5c (mirror not actually short-circuited in the
claim; CDR-exact-ideal rows are a dataset-design caveat, already half-documented
as an accepted quirk).
