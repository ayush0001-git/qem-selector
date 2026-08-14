# Experiment verifier notes (small run) — 2026-07-21

Fresh-eyes verification of the full chain on `configs/small.yaml`.

## What was run (all real, no mocks)

1. `pytest tests -q` (full suite, slow tests included): **266 passed, 0 failed,
   0 skipped** in 77 s. Only 3 benign mitiq "circuit is very short" UserWarnings.
2. `run_experiment.py --config configs/small.yaml --out results/small`
   (fresh dir): completed with exit 0. 120 (circuit, backend) units attempted,
   46 screened out pre-noise by `min_abs_ideal=0.25`
   (`skipped_low_signal.log`, all layered_random/hw_efficient/near_clifford
   ZZ ideals in (−0.25, 0.25)), **74 rows** written.
3. `train_model.py --data results/small/results.csv --out results/small`: OK.
4. `make_report.py`: report.md + 4 PNGs written.
5. `recommend.py --demo` on 2 families: both OK, no sklearn warnings.
   - ghz_plus q3 d4 @ FakeLagosV2 -> **rem** (p=1.0) — sensible (Clifford
     circuit, 20% avg readout error backend).
   - layered_random q2 d4 seed7 @ FakeManilaV2 -> **cdr** (p=1.0) — sensible
     (non-Clifford, low-readout-error backend).

## Paranoid checks

- Rows: 74, all unique on (family, n_qubits, depth, seed, backend);
  37 per backend. Winner == argmin(abs_error) on all 74 rows (0 mismatches).
  0 rows with |ideal| < 0.25 (screen works). 0 rows with cdr_abs_error < 1e-12
  (no Clifford-shortcut artifact rows).
- NaN rate per technique: raw 0%, zne 0%, rem 0%, **cdr 30/74 = 40.5%** —
  above the 30% alarm gate, so reported as a finding. Breakdown: ghz_plus
  22/24, near_clifford 8/8, everything else 0. All 30 are the intentional
  fail-loud `MitigationError` refusals from the enhancement pass (fully
  Clifford circuit, or all training-circuit ideals identical), each logged in
  errors.log; zero crashes/tracebacks. Verdict: structural property of a
  Clifford-heavy benchmark suite + honest guards, NOT a pipeline failure; but
  it does mean CDR has no label signal on 2 of 5 families, and `best_technique`
  for those rows is chosen among raw/zne/rem only.
- Winner balance (best_technique): rem 38, cdr 35, zne 1, raw 0.
  Cost-aware: rem 37, cdr 32, raw 4, zne 1. raw never wins on pure accuracy
  (mitigation correctly wired); zne nearly never wins (readout-dominated
  noise on these 5q/7q fake backends — physically plausible).
- Mean abs_error pooled | Manila | Lagos:
  raw 0.423 | 0.209 | 0.637; zne 0.391 | 0.158 | 0.623;
  cdr 0.040 | 0.012 | 0.068 (non-Clifford rows only, 11x shots);
  rem 0.102 | 0.053 | 0.151. Ordering matches physics (Lagos readout-heavy).
- Plots: error_by_technique 29 KB, win_rate 32 KB, confusion_matrix 33 KB,
  feature_importances 45 KB — all far above the 1 KB triviality bar and
  visually non-empty per file size/structure.

## Model — honest numbers

- Pipeline metrics.json: n=74, class balance cdr 35 / rem 38 / **zne 1**.
  The zne singleton class makes stratified grouped CV undefined ->
  `cv_folds=0` fallback: fit on all rows, evaluated on the training set.
  Pipeline-reported accuracy **0.905** (macro-F1 0.608) vs majority baseline
  0.514 is therefore **optimistic training-set accuracy**, and metrics.json /
  train_model.py say so explicitly. This is the designed honest fallback, not
  a bug — but the headline pipeline number should NOT be quoted as skill.
- Independent verifier CV (ad-hoc, zne singleton dropped, n=73, rem 38/cdr 35,
  StratifiedGroupKFold(5) grouped by (family, n_qubits, depth), fresh
  RandomForest): **accuracy 0.823 +/- 0.088** (folds 0.88/0.86/0.86/0.67/0.86),
  macro-F1 0.821, grouped-majority baseline 0.521. Leave-one-family-out:
  **0.808** (per family 0.92/0.67/0.75/0.83/0.88). Real skill above baseline,
  but at ~74 rows one fold is ~15 rows, so +/- ~0.09 fold spread is expected —
  treat 0.82 as "clearly above 0.52 baseline", not as a precise number.

## Issues / recommendations for the full run

1. cdr NaN 40.5% (designed refusals) — consider either an explicit
   'cdr_refused' feature or documenting that cdr is out of the running for
   fully-Clifford families; the model currently learns this implicitly via
   clifford_fraction (plausible driver of the high LOFO on ghz_plus).
2. zne singleton class breaks pipeline CV at this scale; full.yaml should
   produce enough zne wins for cv_folds>0, otherwise consider min-class
   handling (merge-or-drop) in model.py.
3. 46/120 units (38%) screened by min_abs_ideal — intended, but the surviving
   family mix is skewed (ghz_plus/mirror 24 rows each vs layered_random 8,
   near_clifford 8): worth rebalancing family counts in full.yaml.

Verdict: full chain completed end-to-end on real noisy simulation; outputs are
scientifically sane. e2e PASS.
