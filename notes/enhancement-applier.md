# Enhancement applier — review-fix pass (2026-07-21)

Task: apply the 14 actionable findings from review-science / review-stats /
review-code / review-usability. All 14 applied (none dismissed); two explicit
scope decisions recorded below. Priority order followed: scientific
correctness > statistical honesty > robustness > convenience.

## Verification before changing anything

- Probed mitiq 1.0.0 in the venv: `mitiq.cdr.clifford_utils.is_clifford`
  and `mitiq.cdr.generate_training_circuits` both accept qiskit circuits
  (despite cirq type hints) and return qiskit circuits with correct
  num_qubits; `execute_with_cdr` has `fit_function=`/`num_fit_parameters=`.
- Reproduced the degenerate-training-ideal claim: near_clifford q2/q3 all
  ptp=0.0; ghz_plus fully-Clifford or ptp=0.0; the test-fixture cdr_circuit
  has ptp=0.36 (so the core mitigation tests survive the new guard);
  mirror/hw_efficient/layered_random have healthy spreads.
- Reproduced the uncoupled-pair zero-noise bug and validated the fix:
  with `AerSimulator.from_backend(FakeLagosV2)` + routed transpile, pair
  (3,4) |11> decays 0.9105 -> 0.104 over 60 cx (pre-fix delta was 0.0000).

## Changes by file

- `src/qemsel/backends.py`: executor simulator = AerSimulator.from_backend
  (routing + ECR direction fixing => noise fires on every pair); executor
  rejects circuits wider than the backend; RealHardwareBackend stub message
  now truthful (implement the class; token alone does nothing).
- `src/qemsel/mitigation.py`: _apply_cdr fail-loud guards (fully-Clifford
  and degenerate-training-ideals -> MitigationError; also a clear error for
  the idle-qubit width mismatch); CDR_FIT_FUNCTION / CDR_NUM_FIT_PARAMETERS
  / CDR_MIN_TRAINING_IDEAL_SPREAD constants; module docstring documents both
  custom-regressor routes (parametric via fit_function; sklearn via
  generate_training_circuits).
- `src/qemsel/experiment.py`: per-family pauli dict (single char repeats to
  width) + validation; min_abs_ideal low-signal screening + 
  skipped_low_signal.log; backend-width fail-fast in _validate_config;
  torn-CSV self-heal (_repair_torn_tail in _load_existing; newline guard in
  _append_row).
- `src/qemsel/model.py`: grouped CV (StratifiedGroupKFold, GroupKFold
  fallback; groups=(family,n_qubits,depth); n_splits also capped by group
  count); leave-one-family-out metric ('lofo'); permutation importance on
  held-out folds (training-set importances only in the flagged cv_folds=0
  path); label_column parameter (best_technique | best_technique_cost_aware,
  stored in bundle + metrics); ddof=1 fold std; drops legacy rows with
  cdr_abs_error < 1e-12.
- `src/qemsel/report.py`: cost model unified on err*sqrt(rel cost) with the
  experiment column; cost-aware win-rate + per-family tables in section 3;
  accuracy ± std, LOFO subsection, label + grouping + importance-provenance
  notes in section 5; degenerate-CDR rows excluded from all aggregates
  (count in section 1); _fmt inf-safe.
- `scripts/train_model.py`: --label switch; prints label, grouping and LOFO.
- `configs/small.yaml`, `configs/full.yaml`: pauli {ghz_plus: X, default:
  auto}; min_abs_ideal: 0.25 (documented in-file). tiny.yaml intentionally
  untouched (n=2 GHZ has <ZZ>=1; keeps the smoke run exercising the plain
  string-pauli path).
- `configs/hardware.yaml`, `README.md`: honest hardware wording + real seam
  list; quickstart points at shipped tiny.yaml with a tiny/small/full table;
  inline YAML re-framed as "your own config" under a new name; roadmap 2
  matches small.yaml; roadmap 5 documents the CDR regressor plug-in point.
- Tests: test_mitigation (Clifford + degenerate CDR now expect
  MitigationError; new ghz+t degeneracy test), test_backends_ideal (Lagos
  (3,4) noise regression fast; Jakarta/Sherbrooke pairs marked `slow`;
  width-guard test), test_experiment (torn-tail x2, per-family pauli,
  width validation, min_abs_ideal screening x2), test_model (new metric and
  bundle keys; seed-duplicate leakage regression under a null).
  `slow` marker registered in pyproject.toml.

## Decisions (conflicts / partial implementations)

1. Finding 3 offered three fixes "any/all". Implemented per-family
   observables + min_abs_ideal screening; did NOT implement
   significance-aware 'tie' labels: they change the label alphabet consumed
   by model/report/recommend (and every winner test), while the other two
   fixes remove the root cause (|ideal|~0 rows) and the planned
   seed-averaging covers genuinely-close winners. Revisit before the full
   run if seed-averaging proves insufficient.
2. Finding 5's "ideally add an empirical equal-budget raw baseline"
   deferred: it adds a pseudo-technique to the results schema (columns,
   winner logic, model classes, many tests) for a sub-item the reviewers
   themselves marked optional. The report's section 3 states the sqrt-proxy
   caveat explicitly instead.
3. Old `results/tiny` (pre-fix labels) moved to
   `results/tiny_prefix_artifacts/` and regenerated fresh — resume would
   otherwise have kept artifact rows alongside honest ones.

## Test / run status

- `pytest tests -q -m "not slow"`: 263 passed (4:41).
- `pytest tests -q -m slow`: 3 passed (0:52) — Jakarta/Sherbrooke noise.
- Tiny chain (experiment 30s -> train -> report -> recommend): green.
  Winners rem 11 / cdr 8 / zne 1; 8 honest CDR refusals in errors.log;
  recommend ghz_plus@FakeLagosV2 -> rem (p=0.98).
- `--label best_technique_cost_aware` smoke-tested: trains, 'raw' becomes a
  reachable class (labels cdr/raw/rem on tiny).
