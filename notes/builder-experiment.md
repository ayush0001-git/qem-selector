# Notes — builder-experiment (2026-07-21)

## Delivered (all owned files)
- `src/qemsel/experiment.py` — `run_experiment(config, out_dir)` implemented per stub contract.
- `configs/tiny.yaml` (10 circuits x 2 backends, 2000 shots), `configs/small.yaml`
  (60 x 2, 4000), `configs/full.yaml` (180 x 4, 8000 — "~200 circuits"; n_qubits
  capped at 5 = FakeManilaV2 size), plus `configs/experiment.yaml` (INTERFACES.md
  default: 2-3q, Manila+Lagos, 2048 shots) and `configs/hardware.yaml`
  (`ibm_token: null` placeholder — referenced by RealHardwareBackend). The last
  two were not in my task list but ARE my ownership per INTERFACES.md
  ("configs/ (all files)") and were requested in notes/architect.md open items.
- `scripts/run_experiment.py` — argparse `--config` (default configs/tiny.yaml)
  `--out` (default results/<config-stem>); prints winner distributions;
  `main(argv) -> int` is importable/testable.
- `tests/test_experiment.py` — 24 tests, all green in ~2.5s, zero simulation.

## Things the integrator MUST know
1. **EXTRA COLUMN beyond the interface schema:** `best_technique_cost_aware`,
   appended AFTER `best_technique` (last column). It is
   `argmin_t abs_error_t * sqrt(shots_t / base_shots)` (= abs_error * sqrt(SHOT_MULTIPLIER)),
   i.e. techniques are compared at equal shot budget: a technique only wins if
   its error reduction beats plain extra averaging (shot noise ~ 1/sqrt(shots)).
   `''` when all techniques failed, same as `best_technique`. Rationale
   documented in the experiment.py module docstring. conftest's
   `tiny_results_df` does NOT have this column — model/report should select
   columns by name (they do per interface), so this should be additive-safe.
   FLAG: architect note 5 says schema deviations need conftest sync; this is an
   append-only extension, `best_technique` semantics unchanged.
2. Winner tie-break: strict `<`, so FIRST technique in config order wins ties
   (raw before zne before cdr before rem with the default list).
3. Resume key = (`circuit_id`, `backend`) read from existing results.csv.
   Resuming with a config whose column set differs (e.g. different `techniques`
   list) raises ValueError — use a fresh out_dir.
4. Empty-string winners round-trip: on resume, NaN read back from CSV in the two
   winner columns is restored to `''` (tested).
5. Collaborators are called as module attributes (`_mitigation.apply_technique`
   etc.) so monkeypatching `qemsel.<module>.<fn>` works — keep it that way.
6. Per-technique failures: ANY `Exception` -> NaN triple + line
   `{circuit_id},{backend},{tech}: {exc!r}` in errors.log (there is no
   MitigationError class in mitigation.py; catching Exception covers whatever
   builder-mitigation raises). Failures in ideal/features/generate_suite are
   NOT caught — they abort the run (config/code bug, resume-safe anyway).
7. `run_meta.json` is overwritten at every (re)start; includes full config,
   versions (qiskit, qiskit-aer, mitiq, numpy, scikit-learn, pandas), python,
   qemsel version, ISO timestamp, `resumed_existing_rows` count.
8. Executor is built per (circuit, backend) unit with `seed=spec.seed`
   (per stub contract) — noise model rebuild each unit costs a bit but keeps
   units independently reproducible.
9. `pauli != 'auto'` is validated up front against every suite circuit's
   n_qubits (fail fast before any unit runs).

## Test strategy (why standalone-green while other modules are stubs)
All heavy deps monkeypatched: fake `generate_suite` (uses real `CircuitSpec`),
fake `ideal_expectation`, `extract_features`, `make_executor`,
`apply_technique`, `shots_consumed`. Covered: schema/order, row contents,
CSV==returned df, run_meta, winner==recomputed argmin (both columns),
cost-aware divergence case (cdr best raw error but raw wins cost-aware),
per-technique NaN isolation + errors.log format, all-fail `''` winner (+ CSV
resume round-trip), full/partial restart-skip with call counting, duplicate-free
CSV, config validation (missing keys / unknown backend / unknown technique /
bad shots / pauli mismatch), CLI main() happy + missing-config paths.

Command: `".venv\Scripts\python.exe" -m pytest tests/test_experiment.py -q`
-> 24 passed ~2.5s.
