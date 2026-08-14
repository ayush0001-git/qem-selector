# Notes — B4 experiment (shots axis) 2026-07-23

## Delivered (owned files only)
- `src/qemsel/experiment.py` — implemented `_normalize_shots`; threaded a
  `list_mode` flag through `run_experiment`, `_result_columns`,
  `_aggregated_columns`, `_write_aggregated` (all V2 params keyword-only,
  defaults = V1). Loop is now circuit x backend x **budget (innermost)**.
- `configs/boundary.yaml` (2430 units: 3 fams x {2,3,4}q x {4,8,16}d x 3 seeds
  x 10 backends x shots[256,1024,4096]) + `configs/boundary_smoke.yaml`
  (18 units). Both `feature_version: 2`, techniques incl. `zne_fr`/`cdr_ridge`.
- `tests/test_experiment_v2.py` — 40 tests (39 fast + 1 slow golden).

## Key decisions / constraints
1. `_validate_config` return arity is FROZEN (test_hardware unpacks 6-tuple,
   asserts scalar `shots==128`). So it returns raw `shots`; `run_experiment`
   re-derives `budgets,list_mode` + reads `feature_version` (validated in
   `_validate_config` but NOT returned).
2. feature_version 1 keeps the **exact V1 2-arg** `extract_features` call so
   test_experiment.py's 2-arg fakes still work; v2 passes `version,base_shots`.
3. Byte-identical proven: fresh `tiny.yaml` == `results/tiny/results.csv`
   (slow test) + all pre-V2 tests green.
4. errors.log gains `s{base_shots}` only in list mode; skip screen once per
   (circuit,backend); aggregated grouped by `_AGG_KEY_COLUMNS_V2`.

## Flags for integrator
- Task named configs `boundary.yaml`/`boundary_smoke.yaml`; INTERFACES.md V2.6
  said `boundary_sweep.yaml` — I followed the task. Rename if needed.
- Configs RUN only once B1 (zne_fr/cdr_ridge dispatch) + B5 (features v2) land;
  they validate now (names known). experiment.py is complete + tested vs fakes.
- Pre-existing unrelated failure: `test_hardware.py::test_shipped_hw_first_run
  _config_fits_two_minutes` (frozen `hw_first_run.yaml`, not mine).
