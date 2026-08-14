# architect-v2 — V2 contract design (2026-07-22)

Defined the V2 contracts so 8 builders (B1-B8) work in parallel without
collisions. All stubs raise `NotImplementedError("V2 stub — builder-X ...")`.
Authoritative fine print is in the stub docstrings; cross-team summary +
ownership + regression duties are in `INTERFACES.md` section "V2".

## What I changed (all ADDITIVE — V1 surface frozen)

- `mitigation.py`: `TECHNIQUES_V2`, `SHOT_MULTIPLIER_V2`, `ZNE_FR_*`,
  `CDR_RIDGE_ALPHA`, `CDR_RF_*`, `CDR_SKLEARN_NUM_TRAINING_CIRCUITS`
  (all defined). Stubs: `richardson_coefficients`, `_apply_zne_fr`,
  `_apply_cdr_sklearn`. `apply_technique`/`shots_consumed` now validate on
  `TECHNIQUES_V2` and dispatch the 3 new names. V1 five untouched.
- `backends.py`: `LOW_NOISE_SCALES=(0.25,0.5)` (grammar already parses them;
  B2 verifies + pins the scaled path downward).
- `boundary.py` (NEW): `BoundaryParams`, `variance_k_q`, `estimate_params`,
  `delta_mse`, `regime`, `boundary_eps`, `boundary_shots`,
  `overlay_selector_vs_theory`. Sign convention FROZEN: delta_mse = MSE_raw
  - MSE_zne_fr, positive = zne_fr helps.
- `experiment.py`: `BASE_SHOTS_COLUMN`, `_AGG_KEY_COLUMNS_V2`,
  `_normalize_shots` stub carrying the whole shots-axis contract.
- `features.py`: `FEATURE_NAMES_V2` (+5), `FEATURE_NAMES_BY_VERSION`;
  `extract_features(..., *, version=1, base_shots=None)` — v1 byte-identical,
  v2 stub.
- `stats.py` (NEW): `sigma_shot`, `win_shares`, `bootstrap_ci`,
  `win_share_ci`, `paired_permutation_test`, `cliffs_delta`,
  `summarize_folds`, `koester_checklist` (all stubs). No qemsel imports.
- `model.py`: `SIGNIFICANT_LABEL`, `TIE_CLASS`, `derive_significant_label`
  stub; `train_and_eval`/`train_and_eval_all` gain keyword-only
  `feature_version/calibrate/abstain_threshold/extended_stats`
  (guard-stubbed; defaults byte-identical).
- `recommend.py`: `recommend(..., *, base_shots=None)` — V1 bundle path
  byte-identical, V2-bundle/abstain path guard-stubbed.
- `report.py`: `generate_report(..., *, stats_results=None,
  boundary_overlay=None)` — guard-stubbed sections 8/9; `_CANONICAL_TECHNIQUES`
  extended for display order.

## Key design decisions (so builders don't relitigate)

1. **Frozen V1 names, superset V2 names.** Regression tests pin
   `TECHNIQUES` (5), `SHOT_MULTIPLIER` (5 keys), `FEATURE_NAMES` (10). New
   capability rides `TECHNIQUES_V2`/`SHOT_MULTIPLIER_V2`/`FEATURE_NAMES_V2`
   and keyword-only params whose defaults reproduce V1 byte-for-byte.
2. **`richardson_coefficients` lives in mitigation, imported by boundary.**
   Single source of truth for nodes+coefficients — the zne_fr
   implementation and the K_q theory side can never disagree. mitigation
   never imports boundary (no cycle; matches the existing DAG note).
3. **zne_fr equal-split budget (shot multiplier 1).** The whole point of
   the Angle-3 overlay is an equal-budget ΔMSE(ε,B); splitting one base
   budget across the 3 levels makes the zne_fr-vs-raw comparison exactly
   the quantity the formula describes. `ZNE_FR_SHOT_ALLOCATION` lets the
   spike flip to 'full' if it disagrees (multiplier auto-derives).
4. **cdr_ridge/cdr_rf reuse `_apply_cdr`'s three fail-loud guards verbatim.**
   Angle 2 compares the cdr variants on the SAME accepted-row set, so the
   refusal conditions must be identical; only the regressor differs. The
   `generate_training_circuits` bypass (module docstring route 2) is the
   contracted default.
5. **Shots axis: seeds averaged WITHIN a budget.** `_AGG_KEY_COLUMNS_V2`
   adds `base_shots` after `backend` — the boundary needs per-budget
   labels, never labels averaged across budgets. Scalar-shots stays exactly
   V1 (no base_shots column, V1 agg keys).
6. **feature_version 2 REQUIRES a shots value.** `log2_shots` is THE axis
   that lets the selector's decision move along B; a v2 bundle without
   base_shots at recommend/overlay time is a ValueError, not a silent
   default. A shots-list config with feature_version 1 is legal but trains
   a shots-blind selector (its overlay region is flat in B — that flatness
   is itself a finding).
7. **eps axis = REALIZED get_backend_info numbers, never nominal '@x'.**
   Cap compression on Lagos (PROJECT_STATUS 4.10) means nominal != realized;
   the overlay and its caveats block must quote realized.
8. **stats.py has zero qemsel imports** so it sits at the bottom of the DAG
   (model imports stats; report consumes stats OUTPUT dicts only).

## Verification done

- All 9 modules import cleanly; constants print as expected (TECHNIQUES
  still 5, SHOT_MULTIPLIER still 5 keys, FEATURE_NAMES still 10).
- Fast test subset: 255 passed before the one PRE-EXISTING unrelated
  failure (`test_shipped_hw_first_run_config_fits_two_minutes` — the config
  legitimately ships `hardware_confirmed: true`, consented 2026-07-22 for
  the real n=3 run; I never touched hardware files). Full non-slow suite
  re-run deselecting that test to confirm zero regressions from V2 stubs.

## Landing-order note for builders

No hard ordering — every builder monkeypatches other builders' stubs in
unit tests (V1 pattern). Natural dependency chain for INTEGRATION tests
(put those in the later builder's file, mark slow):
B1(mitigation)+B5(features) -> B4(experiment) -> B7(model)+B6(stats) ->
B8(report/recommend); B3(boundary) depends on B1's `richardson_coefficients`
and B7's v2 bundle; B2(backends) is independent.
