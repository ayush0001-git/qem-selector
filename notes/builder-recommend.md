# Notes — builder-recommend (2026-07-21)

## Delivered (all owned files created; 20/20 tests green)

- `src/qemsel/recommend.py` — `recommend(model_path, circuit, backend_name)`
  implemented per contract. Loads the joblib BUNDLE, validates it
  (FileNotFoundError on missing file; ValueError on non-dict bundle, missing
  keys `model/feature_names/classes`, model without `predict_proba`, or
  class/proba length mismatch). Features via
  `qemsel.features.extract_features`, aligned to the bundle's
  `feature_names` order with the `feat_` prefix stripped; a clear ValueError
  ("feature mismatch ...") lists unmatched columns. Prediction row is passed
  as a **pandas DataFrame with the bundle's column names** so sklearn's
  fitted feature-name check passes (model.py trains on named columns —
  integrator: keep doing that). Class order for probabilities comes from
  `model.classes_` (fallback: bundle `classes`). Returns exactly
  `{'technique','probabilities','features'}`.
- `src/qemsel/report.py` — `generate_report(df, metrics, out_dir)` writes
  `report.md` + exactly `error_by_technique.png`, `win_rate.png`,
  `confusion_matrix.png`, `feature_importances.png` (Agg backend set before
  pyplot import; every figure closed). Sections 1-6 in contract order,
  numbers at 3 sig figs, PNGs referenced by relative filename. Includes:
  per-backend + pooled mean/median abs_error tables with NaN failure
  counts; cost-normalized table (mean shots/row, relative cost vs cheapest,
  cost-weighted error, total shots); winner counts overall/per-family/
  per-backend; model vs majority baseline + per_model table + cv_folds==0
  warning; reproducibility with package-version table. Techniques are
  auto-detected from `<tech>_abs_error` columns (canonical raw/zne/cdr/rem
  order first), so extra techniques won't break it. ValueError on missing
  df columns, no `_abs_error` columns, empty df, or missing metric keys.
- `scripts/recommend.py` — `--model --backend` + mutually exclusive
  `--qasm <file>` (qasm2.load, final measurements stripped) / `--demo
  <family>` (lazy import of `qemsel.circuits.FAMILIES`; extra `--qubits
  --depth --seed`, defaults 3/4/0). Prints result JSON. `main(argv)` is
  testable.
- `scripts/make_report.py` — `--data --metrics --out`; reads csv/json,
  calls `generate_report`, prints path. `main(argv)` testable.
- `tests/test_recommend_report.py` — 20 tests, all passing standalone in
  ~32 s via the venv python. Uses conftest `tiny_results_df`/`tiny_circuit`/
  `out_dir`; trains a real 10-tree RandomForest bundle; **monkeypatches
  `qemsel.recommend.extract_features`** so nothing depends on the (possibly
  unimplemented) features module. Covers: report path/headers/content/PNG
  size>1KB/NaN handling/all ValueError branches; recommend missing-file,
  malformed bundles, happy path (exact keys, probs sum to 1, argmax),
  determinism, feature mismatch; both CLIs (incl. QASM path with
  measurement stripping).

## Flags for integrator

1. **Test-file name**: my task assigned `tests/test_recommend_report.py`
   (one file) while INTERFACES.md ownership table says
   `tests/test_recommend.py` + `tests/test_report.py`. I followed the task.
   No file named test_recommend.py/test_report.py exists — don't expect them.
2. `recommend()` builds its X as a DataFrame with `feature_names` columns.
   If model.py ever trains on a bare ndarray, that still works (sklearn only
   warns/errors when names were seen at fit time), so no action needed.
3. `report.generate_report` needs metrics keys: best_model_name, accuracy,
   macro_f1, baseline_accuracy, labels, confusion_matrix,
   feature_importances (per_model/n_samples/cv_folds optional-with-default).
   That is a subset of the train_and_eval contract — compatible.
4. Demo CLI path (`--demo`) will raise NotImplementedError until
   builder-circuits lands; the `--qasm` path is fully functional now.
