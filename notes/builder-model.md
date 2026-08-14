# Notes — builder-model (2026-07-21)

## Delivered
- `src/qemsel/model.py` — `train_and_eval(df, out_dir)` per INTERFACES.md.
- `scripts/train_model.py` — CLI. Primary flag is `--data`; `--results` is an
  accepted ALIAS (same dest) because INTERFACES.md said `--results` while my
  build spec said `--data`. Both work: `--data results.csv --out outdir`.
- `tests/test_model.py` — 11 tests, all green in 37s:
  `.venv\Scripts\python.exe -m pytest tests\test_model.py -q`.

## Behaviour details integrators should know
- Feature matrix: columns `['feat_'+n for n in features.FEATURE_NAMES]` in that
  exact order (only cross-module import: `qemsel.features.FEATURE_NAMES`, a
  module-level constant — safe even while features.py body is a stub).
- Drops rows with NaN/empty `best_technique` OR any NaN feature; raises
  ValueError on missing columns or zero usable rows.
- Model names: `random_forest`, `gradient_boosting`, baseline `dummy_majority`.
  Best non-dummy picked by out-of-fold macro-F1 (tie -> random_forest), refit
  on all rows, saved as the bundle dict {'model','feature_names','classes',
  'model_name','qemsel_version'} in `model.joblib`.
- Returned metrics keys are EXACTLY the 10 contract keys. Two additions INSIDE
  allowed structures (flagged): `per_model` also contains the dummy baseline
  entry, and each per_model entry has an extra `accuracy_std` key (spec
  required mean AND std for all three). Top-level keys unchanged; metrics.json
  == returned dict, fully JSON-serializable (plain python types).
- `feature_importances` = permutation importance (n_repeats=10, random_state=0)
  of the refit best model on all data — values can be ~0 or slightly negative
  for useless features; report.py should not assume they sum to 1.
- Class balance is printed to stdout (`[qemsel.model] ... class balance: ...`),
  not returned (kept return keys exact). cv fallback (<2 in smallest class)
  prints an honest WARNING and sets cv_folds=0; 2..4 -> reduced-fold message.
- CV: StratifiedKFold(shuffle=True, random_state=0); out-of-fold predictions
  drive accuracy/macro-F1/confusion matrix (rows/cols ordered by sorted labels).
