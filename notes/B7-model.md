# B7-model notes

Implemented V2 in `src/qemsel/model.py`, `scripts/train_model.py`,
`tests/test_model_v2.py`.

- `derive_significant_label`: per-seed (`<t>_value/_shots` -> `stats.sigma_shot`)
  and aggregated (`1-mean_err`, `n_seeds*shots_consumed`, needs `base_shots`)
  routes; winner unless `runner-winner < k_sigma*sqrt(sw^2+sr^2)` -> `tie`;
  `''` all-failed; lone technique -> itself.
- `train_and_eval` kwargs (all default-off, additive): `feature_version`
  (FEATURE_NAMES_BY_VERSION), `calibrate` (grouped-CV sigmoid
  CalibratedClassifierCV; `FrozenEstimator` prefit fallback — sklearn 1.9
  dropped `cv='prefit'`; metrics `calibration` Brier before/after),
  `abstain_threshold` (bundle + `abstain_rate_cv`), `extended_stats`
  (`fold_accuracies`+`fold_summary`). Auto `loso` when `base_shots>=2` vals.
- V2 bundle keys added only on a V2 path; default bundle/metrics
  byte-identical.
- CLI: `--label significant`, `--k-sigma --feature-version --calibrate
  --abstain-threshold --stats`.

CAPTURE-FIRST: pre-edit code reproduced stored `results/research/metrics.json`
exactly; regression test pins it (both labels).

Suite: 736 passed, 1 pre-existing UNRELATED fail
(`test_hardware.py` hw_first_run `hardware_confirmed=true`, config dated
07-22, outside B7 ownership). `stats` stubs (B6) monkeypatched per conv. 15.
