# Final trainer — research-run model training (2026-07-22)

Owner: MODEL TRAINER agent. Scope: `results\research` model artifacts, training
outputs, `report.md` regeneration. All numbers below come from runs executed in
this session (commands + artifact paths listed at the bottom).

## 0. Seed-averaged training path — resolved WITHOUT merge-back

The science review (notes\review-science-research.md Q2) prescribed a merge-back
fix because the smoke-era `aggregated.csv` lacked `feat_*` columns. **That is no
longer true**: the research `aggregated.csv` (540 rows) carries all 10 seed-mean
`feat_*` columns plus per-technique `<tech>_n_seeds` coverage counts, so it
trains directly through `model.train_and_eval` — verified, no merge-back needed.
The reviewer's underlying concern (headline numbers trained on per-seed labels)
is addressed by training the headline/production bundles on `aggregated.csv`.

Data verification (independent, plain pandas):
- results.csv: 1620 rows, aggregated.csv: 540 rows, 9 backends, 5 families,
  0 NaN-feature rows, 0 empty labels in either file.
- Per-seed winner tallies match the sweep-level counts exactly:
  best_technique cdr 1008 / rem 485 / zne 78 / raw_plus 37 / raw 12;
  cost-aware cdr 789 / rem 509 / raw 272 / zne 50.
- Seed-averaged tallies: best_technique cdr 321 / rem 172 / zne 25 /
  raw_plus 16 / raw 6; cost-aware cdr 236 / rem 189 / raw 109 / zne 6.
- **All 540 aggregate winners have full 3/3 seed coverage** — the smoke-era
  "1-of-3-seed winner" pathology (review Q2 tail) does not occur in the
  research data.
- 99/540 aggregate rows have best_technique != naive argmin of mean errors;
  **all 99 are the coverage rule working as designed** (the naive argmin is a
  partial-coverage technique, excluded from winning). 0 unexplained.
- Label-noise measurement: per-seed winner != seed-averaged group winner on
  **338/1620 = 20.9%** of rows (cost-aware 350/1620 = 21.6%) — the label noise
  seed-averaging removes (smoke estimate was 28.9%).

## 1. The four model runs

Two `train_model.py --label both` invocations = 4 bundles. LODO in `model.py`
already pools all `@x<scale>` siblings per device via `_base_device()` — no
manual GroupKFold needed; verified 3 device folds in every run.
`dropped_classes = []` in ALL four runs (smallest class raw=6 / zne=6 on
seed-avg — above the 2-member CV floor). CV = 5-fold StratifiedGroupKFold,
grouped by (family, n_qubits, depth), in all four runs.

### A. SEED-AVERAGED aggregated.csv (540 rows) — HEADLINE + PRODUCTION

**A1. best_technique** (accuracy-at-any-cost) — `model.joblib`, `metrics.json`
- best model: gradient_boosting
- grouped 5-fold CV accuracy **0.796 +/- 0.053**, macro-F1 **0.417**,
  majority baseline **0.594** (+0.202 over baseline)
- LOFO (new circuit family) **0.787** / macro-F1 0.440
  (ghz_plus 0.704, hw_efficient 0.852, layered_random 0.778, mirror 0.806,
  near_clifford 0.796)
- LOBO (scale interpolation) **0.893** / macro-F1 0.607
- LODO (new device — the honest new-noise-environment headline) **0.865** /
  macro-F1 0.357 (Jakarta 0.967, Lagos 0.694, Manila 0.933)
- class balance: cdr 321, rem 172, zne 25, raw_plus 16, raw 6
- top features: feat_n_non_clifford 0.240, feat_clifford_fraction 0.024,
  feat_backend_avg_readout_error 0.016

**A2. cost-aware** — `model_cost_aware.joblib`, `metrics_cost_aware.json`
(also embedded in metrics.json under `cost_aware`)
- best model: gradient_boosting
- CV accuracy **0.728 +/- 0.133**, macro-F1 **0.583**, baseline **0.437**
  (+0.291 over baseline)
- LOFO **0.702** / macro-F1 0.513 (ghz_plus 0.824, hw_eff 0.741,
  layered 0.574, mirror 0.583, near_clifford 0.787)
- LOBO **0.783** / macro-F1 0.573
- LODO **0.704** / macro-F1 0.428 (Jakarta 0.856, **Lagos 0.422**,
  Manila 0.833)
- class balance: cdr 236, rem 189, raw 109, zne 6
- top features: feat_n_non_clifford 0.173, feat_backend_avg_readout_error 0.145

### B. PER-SEED results.csv (1620 rows) — ABLATION, `per_seed\` subdir

**B1. best_technique** — best model: random_forest
- CV **0.772 +/- 0.037**, macro-F1 0.382, baseline **0.622**
- LOFO **0.712** / 0.334; LOBO 0.825 / 0.471; LODO **0.816** / 0.337
  (Jakarta 0.900, Lagos 0.648, Manila 0.900)

**B2. cost-aware** — best model: gradient_boosting
- CV **0.633 +/- 0.029**, macro-F1 0.467, baseline **0.487**
- LOFO **0.598** / 0.434; LOBO 0.756 / 0.575; LODO **0.688** / 0.429
  (Jakarta 0.819, Lagos 0.439, Manila 0.807)

### Ablation story (the paper's seed-averaging justification)

Seed-averaging removes ~21% label flips and improves EVERY held-out number:
CV +0.024 (0.772->0.796) / +0.095 (0.633->0.728); LOFO +0.075 (0.712->0.787) /
+0.104 (0.598->0.702); LODO +0.049 (0.816->0.865) / +0.016 (0.688->0.704),
accuracy-label / cost-aware respectively. Baselines also shift
(0.622->0.594, 0.487->0.437), so the above-baseline margin grows even more.

## 2. Production bundle verified

`recommend.py --model results\research\model.joblib` (seed-averaged,
accuracy label, gradient_boosting):
- `--backend FakeLagosV2 --demo ghz_plus` -> **rem** (p=0.989) — correct
  (fully-Clifford circuit, readout-dominated device)
- `--backend FakeManilaV2 --demo layered_random` -> **cdr** (p=0.974) — correct
  (non-Clifford circuit, low-noise device)

## 3. Report regenerated

`make_report.py --data results.csv --metrics metrics.json` ->
`results\research\report.md` (16.5 KB) + 5 PNGs all fresh (22:28 local):
confusion_matrix, error_by_technique, feature_importances, win_rate, and
**winner_vs_noise.png** (data spans 3 noise scales). Report §6 renders
LOFO/LOBO/LODO with the interpolation caution and the both-labels side-by-side
table (verified in the rendered text). NOTE: `results\research\figs\`
(error_vs_scale / winner_share_vs_scale / zne_win_region, 22:25 timestamps)
belongs to a parallel plotting agent — not touched.

## 4. Caveats to carry into the paper

1. **Macro-F1 is much lower than accuracy everywhere** (headline 0.417 CV,
   0.357 LODO). Accuracy is carried by cdr/rem; the minority classes (raw 6,
   raw_plus 16, zne 25 on seed-avg) are rarely predicted right on holdouts.
   Quote accuracy AND macro-F1 together; never accuracy alone.
2. **Lagos is the hard device in every holdout** (LODO 0.694 accuracy-label,
   0.422 cost-aware) — consistent with the review's Q6 cap-compression finding:
   Lagos' readout dial is compressed/non-monotone, and it is the one
   readout-dominated device, so a model trained on Manila+Jakarta has never
   seen REM-dominant physics. The cost-aware Lagos LODO fold (0.422) is the
   single weakest number in the run.
3. Cost-aware CV fold std is large (+/-0.133 on 5 folds) — quote the interval.
4. Winner's-curse disclosure still applies (best-of-2 model by CV macro-F1,
   PROJECT_STATUS §6.14); seed-avg picked gradient_boosting 2x, per-seed
   split RF/GB.
5. Review majors status: (1) seed-averaged label training — DONE (direct, no
   merge-back needed); (2) LODO beside LOBO — DONE (built into model.py, 3
   device folds verified); (3) disclosures (conditioned ensembles, Lagos cap,
   x1.0 model-family switch) — report §1/§5 carry the conditioning and
   realized-rate text; final paper text remains a writing task.

## 5. Artifacts (all under `E:\quatum  computiiing\qem-selector\results\research\`)

- `model.joblib` (655 KB, GB, seed-avg accuracy label — PRODUCTION),
  `model_cost_aware.joblib`, `metrics.json` (cost_aware embedded),
  `metrics_cost_aware.json`
- `per_seed\model.joblib`, `per_seed\model_cost_aware.joblib`,
  `per_seed\metrics.json`, `per_seed\metrics_cost_aware.json` (ablation)
- `report.md` + confusion_matrix / error_by_technique / feature_importances /
  win_rate / winner_vs_noise PNGs

Commands (venv python `E:\quatum  computiiing\qem-selector\.venv\Scripts\python.exe`):
```
train_model.py --data results\research\results.csv    --out results\research\per_seed --label both
train_model.py --data results\research\aggregated.csv --out results\research          --label both
recommend.py   --model results\research\model.joblib --backend FakeLagosV2 --demo ghz_plus
make_report.py --data results\research\results.csv --metrics results\research\metrics.json --out results\research
```
