# MODEL+REPORT BUILDER — research-scale upgrade (2026-07-21)

Owned files touched: `src/qemsel/model.py`, `src/qemsel/report.py`,
`tests/test_model.py`, `tests/test_recommend_report.py`. Nothing else edited.

**Verification: full suite `pytest tests -q` = 409 passed, 0 failed**
(4m21s; only benign mitiq short-circuit UserWarnings). Also smoke-tested on
the REAL `results/small/results.csv` (see bottom).

## 1. model.py — what changed

### (a) Singleton classes no longer collapse CV -> `dropped_classes`
- Classes with **< 2 members** are DROPPED from the CV evaluation (printed
  WARNING + recorded in new metrics key `dropped_classes: list[str]`). One
  lone zne win no longer forces the cv_folds=0 training-set fallback.
- The **refit-on-all bundle still trains on EVERY usable row** — dropped
  classes stay predictable at recommendation time (asserted by test:
  `'zne' in bundle['model'].classes_`).
- Honest edge cases kept: if after dropping < 2 classes remain (a
  single-class "CV" would score a meaningless 1.0, and GradientBoosting
  cannot even fit one class) or < 2 groups remain, the cv_folds=0 fallback
  still runs — evaluated on ALL rows, `dropped_classes == []` there by
  definition. Fold-reduction for small-but->=2 classes is unchanged
  (12 raw + 3 zne -> 3-fold, existing test preserved).
- New metrics key `cv_n_samples` = rows actually cross-validated
  (== n_samples unless classes were dropped). Confusion matrix now sums to
  cv_n_samples; dropped-class rows are all-zero rows in it.

### (b) Primary CV unchanged + hardened
- StratifiedGroupKFold grouped by (family, n_qubits, depth), GroupKFold
  fallback; `cv_grouping` records which ran. Fallback now also triggers if
  stratification returns an empty train/test fold (not just on raise).

### (c) LOFO + LOBO — the honest headline numbers
- `lofo` (existing key, extended): now ALWAYS computed when >= 2 families
  (no longer gated on cv_folds >= 2), on ALL usable rows, and carries
  `per_family_macro_f1` in addition to `per_family_accuracy` + pooled
  accuracy/macro-F1 + n_families.
- **NEW `lobo` key**: leave-one-backend-out, same shape
  (`accuracy`, `macro_f1`, `per_backend_accuracy`, `per_backend_macro_f1`,
  `n_backends`). Each distinct backend STRING is one held-out environment —
  noise-scaled names like `FakeManilaV2@x1.5` are separate environments, so
  this is the honest "generalizes to NEW noise" number. `lobo` is None when
  df has no backend column (graceful) or < 2 backends.

### (d) Two models when both label columns exist
- New public `train_and_eval_all(df, out_dir)`:
  - always trains `best_technique` -> `model.joblib` + `metrics.json`;
  - if `best_technique_cost_aware` exists with usable rows, also trains ->
    **`model_cost_aware.joblib` + `metrics_cost_aware.json`**;
  - then REWRITES `metrics.json` embedding the cost-aware metrics under a
    top-level `"cost_aware"` key -> the existing `make_report.py` CLI
    renders both label variants side by side with ZERO plumbing changes
    (report auto-detects the key). Returned dicts keep the exact
    single-label schema (no `cost_aware` key inside them).
  - legacy CSV without the column, or an all-empty column: prints a NOTE
    and returns `{'best_technique_cost_aware': None}` — never crashes.
- `train_and_eval` gained keyword-only `bundle_filename` /
  `metrics_filename` (defaults unchanged -> full backward compat with
  `scripts/train_model.py`, which still works as-is).

### (e) Aggregated schema accepted
- Nothing in model.py reads `seed`; `aggregated.csv` (n_seeds present, no
  seed column) trains identically. Explicit test added.

### Metrics schema (train_and_eval) — 3 NEW keys, none removed
`cv_n_samples` (int), `dropped_classes` (list[str]), `lobo` (dict|None);
`lofo` gained `per_family_macro_f1`. Everything else identical, still
JSON-round-trip clean.

## 2. report.py — what changed

Sections renumbered (NEW section 5): 1 Overview, 2 Technique comparison,
3 Cost-normalized view, 4 Win rates, **5 Noise-scale sweep**, 6 Model
evaluation, 7 Reproducibility.

- **`winner_vs_noise.png` — the money plot.** Parses `@x<scale>` from the
  backend column (plain name = scale 1.0; helper `_parse_backend`). Two
  stacked line panels: win rate per technique vs noise scale, and mean
  abs_error per technique vs noise scale. Written ONLY when >= 2 distinct
  scales exist; section 5 is always present and says "not applicable" for
  single-scale datasets (so old CSVs render fine and section numbering is
  stable). Section 5 also has markdown tables: per-scale win counts
  (+rates) with a "Top technique" column, and per-scale mean abs errors.
- **Model evaluation (§6):** accuracy ± fold-std everywhere it exists;
  dropped-classes note; LOFO table (accuracy + macro-F1 per held-out
  family) and NEW LOBO table (per held-out backend); "(N in CV after class
  drops)" annotation; **"Both winner labels side by side"** comparison
  table (best model, CV acc±std, macro-F1, baseline, folds, samples,
  dropped classes, LOFO/LOBO, classes) rendered when cost-aware metrics
  are available — via new keyword arg
  `generate_report(..., cost_aware_metrics=...)` OR auto-detected from
  `model_metrics['cost_aware']` (what train_and_eval_all writes).
- **Overview (§1):** notes seed-averaged datasets (`n_seeds` column) and
  lists parsed noise scales when > 1.
- **`raw_plus`** (empirical equal-budget baseline) added to the canonical
  technique ordering (right after raw) and §3 explains it when the column
  exists; when absent, §3 keeps the old sqrt-proxy caveat. Technique
  detection was already column-driven, so raw_plus flows through every
  table/figure automatically.
- Backward compat: PRE-upgrade metrics dicts (no lofo/lobo/
  dropped_classes/cv_n_samples/accuracy_std) still render — everything new
  is `.get()`-guarded; explicit regression test added.

## 3. Test changes (extend-don't-weaken accounting)

- `EXPECTED_KEYS` extended with the 3 new metrics keys.
- One test's expectation was SUPERSEDED BY SPEC (documented in the test
  docstring): old `test_degenerate_single_member_class_falls_back`
  (10 raw + 1 zne -> cv_folds == 0) split into
  `test_singleton_class_dropped_from_cv_not_collapsed`
  (rem 10/cdr 10/zne 1 — the actual small-run shape — asserts CV runs,
  zne dropped+recorded, refit keeps zne) and
  `test_single_surviving_class_falls_back` (10 raw + 1 zne still honestly
  falls back, keeping the old baseline assertion). All other existing
  assertions kept or strengthened.
- New model tests: aggregated schema, LOBO (2 backends incl. an `@x2.0`
  name, accuracy >= 0.8 on a learnable rule), lobo None without backend
  column, LOFO independent of CV feasibility, train_and_eval_all (both
  bundles + embedded cost_aware + legacy/no-column/empty-column paths),
  all-singletons fallback.
- New report tests (class `TestReportResearchPass`): multi-scale ->
  winner_vs_noise.png (>1 KB) + sweep tables; single-scale -> no PNG +
  "not applicable"; `_parse_backend` unit test (incl. malformed suffix);
  LOFO/LOBO tables; dropped-classes note; "0.75 ± 0.06" rendering;
  side-by-side via kwarg AND via embedded key + absence when no cost
  metrics; raw_plus detection; aggregated schema; legacy metrics dict.
- Section-header test updated to the new 7-section layout.

## 4. Real-data smoke (results/small/results.csv, 74 rows)

The zne-singleton problem from PROJECT_STATE "Final verification" is fixed:
the pipeline now reports what the verifier previously had to compute by
hand. `train_and_eval_all` output:
- primary: random_forest, **CV acc 0.822 (5-fold stratified_group) vs
  baseline 0.521, macro-F1 0.821**, dropped_classes ['zne'],
  n_samples 74 / cv_n_samples 73 — matches the independent verifier CV
  (0.823 ± 0.088) almost exactly; the old misleading 0.905 training-set
  number is gone.
- **LOFO 0.824** (per-family 0.60-0.92), **LOBO 0.797**
  (Manila 0.757 / Lagos 0.838).
- cost-aware model trained too (acc 0.658, 4-fold, labels incl. 'raw'),
  both bundles + both metrics files written, metrics.json embeds
  cost_aware, report renders all new sections, winner_vs_noise correctly
  absent (single scale).

## 5. Notes for other agents / integration

- `scripts/train_model.py` (not mine) still works unchanged; switching it
  (or the pipeline runner) to `qemsel.model.train_and_eval_all(df, out)`
  gets both bundles + the side-by-side report for free.
- `scripts/make_report.py` needs NO change: the embedded
  `metrics.json["cost_aware"]` key is auto-detected by generate_report.
- Contract I coded against for the experiment/aggregation agents:
  noise-scaled backends named `<Base>@x<scale>` in the `backend` column;
  equal-budget baseline as `raw_plus_value/_abs_error/_shots` columns;
  aggregated.csv = same columns minus `seed` plus `n_seeds`. If those
  agents chose different spellings, the seams to adjust are
  `report._NOISE_SCALE_RE`, `report._CANONICAL_TECHNIQUES`, and (for
  model.py) nothing — it only needs feat_*/family/n_qubits/depth/backend/
  label columns.
- Doc drift for the docs owner: report now has 7 sections and up to 5 PNGs
  (winner_vs_noise.png conditional); README/LEARNING_GUIDE mention 4.
- recommend.py bundle shape untouched (`label_column` key already existed);
  `model_cost_aware.joblib` is loadable by the same recommend() today via
  `--model .../model_cost_aware.joblib`.
