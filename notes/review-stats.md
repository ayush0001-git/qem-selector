# Statistics review — model evaluation honesty (review-stats agent, 2026-07-21)

Scope: `src/qemsel/model.py`, `src/qemsel/report.py`, `scripts/train_model.py`,
`tests/test_model.py`, the produced `results/tiny/results.csv` + `metrics.json`
+ `report.md`. All claims below were verified by running code with the project
venv; scripts live in the session scratchpad (`check1_duplicates.py`,
`check2_leakage.py`, `check3_leakage2.py`, `check4_perm.py`).

## What checks out (no action needed)

- **Majority baseline is always shown next to model accuracy** — in
  `metrics.json` (`baseline_accuracy`), in the report's main table, in the
  per-model table (`dummy_majority` row is cross-validated with the same
  folds), and in the CLI summary. Good.
- **CV mechanics are valid at tiny sizes**: `n_splits = min(5, smallest
  class)` never exceeds the smallest class count, so StratifiedKFold cannot
  produce empty-class folds; the `< 2`-member fallback honestly sets
  `cv_folds = 0`, prints a warning, and report.md prints a warning block.
  Reproduced on `results/tiny/results.csv`: class balance cdr=16/rem=3/zne=1
  -> cv_folds=0, accuracy 0.800 == baseline 0.800, exactly as recorded.
- **Out-of-fold pooling**: accuracy/macro-F1/confusion matrix come from pooled
  out-of-fold predictions (not per-fold averages), which is the right choice
  when folds are tiny (per-fold macro-F1 would be undefined/noisy).
- **Permutation importance is used instead of impurity importance** — the
  right family of method (impurity importance would additionally be biased
  toward high-cardinality features). But see finding 2: it is computed on the
  wrong data.
- No target leakage into the feature matrix itself (X is strictly the 10
  `feat_*` columns; labels/errors never enter X).

## Findings

### 1. HIGH — Seed-duplicate rows leak across CV folds; group the folds by circuit config before the small/full-run evaluation

`train_and_eval` uses plain `StratifiedKFold` over (circuit, backend) rows.
But rows are not independent samples:

- Verified on the real `results/tiny/results.csv`: **16 of 20 rows share an
  exact-duplicate 10-feature vector with another row**. For 4 of 5 families
  (layered_random, near_clifford, hw_efficient_ansatz, mirror_circuit) the
  circuit features are **seed-invariant** — gate counts/depth are fixed by
  (family, n_qubits, depth); the seed only changes rotation angles, which no
  feature sees. layered_random and mirror_circuit even collide with EACH
  OTHER (identical vectors across families).
- The planned small/full configs use 3 seeds per config, so each
  (family, q, d, backend) cell will contribute ~3 byte-identical rows.
  StratifiedKFold(shuffle=True) puts identical rows in train and test; the
  model can score on a test row by memorizing its twin.

Quantified under a null model matching the observed data structure (cells with
identical features across seeds; per-cell winner assigned independently of the
features, rows agree with the cell winner with p=0.8 — the within-cell
consistency actually seen in tiny data; 120 rows like small.yaml; 10 draws):

| evaluation | accuracy |
| --- | --- |
| majority baseline | 0.480 ± 0.067 |
| current code (StratifiedKFold) | **0.682 ± 0.052** |
| GroupKFold by feature-cell | 0.367 ± 0.079 |

The current evaluation reports **+0.20 spurious "skill" over baseline** when
there is *zero* generalizable feature→label signal. Since the project's claim
(README) is recommending for a **new** circuit, this directly overstates the
headline result the small/full run will produce.

**Fix:** in `model.py`, cross-validate with
`StratifiedGroupKFold(n_splits, shuffle=True, random_state=0)` (fall back to
`GroupKFold` if stratification constraints fail) with
`groups = family + n_qubits + depth` (all seeds AND both backend rows of one
circuit config in the same fold — the deployment question is "new circuit on a
known backend", and the backend features are legitimately reusable). Requires
passing the `circuit_id`/`family`/`n_qubits`/`depth` columns (already in df)
into the CV, e.g. `groups = work['family'].astype(str) + '_q' +
work['n_qubits'].astype(str) + '_d' + work['depth'].astype(str)`.
Keep the ungrouped metric only if clearly labeled "seen-configuration
accuracy" — it answers a different (weaker) question.
Note: the planned seed-averaging of labels (PROJECT_STATE next steps) reduces
but does not remove this — the same config still appears on multiple backends
sharing 8/10 features.

### 2. MEDIUM/HIGH — Permutation importance computed on TRAINING data of the refit model

`model.py` refits the best model on ALL rows and calls
`permutation_importance(best_model, X, y)` on that same training data. For a
memorizing RandomForest this measures "which features index the memorized
rows", not which features generalize. Verified under the same null (no true
feature→label signal, training accuracy 0.825): training-data permutation
importance reports up to **+0.12** for `feat_backend_avg_2q_error` and
+0.08/+0.05 for readout error/clifford_fraction — pure memorization artifacts
that `feature_importances.png` would present as science (and LEARNING_GUIDE.md
explicitly says the importance narrative "matters for the write-up").

**Fix:** compute permutation importance on held-out data: inside the existing
CV loop, for each fold run `permutation_importance(fold_model, X_test, y_test,
n_repeats=10, random_state=0)` and average across folds (report std across
folds too if convenient). In the cv_folds=0 degenerate path, either skip
importances or label them "training-set (unreliable)".

### 3. MEDIUM — Fold-to-fold spread (`accuracy_std`) is computed but never shown in report.md

`metrics.json` carries `accuracy_std` per model and `train_model.py` prints
`mean ± std`, but `report.py::_section_model_eval` — the actual deliverable —
shows only point accuracies. At n≈120 with 5 folds the fold std will be
several percentage points; a point "model 0.68 vs baseline 0.52" claim without
spread is not supportable.

**Fix:** in `_section_model_eval`, render accuracy as
`f"{acc:.3g} ± {std:.2g}"` in both the headline and per-model tables (pull
`accuracy_std` from `per_model`; the baseline row has one too). Minor
sub-point: `np.std(fold_accs)` uses ddof=0; with 2–5 folds `ddof=1` is the
conventional choice — either is defensible if stated, but switch or state it.

### 4. LOW — Degenerate cv_folds=0 outputs are mislabeled "cross-validated"/"out-of-fold"

When `cv_folds == 0`, metrics come from predicting the training set, yet:
- report.md's table heading still says "Per-model **cross-validated** metrics"
  (see `results/tiny/report.md`), and
- the confusion-matrix figure title is hard-coded
  "Recommender confusion matrix **(out-of-fold)**" (`report.py::_save_confusion_matrix`).

The warning paragraph exists, but the PNG is self-contained and will be pasted
into slides without it. **Fix:** thread `cv_folds` into the two titles
("training-set (degenerate)" vs "out-of-fold").

### 5. LOW — Model selection and headline metric share the same CV (mild winner's curse)

The best of {random_forest, gradient_boosting} is chosen by macro-F1 on the
same out-of-fold predictions that are then reported as the headline accuracy.
With only 2 correlated candidates the optimism is small, and the per-model
table exposes both — acceptable for this project if the paper says "we report
the better of two models by CV macro-F1" explicitly. Nested CV would be the
strict fix; not worth it at this scale, but the sentence should be in the
report/paper.

## Notes for the report/paper wording

- The tiny-run report itself does not overstate: 0.800 vs baseline 0.800 with
  the cv_folds=0 warning is honest. The risk is entirely in the upcoming
  small/full evaluation (finding 1) and the importance narrative (finding 2).
- With only 2 backends, `backend_avg_2q_error`/`backend_avg_readout_error`
  take exactly 2 value-pairs — they are a backend ID in disguise. Fine for
  "known backend" deployment, but the paper must not claim the model learned a
  *continuous* noise-level relationship until ≥3–4 backends (full.yaml) exist.
