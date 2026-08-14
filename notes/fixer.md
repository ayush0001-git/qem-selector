# Fixer pass — verifier findings (2026-07-21)

Scope: the 8 findings from the research-pass verifiers (3 major, 5 minor).
All 3 majors fixed at root cause; 4 minors fixed; 1 minor (CDR-refusal
indicator feature, PROJECT_STATUS §6.2) deliberately left OPEN (changes the
frozen FEATURE_NAMES interface — a feature-ablation-section candidate, not
a quick fix). Full suite green; research_smoke chain re-verified end to end.
The research sweep had NOT started yet — every fix below lands before it.

## Finding 3 (major) — seed-averaged labels never consumable -> FIXED

Root cause: `aggregated.csv` had no `feat_*` columns, so
`train_and_eval(aggregated_df)` raised ValueError and every headline number
was trained on per-seed labels (which disagree with the seed-averaged winner
on 13/45 = 28.9% of smoke rows).

* `experiment.py::_write_aggregated` / `_aggregated_columns`: aggregated.csv
  now carries (a) seed-MEAN `feat_*` columns (directly trainable), (b)
  per-technique `<tech>_n_seeds` seed-coverage counts, and (c) a COVERAGE
  RULE for both winner columns: only techniques with the group's maximum
  seed coverage may win. This kills the related artifact (cdr "winning"
  3 ghz_plus smoke aggregates from its single non-refused seed vs
  competitors' 3-seed means) — those 3 groups now go to rem. If every
  technique misses a seed, the max-coverage set competes normally (group is
  not discarded).
* `scripts/train_model.py`: docstring + a printed nudge when trained on
  results.csv while a sibling aggregated.csv exists. README workflow step 2
  now trains on aggregated.csv.
* Smoke verification (fresh aggregated.csv, 15 rows, all 25 new columns):
  winners rem 7 / cdr 8; `train_model --data aggregated.csv --label both`:
  primary CV 0.733 +/- 0.435 (5-fold stratified_group) vs baseline 0.133,
  LOFO 0.733, LOBO 0.800, LODO 0.867 — read as "wired and plumbed", not as
  results (n=15).
* PROJECT_STATE's "§6.4 seed-averaged labels -> RESOLVED" was over-claimed
  (data existed, model couldn't eat it) — corrected in the changelog entry.

## Finding 4 (major) — LOBO scale-sibling leakage -> FIXED

6 of 9 research LOBO folds hold out e.g. FakeManilaV2@x1.5 while
FakeManilaV2 and @x2.0 (same circuits, bracketing backend features) stay in
training: that is noise-level INTERPOLATION, not a new environment.

* `model.py`: new metrics key `lodo` — leave-one-DEVICE-out, all `@x<scale>`
  siblings of one base device held out together ('accuracy', 'macro_f1',
  'per_device_accuracy', 'per_device_macro_f1', 'n_devices'; None for < 2
  devices or no backend column). LOBO kept but relabeled everywhere
  (docstrings, train_model summary, report §6) as the interpolation number;
  LODO is the paper's "new noise environment" headline. research.yaml gives
  3 LODO folds (Manila/Lagos/Jakarta).
* Report §6: LOBO section retitled "Noise-level interpolation
  (leave-one-backend-out)" with the caution spelled out; new LODO section
  "Generalization to a NEW noise environment (leave-one-device-out)";
  side-by-side label table carries both rows. Metrics dicts without `lodo`
  (legacy) still render.

## Finding 5 (major) — readout-cap dial compression on Lagos -> DISCLOSED

The 0.45 readout cap makes Lagos' nominal x1.5/x2.0 realize only
~x1.28/~x1.44 average readout scaling (q2: 0.464 -> 0.45 i.e. DECREASES at
x1.5; max_readout_error non-monotone). The cap itself is necessary
(uncapped q2 at x1.5 gives negative REM damping) — fix is disclosure +
realized numbers:

* `backends.py` module docstring: consequence spelled out with the measured
  realized factors; "quote realized get_backend_info numbers, not the
  nominal suffix".
* Report §5: new "Per-scale device composition and realized noise levels"
  table (realized feat_backend_avg_2q/readout per backend — smoke shows
  Lagos readout 0.204 -> 0.260 = x1.27, matching the verifier's 1.277) +
  caveat 1 (nominal vs realized); winner_vs_noise.png x-axis relabeled
  "NOMINAL noise scale".
* `configs/research.yaml`: "scale and topology unconfounded" comment
  replaced with the honest version + Lagos caveat.

## Finding 6 (minor) — min_abs_ideal conditioning disclosure -> FIXED

* `report.generate_report` gains optional `run_config`;
  `scripts/make_report.py` reads the `run_meta.json` sidecar next to --data
  and passes its config. Section 1 now renders a "Circuit-selection
  conditioning" bullet whenever min_abs_ideal > 0 (experiment or circuits
  level) OR — config absent — when bumped rejection-sampling seeds
  (>= SUB_SEED_STRIDE) appear in the data: accepted circuits are the
  atypical high-|<Z^n>| tail (worst for near_clifford, unconditioned median
  |ideal| = 0), family- and size-dependent; results are claims about
  "random circuits conditioned on |<Z^n>| >= t". Also notes the
  cross-family |ideal| magnitude differences that abs_error comparisons
  inherit.

## Finding 7 (minor) — x1.0 vs scaled noise-model family -> DISCLOSED

Report §5 caveat 2: plain x1.0 runs from_backend composite channels
(thermal relaxation), scaled runs pure depolarizing + symmetric readout —
the x1.0 -> x1.5 step changes noise CHARACTER (ZNE most sensitive); readout
symmetric on both paths so REM comparable. Same caveat added to
research.yaml's backends comment.

## Finding 8 (minor) — record corrections -> FIXED

* (a) Corrected in PROJECT_STATE changelog: smoke CDR refusals are
  near_clifford 9/9 + ghz_plus 6/9 (integrator had them swapped), and the
  ghz_plus refusals are degenerate-training-SPREAD refusals, not
  "fully-Clifford".
* (b) research.yaml header corrected: CDR pre-guards pass 1206/1620 units
  (74%), spanning ALL 5 families (layered/hw_eff/mirror 36/36 circuits
  each, ghz_plus 11/36, near_clifford 15/36) — the old "972 units,
  3 families" undersold it. (Answers Q4: PASS.)
* (c) Report §5 now prints per-scale device composition and WARNS when
  compositions differ across scales (fires on the asymmetric smoke config,
  by design; silent on the symmetric research config).
* raw_plus: §3 note added — in the cost-aware label it is a comparison
  column, structurally near-unwinnable (sqrt(11) penalty on a
  bias-dominated error), not a reachable class; 0 cost-aware wins is
  correct behavior.

## Finding 1 (minor) — tester task-spec nit -> RECORDED ONLY

Task wording said x2.0, smoke config has x1.5; monotonicity was verified at
both by the tester. No code change needed.

## Finding 2 (minor) — CDR-refusal indicator feature -> STILL OPEN

Adding a `cdr_refused` feature changes the interface-frozen FEATURE_NAMES
and every bundle/test downstream; PROJECT_STATUS §6.2 already tracks it as
a paper-ablation candidate. Not a quick fix — left open deliberately.

## Test changes (extensions only; nothing weakened)

* test_experiment.py: AGG_COLS extended to the new schema; new tests —
  feat columns are seed-means, partial-coverage technique cannot win,
  max-coverage fallback when no full group, aggregated.csv trains through
  `qemsel.model.train_and_eval` (the wiring regression).
* test_model.py: EXPECTED_KEYS + `lodo`; new tests — LODO pools scale
  siblings (4 backend strings -> 2 devices), LODO None for single-device
  multi-scale, LODO==LOBO on plain 2-backend data; lodo None without
  backend column.
* test_recommend_report.py: rich_metrics + lodo fixture; LODO/LOBO-relabel
  render assertions; legacy-metrics-without-lodo renders; TRUE aggregated
  schema (`<tech>_mean_abs_error`, no value/shots) renders with valid
  winners (was: phantom 'raw_mean' techniques, 0 valid rows); realized
  table + caveats render; unequal-composition warning; conditioning
  disclosure via run_config / bumped seeds / absent by default; make_report
  CLI passes run_meta.json config through.

## E2E verification (research_smoke)

1. `run_experiment` resume: 45/45 units skipped (results.csv untouched),
   aggregated.csv regenerated with the new 27-column schema.
2. `train_model --data aggregated.csv --label both`: both bundles trained,
   LODO printed, exit 0.
3. `make_report` (with run_meta.json sidecar): all new sections render —
   conditioning disclosure, realized-noise table, composition warning
   (correctly fires on the asymmetric smoke config), interpolation-relabeled
   LOBO, LODO headline section, raw_plus cost-aware note.
4. `recommend --demo ghz_plus --backend FakeLagosV2` -> rem (p=0.99), exit 0.

## Updated recommended research-run pipeline

```powershell
& ".\.venv\Scripts\python.exe" scripts\run_experiment.py --config configs\research.yaml --out results\research
& ".\.venv\Scripts\python.exe" scripts\train_model.py --data results\research\aggregated.csv --out results\research --label both
& ".\.venv\Scripts\python.exe" scripts\make_report.py --data results\research\results.csv --metrics results\research\metrics.json --out results\research
```

Headline numbers for the paper: LOFO (new circuit family) and LODO (new
noise environment / device, 3 folds) from the aggregated-label model; LOBO
is the scale-interpolation number. Optionally also train on results.csv for
a per-seed-vs-aggregated label ablation.
