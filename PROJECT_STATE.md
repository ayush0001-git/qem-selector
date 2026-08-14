# QEM-Selector — Project State Log

## 2026-07-21 — Project kickoff (environment setup agent)

**Project goal:** Build a benchmark-and-recommend system for quantum error mitigation (QEM). We will benchmark three QEM techniques — Zero-Noise Extrapolation (ZNE), Clifford Data Regression (CDR), and Readout-Error Mitigation (REM) — on noisy simulated quantum backends (Qiskit Aer with noise models derived from IBM fake backends). For a suite of benchmark circuits we will measure how much each technique improves observable-expectation accuracy versus the unmitigated result, extract per-circuit features (qubit count, depth, gate composition, two-qubit gate count, entanglement structure, etc.), and train a supervised ML classifier (scikit-learn) that, given a new circuit's features, recommends the QEM technique expected to perform best. Deliverables: benchmarking pipeline, feature extractor, labeled dataset, trained classifier, and evaluation report.

## 2026-07-21 — Environment setup COMPLETE (environment setup agent)

**Status: all verification checks passed** (`scripts/verify_env.py` exits 0).

- **Venv:** `E:\quatum  computiiing\qem-selector\.venv` (Python 3.12.3). Use `.venv\Scripts\python.exe` for everything. NOTE: parent dir has DOUBLE spaces — always quote paths.
- **Pinned versions** (full list in `requirements.txt`): qiskit 2.5.0, qiskit-aer 0.17.2, qiskit-ibm-runtime 0.48.0, **mitiq 1.0.0**, cirq-core 1.6.1, numpy 2.2.6, scipy 1.17.1, scikit-learn 1.9.0, pandas 2.3.3, matplotlib 3.11.1, PyYAML 6.0.3, pytest 9.1.1, joblib 1.5.3, ply 3.11.
- **Gotcha fixed:** mitiq's qiskit frontend imports `cirq.contrib.qasm_import`, which needs `ply` — not installed by default with mitiq/cirq-core. `ply==3.11` is now installed and in requirements.txt. Without it, `execute_with_zne` on a qiskit circuit raises `ModuleNotFoundError: No module named 'ply'`.
- **Fake backends** (from `qiskit_ibm_runtime.fake_provider`): 41 `Fake*V2` classes exist. **Verified working with `NoiseModel.from_backend` + AerSimulator:** `FakeManilaV2` (5q, Bell P(00)+P(11)=0.94), `FakeJakartaV2` (7q, 0.95), `FakeSherbrooke` (127q Eagle, no V2 suffix, 0.97), `FakeLagosV2` (7q, 0.74 — its stored calibration has ~27% readout error on q0/q1; measured empirically with a measure-only circuit). FakeLagosV2 is a good high-readout-error target for REM benchmarks; use FakeManilaV2/FakeJakartaV2 as defaults.
- **mitiq ZNE smoke test passed:** `zne.execute_with_zne(qiskit_circuit, executor)` works end-to-end on a 1-qubit circuit with a noisy AerSimulator executor (unmitigated 0.966 → mitigated 0.957; no improvement expected here since the toy circuit's error is readout-dominated, which ZNE does not target — the point was that the machinery runs).
- **mitiq 1.0.0 top-level modules** — `dir(mitiq)`: `Bitstring, Calibrator, Executor, MeasurementResult, Observable, PauliString, QPROGRAM, QuantumResult, SUPPORTED_PROGRAM_TYPES, Settings, ZNE_SETTINGS, about, benchmarks, calibration, cdr, compare_cost, ddd, execute_with_mitigation, executor, interface, lre, observable, pec, pt, qem_methods, qse, raw, rem, typing, utils, zne`. Entry points confirmed importable: `mitiq.zne.execute_with_zne`, `mitiq.cdr.execute_with_cdr`, `mitiq.rem.execute_with_rem`.
- **Transpile tip:** pass only the AerSimulator (built with the noise model) to `transpile(...)`; also passing `basis_gates=` triggers a qiskit 2.5 UserWarning about invalidating backend error rates. Use `optimization_level=0` in executors so mitiq's folded gates are not optimized away.
- **Files:** `scripts/verify_env.py` (rerunnable env check), `requirements.txt` (exact pins), `notes/env-setup.md` (agent notes).

## 2026-07-21 — Integration COMPLETE (integrator agent)

**Status: full suite 251/251 green; tiny end-to-end chain (experiment -> train -> report -> recommend) verified on real noisy simulation.**

### What broke between modules (and fixes)

Remarkably little — the 8 builders' self-reports were accurate and the first full
`pytest tests -q` run passed 249/249 with zero integration failures. Three real
seams were found while running the LIVE tiny pipeline (things unit tests with
monkeypatched collaborators could not see):

1. **mirror_circuit ideal was not exactly +1.0** (`src/qemsel/ideal.py`).
   Statevector simulation of U then U-dagger leaves float dust: seed-0 mirror
   rows recorded `ideal = 0.9999999999999996`, violating the architect's
   known-answer contract ("ideal EXACTLY +1 for 'Z'*n"). Fix: `ideal_expectation`
   now snaps results within 1e-10 of an integer (-1/0/+1) to that integer —
   far above float dust (~1e-16), far below any physically meaningful deviation.
   Two regression tests appended to `tests/test_backends_ideal.py`. The 4 mirror
   rows were dropped from results.csv and recomputed via the resume path (which
   also live-verified crash-safe resume: 16 skipped, 4 recomputed).
2. **Feature-name seam between model.py and recommend.py** (`src/qemsel/model.py`).
   model.py fitted on a bare ndarray while recommend.py predicts with a named
   DataFrame -> sklearn UserWarning "X has feature names, but ... was fitted
   without feature names" on every recommendation, and no name validation.
   Fix: model.py now fits on the named feat_* DataFrame (CV split via .iloc), so
   sklearn stores and VALIDATES column names at predict time. One test in
   `tests/test_model.py` updated to predict with a named row (the documented
   bundle-consumption pattern) instead of an ndarray — a strengthening, not a
   weakening.
3. **README CLI drift** (`README.md`, flagged by builder-docs). Step 3 used
   `make_report.py --results` (actual flags: `--data --metrics --out`) and step 4
   called `recommend.py` without the required `--qasm`/`--demo` source argument.
   Both commands corrected to the implemented flags.

Known accepted quirks (documented, not bugs): CDR short-circuits fully-Clifford
circuits to the ideal value (mitiq behavior — biases ghz_plus/mirror winners
toward cdr; visible in the stats below); `best_technique_cost_aware` is an
append-only extra column; `tests/test_backends_ideal.py` and
`tests/test_recommend_report.py` are combined files (task assignments overrode
the INTERFACES.md ownership-table filenames); `.gitignore` ignores
`configs/hardware.yaml` (token safety — commit a `.example` copy if this ever
becomes a git repo).

### Tiny run stats (configs/tiny.yaml -> results/tiny, 20 units, 2000 base shots)

- Rows: 20 (10 circuits x FakeManilaV2/FakeLagosV2); errors.log absent.
- NaN rate per technique: raw 0%, zne 0%, cdr 0%, rem 0% (all << 30% gate).
- mirror_circuit ideal == +1.0 exactly on all 4 rows (post-fix).
- Winner breakdown (best_technique): cdr 16, rem 3, zne 1, raw 0.
  Cost-aware: cdr 14, rem 5, raw 1. raw never wins on accuracy -> mitigation
  is correctly wired (every technique beats raw where it wins).
- Per-technique mean abs_error (pooled | Manila | Lagos):
  raw 0.237 | 0.090 | 0.384; zne 0.228 | 0.075 | 0.382;
  cdr 0.023 | 0.021 | 0.025; rem 0.029 | 0.023 | 0.035.
  REM's edge on readout-dominated Lagos and CDR's on Manila match the
  builder-mitigation smoke study; cdr dominance is partly the Clifford
  short-circuit (ghz_plus/mirror/near_clifford families).
- Model: 20 samples, class balance cdr=16/rem=3/zne=1 -> honest cv_folds=0
  fallback; best random_forest, accuracy 0.800 == majority baseline, macro-F1
  0.296 (identical feature vectors with conflicting labels across seeds — a
  data-scale property, expected at 20 rows; the small/full runs are the real
  evaluation).
- Report: results/tiny/report.md + 4 PNGs (all > 1 KB) generated.
- Recommend CLI verified on 2 families: ghz_plus(3,4,0)@FakeLagosV2 -> cdr
  (p=0.90), layered_random(2,4,7)@FakeManilaV2 -> cdr (p=0.78); no sklearn
  warnings.

### Next steps
- Run configs/small.yaml (60 circuits) for the first meaningful model eval.
- Consider seed-averaging labels before the full run (winner flips between
  close techniques are label noise at 2000 shots).

## 2026-07-21 — Review fixes applied (enhancement-applier agent)

**All 4 reviewers' actionable findings applied (7 HIGH-class, 7 MEDIUM);
full fast suite 263/263 green (+3 new slow-marked regression tests, also
green); tiny chain re-run end-to-end on the fixed pipeline.** Detail in
`notes/enhancement-applier.md`. Changelog:

### Scientific correctness (HIGH)
1. **CDR fail-loud guards** (`mitigation.py::_apply_cdr`): raises
   `MitigationError` when (a) the compiled circuit is fully Clifford (mitiq
   1.0.0 short-circuits to OUR ideal simulator value — zero error by
   construction) or (b) all pre-generated training-circuit ideals are
   identical (`np.ptp < 1e-9`; regression collapses to the classical
   constant). The pre-fix "CDR wins 80%" was this artifact: post-fix tiny
   winners are rem 11 / cdr 8 / zne 1 (was cdr 16 / rem 3 / zne 1), with 8
   honest `[cdr]` refusals in errors.log (4 Clifford ghz_plus, 4 degenerate
   near_clifford). Legacy-data guard: rows with `cdr_abs_error < 1e-12` are
   excluded from model training AND all report aggregates (count shown in
   report §1). The old "known accepted quirk" about the Clifford
   short-circuit is hereby superseded.
2. **2q noise on uncoupled pairs / wrong-direction ECR**
   (`backends.py::make_executor`): simulator is now
   `AerSimulator.from_backend(backend)` and the per-call transpile runs
   against it, so routing + ECR direction-fixing happen and every 2q gate
   carries device noise. Verified: Lagos pair (3,4) deep-cx now decays
   0.91→0.10 (was EXACTLY 0.0). Regression tests: fast (Lagos (3,4)) + slow
   (`-m slow`: Jakarta (3,4), Sherbrooke (0,1)/(3,4)). full.yaml n=4,5 on
   Lagos/Jakarta/Sherbrooke is now physically meaningful; routed SWAP
   overhead is device truth.
3. **Near-zero-ideal lottery labels** (finding: 42% of full.yaml has
   |ideal|<0.1): (a) config `pauli` now accepts a per-family dict
   (single char repeats to width) — small/full.yaml set `ghz_plus: X`
   (<X^n>=+1 at every n; odd-n GHZ <Z^n>=0 rows are gone); (b) new config
   key `min_abs_ideal` (small/full: 0.25) skips low-signal units before any
   noisy work, logged to `skipped_low_signal.log`. DECISION: the reviewers'
   third option (significance-aware 'tie' labels) NOT implemented — it
   changes the label alphabet consumed by model/report/recommend; the
   observable+screening fixes remove the root cause (|ideal|≈0) and the
   already-planned seed-averaging addresses genuinely-close winners. Also
   new: executor + `_validate_config` reject circuits wider than the
   backend (extra qubits simulated noiselessly before).

### Statistical honesty (model/report)
4. **Grouped CV** (`model.py`): `StratifiedGroupKFold` (GroupKFold
   fallback) with groups=(family, n_qubits, depth) — seed/backend
   duplicates of one config can no longer straddle train/test (pre-fix
   +0.20 spurious skill under a null). New leakage regression test.
5. **Leave-one-family-out** metric (`lofo` key + report §5) = headline
   "new circuit" generalization number.
6. **Permutation importance on held-out folds** (mean over CV folds);
   cv_folds=0 path labels importances "training-set (UNRELIABLE)" in
   metrics, report and the PNG title.
7. **accuracy ± std** (ddof=1) rendered in report §5 tables.
8. **Cost model unified on sqrt** (report §3 == experiment's
   `best_technique_cost_aware`), cost-aware win-rate + per-family tables
   added to §3 from the stored column, and `train_model.py --label
   best_technique_cost_aware` trains the equal-budget model ('raw' becomes
   a reachable class; verified on tiny). DECISION: empirical equal-budget
   baseline (raw at 11x shots as a pseudo-technique) NOT implemented —
   schema/tests churn out of proportion to a "ideally add" sub-item;
   report §3 carries an explicit caveat instead.

### Robustness / docs (MEDIUM)
9. **Torn-CSV self-heal** (`experiment.py`): `_load_existing` truncates a
   partial final line (crash mid-append) with a log message and the unit is
   recomputed; `_append_row` writes a newline first if the last byte isn't
   one. Both tested (pre-fix: poisoned done_pairs or permanent ParserError).
10. **Hardware stub honesty**: stub message, `hardware.yaml` comment and
    README now say the class must be IMPLEMENTED (token alone does nothing)
    and list the real dispatch seams (BACKENDS / get_backend_info /
    make_executor / _validate_config).
11. **CDR regressor plug-in point documented** (mitigation.py docstring +
    README roadmap 5): `CDR_FIT_FUNCTION`/`CDR_NUM_FIT_PARAMETERS`
    constants pass through to `execute_with_cdr` (parametric fits);
    sklearn regressors need the `generate_training_circuits` route (the
    degeneracy guard already generates the training set to extend from).
12. **README quickstart** points at shipped configs\tiny.yaml (table of
    tiny/small/full with sizes/durations), inline YAML re-framed as
    "write your own config" under a NEW filename, roadmap step 2 aligned
    with the real small.yaml.

### Tiny chain re-run (post-fix)
- Old pre-fix data preserved at `results/tiny_prefix_artifacts/`;
  `results/tiny/` regenerated (30 s): 20 rows, cdr NaN on 8
  (ghz_plus+near_clifford) with honest errors.log lines; winners
  rem 11 / cdr 8 / zne 1; cost-aware rem 13 / cdr 6 / raw 1.
- train (cv_folds=0 honest fallback, acc 0.80 vs baseline 0.55), report
  (all new sections render), recommend (ghz_plus@Lagos -> rem 0.98 — the
  pre-fix 'cdr' answer was the artifact) all green.
- NOTE for small/full runs: `results.csv` schema unchanged, but labels are
  no longer comparable to any pre-fix CSV (CDR refusals, X observable for
  ghz_plus, low-signal screening). Use fresh out_dirs.

## Final verification (small run)

**2026-07-21 — experiment verifier agent. Full chain re-run from scratch on
`configs/small.yaml` into fresh `results/small/`. Verdict: e2e PASS —
completed end-to-end on real noisy simulation, outputs scientifically sane.
Detail in `notes/verifier.md`.**

- **Tests:** `pytest tests -q` (slow included): **266/266 passed**, 0 skipped
  (77 s; only benign mitiq short-circuit UserWarnings).
- **Experiment:** exit 0. 120 units attempted; 46 screened pre-noise by
  `min_abs_ideal=0.25` (skipped_low_signal.log); **74 rows** in results.csv
  (37 per backend, all unique, winner==argmin on every row). errors.log: 30
  lines, ALL intentional CDR fail-loud refusals; 0 tracebacks.
- **NaN rate per technique:** raw 0%, zne 0%, rem 0%, **cdr 40.5% (30/74)** —
  above the 30% gate, reported honestly: all 30 are the designed
  MitigationError refusals (ghz_plus 22/24 fully-Clifford, near_clifford 8/8
  degenerate training ideals). Structural (Clifford-heavy suite + honest
  guards), not a crash — but CDR carries no signal on 2 of 5 families.
- **Winners (best_technique):** rem 38 / cdr 35 / zne 1 / raw 0.
  Cost-aware: rem 37 / cdr 32 / raw 4 / zne 1.
- **Mean abs_error (pooled | Manila | Lagos):** raw .423|.209|.637,
  zne .391|.158|.623, cdr .040|.012|.068 (non-Clifford rows, 11x shots),
  rem .102|.053|.151 — physically consistent (Lagos readout-dominated).
- **Model (honest):** zne singleton class -> pipeline cv_folds=0 fallback;
  metrics.json accuracy **0.905** / macro-F1 0.608 vs baseline 0.514 is
  **training-set (optimistic)** and flagged as such by the pipeline.
  Independent verifier CV (zne row dropped, n=73, StratifiedGroupKFold(5) by
  (family,n_qubits,depth)): **0.823 +/- 0.088** vs grouped-majority baseline
  0.521, macro-F1 0.821; leave-one-family-out **0.808**. At ~74 rows these
  are noisy (+/- ~0.09 across folds) — read as "clearly above baseline",
  not a precise number.
- **Report:** results/small/report.md + 4 PNGs (29–45 KB each, all > 1 KB).
- **Recommend:** ghz_plus q3@FakeLagosV2 -> rem (p=1.0);
  layered_random q2 s7@FakeManilaV2 -> cdr (p=1.0); no sklearn warnings.
- **For full run:** (1) full.yaml needs enough zne wins for cv_folds>0 (or
  min-class handling in model.py); (2) 38% low-signal screening skews the
  family mix (ghz_plus/mirror 24 rows vs layered_random/near_clifford 8) —
  consider rebalancing; (3) document/feature-ize CDR refusals.

## 2026-07-21 — Research-run integration COMPLETE (integrator agent)

**Status: full suite 411/411 green (twice: pre- and post-integration edits);
fresh research_smoke end-to-end verified with independent pandas recomputation;
tiny.yaml untouched-path guarantee re-proven BYTE-IDENTICAL;
configs/research.yaml resized to 9 noise environments, final estimate ~4.9 h.**

### 1. Test suite
`pytest tests -q` (slow included): **411 passed / 0 failed** (307 s), only the
3 benign mitiq short-circuit warnings. Zero cross-agent integration failures —
the three parallel builders' seams (scaled-name grammar in `_validate_config`,
`raw_plus` in `report._CANONICAL_TECHNIQUES`, `@x<scale>` in
`report._NOISE_SCALE_RE`) were all already wired to each other's spellings.
Suite re-run after the integration edits below: green again.

### 2. Fresh end-to-end on configs/research_smoke.yaml
Config amended at integration: plain `FakeLagosV2` added as a 3rd backend so
the smoke data contains a scaled backend AND its x1.0 sibling (the
"scaled noise really is noisier" check needs both in one dataset) -> 45 units.
Fresh run (old results/research_smoke deleted): **45/45 units, 89.9 s
(2.0 s/unit blended), exit 0**. Independent pandas verification
(scratchpad/verify_smoke.py, 20 checks, ALL PASS):

- `raw_plus_*` columns present; `raw_plus_value != raw_value` on 45/45 rows;
  `raw_plus_shots == 11x raw_shots`; pooled mean |err| raw_plus 0.5604 <=
  raw 0.5615. The tiny margin is the correct physics: raw error is noise
  BIAS, not shot variance — which is exactly the point of the equal-budget
  control (mitigation wins survive giving "just take more shots" 11x budget).
- Per-row `best_technique` == recomputed argmin over non-NaN abs errors (45/45).
- `aggregated.csv` (15 groups, n_seeds=3 everywhere): every
  `<tech>_mean_abs_error` matches an independent NaN-skipping mean to <1e-12;
  `best_technique` == argmin over means on 15/15; `best_technique_cost_aware`
  == argmin over sqrt(shots_consumed/base)-penalized means on 15/15.
- Noise scaling: FakeLagosV2@x1.5 mean raw error 0.7151 > plain 0.7033;
  paired per-circuit the scaled row is worse on 87% of 15 pairs.
- NaN audit: raw/raw_plus/zne 0%; cdr 33% (15/45 — ghz_plus 9/9 fully-
  Clifford + near_clifford 6/9 degenerate-training refusals, ALL intentional,
  each logged); rem 4.4% (2/45 honest damping-floor refusals on the Lagos
  variants). **CDR has real values on 18/18 layered_random + hw_efficient
  rows and 9/9 mirror rows** — its label signal now spans 3 of 5 families.
- errors.log: 17 lines, all `[cdr]`/`[rem]` refusals, zero tracebacks.
- Train: `train_model.py --label both` (NEW CLI mode, see §4) -> primary
  model gradient_boosting **CV 0.727 +/- 0.174 (5-fold stratified_group) vs
  baseline 0.341**, zne dropped from CV as singleton (warned + recorded),
  LOFO 0.711 / LOBO 0.689 (3 backends); cost-aware model trained to
  model_cost_aware.joblib (CV 0.533 vs baseline 0.556 — below baseline at
  45 rows; smoke-scale noise, the research run is the real evaluation) and
  embedded under metrics.json['cost_aware'].
- Report: report.md with all 7 sections + 5 PNGs incl. winner_vs_noise.png
  (2 scales), raw_plus in every table, side-by-side label comparison,
  LOFO/LOBO tables, dropped-class note.
- Recommend: ghz_plus q3@FakeLagosV2 -> rem; layered_random
  q3@FakeLagosV2@x1.5 -> cdr (scaled backend features exactly 1.5x the
  plain ones in the printed feature dict). Both physics-consistent.

### 3. Untouched-path guarantee re-proven (tiny.yaml)
Fresh run into results/tiny_recheck: `results.csv` **byte-identical** to the
pre-noise-scaling results/tiny/results.csv (cmp), winners exactly
**rem 11 / cdr 8 / zne 1**, cost-aware rem 13 / cdr 6 / raw 1. The
noise-scaling change did not perturb plain-name backends by a single byte.
(tiny.yaml pins `techniques: [raw, zne, cdr, rem]`, so raw_plus correctly
does not run there.) New behavior: tiny now also emits aggregated.csv
(10 groups, n_seeds=2) — additive, schema-correct.

### 4. Integration edits (beyond verification)
1. **scripts/train_model.py**: new `--label both` mode wired to
   `model.train_and_eval_all` — trains BOTH label variants in one call
   (model.joblib + model_cost_aware.joblib + embedded cost_aware metrics).
   Needed because running the CLI twice with different `--label` values
   would clobber model.joblib/metrics.json (both labels wrote the same
   filenames). Summary printer extended (dropped classes, LOBO line).
   Existing single-label modes unchanged; test_model.py 21/21 green.
2. **configs/research.yaml resized 7 -> 9 environments** (task: land the
   estimate in 4-6 h): added FakeJakartaV2@x1.5 and @x2.0, making a
   symmetric 3-device x 3-scale design (scale and topology unconfounded).
   1620 units; 5.4 s/unit measured blend -> 2.43 h basis, **x2 margin ->
   ~4.9 h FINAL ESTIMATE** (header updated; fresh smoke re-measured
   2.0 s/unit at n=3, so 5.4 is conservative). Aggregated output will be
   540 rows; CDR-viable units 972/1620.
3. **configs/research_smoke.yaml**: 3rd backend (plain Lagos) + header
   updates (45 units, x1.0-sibling rationale).
4. **Stale Lagos "~27% q0/q1" comments** fixed in tiny/small/full.yaml
   (PROJECT_STATUS §6 item 9; backends.py copy already fixed by the
   noise-scaling builder). The same stale wording in the 2026-07-21 env-
   setup entry above is historical record — corrected here, not rewritten:
   actual stored readout errors are q0 16.9%, q1 13.6%, q2 46.4%.
5. **README**: workflow step 2 now shows `--label both`; step 3 mentions
   the winner-vs-noise-scale plot.

### 5. State of the §6 unresolved items after this pass
- (1) zne singleton -> RESOLVED (model.py drops singletons from CV, keeps
  them in refit/LOFO/LOBO). (3) family skew -> RESOLVED at source
  (generate_suite min_abs_ideal rejection sampling; 180/180 balanced).
- (4) seed-averaged labels -> RESOLVED (aggregated.csv every run, winners
  from means). (6) equal-budget baseline -> RESOLVED (raw_plus).
- (8) REM damping floor -> RESOLVED (0.02, refusals observed live).
  (9) stale Lagos comment -> RESOLVED everywhere.
- (2) CDR-refusal indicator feature — still OPEN (model learns it only via
  clifford_fraction); candidate for the paper's feature-ablation section.
- (5) tie labels, (7) REM affine offset, (10) recommend.py traceback,
  (12-14) — unchanged, none block the research run.

### Next step
Run the research sweep (resumable, ~4.9 h estimate):
`& ".\.venv\Scripts\python.exe" scripts\run_experiment.py --config configs\research.yaml --out results\research`
then `train_model.py --label both`, `make_report.py`, and compare LOBO
across the 9 environments — that LOBO number is the paper's headline
"generalizes to a new noise environment" claim.
(SUPERSEDED by the fixer entry below: train on aggregated.csv, and the
headline "new noise environment" number is LODO, not LOBO.)

## 2026-07-21 — Verifier findings fixed (fixer agent)

**All 3 major + 4 of 5 minor verifier findings fixed at root cause BEFORE
the research sweep (results\research does not exist yet — no re-sweep
needed). Full suite green post-fix; research_smoke chain re-verified end to
end (resume 45/45 skipped, results.csv untouched; aggregated.csv, both
model bundles, report, recommend all regenerated). Detail in
`notes/fixer.md`.**

1. **Seed-averaged labels actually WIRED (major).** aggregated.csv now
   carries seed-mean `feat_*` columns, per-technique `<tech>_n_seeds`
   coverage counts, and a coverage rule: only max-seed-coverage techniques
   may win a group (a 1-of-3-seed CDR mean can no longer beat 3-seed means
   — the 3 ghz_plus smoke aggregates flipped cdr->rem; winners now
   rem 7 / cdr 8). `train_and_eval(aggregated_df)` works (regression test
   added; it previously raised ValueError). Headline pipeline now trains on
   aggregated.csv (README + config comments + train_model nudge updated).
   CORRECTION of the integrator entry above: "§6.4 seed-averaged labels ->
   RESOLVED" was over-claimed — the data existed but the model could not
   consume it, so all previously quoted CV/LOFO/LOBO numbers are per-seed-
   label numbers (per-seed winners disagree with seed-averaged winners on
   13/45 = 28.9% of smoke rows).
2. **LODO added; LOBO relabeled (major).** New metrics key `lodo`
   (leave-one-DEVICE-out: all @x<scale> siblings of a base device held out
   together) is the honest "new noise environment" headline (3 folds on
   research.yaml). LOBO folds whose device stays in training at other
   scales measure noise-level INTERPOLATION and are labeled as such in
   model.py, train_model.py and report §6. Smoke sanity (n=15, noisy):
   CV 0.733 vs baseline 0.133, LOFO 0.733, LOBO 0.800, LODO 0.867.
3. **Noise-dial cap compression disclosed (major).** Lagos' 46.4% q2
   readout sits above the 0.45 cap: realized avg readout scaling is
   ~x1.28/x1.44 at nominal x1.5/x2.0 (q2 DECREASES 0.464->0.45 at x1.5;
   max_readout_error non-monotone). Disclosed in backends.py docstring,
   research.yaml comments ("scale and topology unconfounded" claim
   corrected), and report §5, which now prints a per-backend REALIZED
   error-rate + device-composition table (smoke confirms x1.27 realized),
   warns when per-scale device composition is unequal (fires on the
   asymmetric smoke config, silent on the symmetric research config), and
   labels the plot axis NOMINAL. Also §5 caveat: x1.0 (from_backend
   composite channels) vs scaled (pure depolarizing+readout) differ in
   noise CHARACTER, not just level.
4. **min_abs_ideal conditioning now user-facing (minor).** make_report.py
   reads the run_meta.json sidecar and report §1 disclosses the rejection
   sampling ("random circuits conditioned on |<Z^n>| >= 0.25", atypical
   high-|ideal| subset, family/size-dependent, near_clifford worst) —
   bumped seeds in the data trigger it even without the config.
5. **Record corrections (minor).** (a) Smoke CDR refusals are
   near_clifford 9/9 + ghz_plus 6/9 (the integrator entry above swapped
   them), and the ghz_plus refusals are degenerate-training-SPREAD
   refusals, NOT "fully-Clifford". (b) research.yaml CDR-signal comment
   corrected: 1206/1620 units (74%) pass the pre-guards, spanning ALL 5
   families (was "972, 3 families"). (c) report §3 notes raw_plus is a
   comparison column, structurally near-unwinnable in the cost-aware label
   (correct behavior). (d) Report accepts the REAL aggregated.csv schema
   (`<tech>_mean_abs_error` aliased; previously phantom 'raw_mean'
   techniques made every winner invalid).
6. **Left open deliberately:** CDR-refusal indicator feature (§6.2 — frozen
   FEATURE_NAMES interface; paper-ablation candidate). Tester finding 1
   (smoke has x1.5 not x2.0) was a task-wording nit; monotonicity verified
   at both scales, no code change.

**Updated research-run pipeline** (train on the seed-averaged file):
```
scripts\run_experiment.py --config configs\research.yaml --out results\research
scripts\train_model.py --data results\research\aggregated.csv --out results\research --label both
scripts\make_report.py --data results\research\results.csv --metrics results\research\metrics.json --out results\research
```
Paper headlines: LOFO (new family) + LODO (new device/noise environment);
LOBO = scale interpolation. results.csv remains available for a per-seed vs
seed-averaged label ablation.

## 2026-07-23 — V2 integration COMPLETE (integrator agent)

**Status: full suite 737/737 green; fresh boundary_smoke end-to-end chain
(experiment -> both+significant training -> stats -> report §8/§9 ->
recommend incl. abstain -> Angle-3 overlay) verified on live simulation;
V1 regressions byte-identical; configs/boundary.yaml sized at ~5.2 h
(<= 8 h with safety) from a measured 7-technique benchmark.**

### 1. Full suite

`pytest tests -q`: **737 passed, 0 failed** (1776.8 s under load; an earlier
run of the identical tree took 445 s idle — pure machine contention). The one
failure every builder reported (`test_shipped_hw_first_run_config_fits_two_minutes`)
was already root-cause-fixed before this pass: `configs/hw_first_run.yaml` had
shipped `hardware_confirmed: true` after the consented 2026-07-22 Heron run and
has been reverted to `false` (safe shipped default, revert comment in the file).
No test was weakened; no source file was changed by integration.

### 2. Fresh boundary_smoke e2e (results\boundary_smoke)

Integrator change: added `FakeManilaV2@x0.5` to `configs/boundary_smoke.yaml`
(task requires BOTH dial-down points in the e2e; 18 -> 24 units, header math
updated; no test pins this config). Full chain, all fresh:

- **run_experiment** — 24/24 units in 104 s, zero refusals. Schema verified:
  `base_shots` int column sits between `pauli` and `ideal`; all 7 techniques
  (`raw, raw_plus, zne, zne_fr, cdr, cdr_ridge, rem`) have value/abs_error/
  shots columns; 15 `feat_*` columns incl. `feat_log2_shots` (fv2).
- **Low-noise dial physics:** mean raw_abs_error x0.25 = 0.046 < x0.5 = 0.067
  < x1.0 = 0.103 (Manila siblings; 5/6 rows individually lower, the one
  exception is a 256-shot row where shot noise dominates — expected).
- **Determinism:** the 18 units shared with the interrupted 2026-07-22 run
  (results\boundary_smoke_prev, kept as cross-reference) are value-identical
  column-for-column.
- **train_model** on aggregated.csv, `--feature-version 2 --calibrate --stats`:
  `--label both` (model.joblib + model_cost_aware.joblib; LOSO
  leave-one-shot-budget-out auto-computed from the 2 budgets) AND `--label
  significant` (k_sigma 2.0 -> 9/24 rows become `tie`; bundle stores
  abstain_threshold 0.9). Smoke-scale metrics are noise (n=24, CV ~ baseline)
  — plumbing check only, not science.
- **stats.json** (scratch driver over qemsel.stats): win-share bootstrap CIs
  for both label columns, raw_plus-vs-raw + cdr-vs-rem permutation tests,
  Cliff's deltas, Koester checklist **passed=True**.
- **make_report** with `--stats-json --boundary-json`: report.md renders §8
  (CI columns, paired tests, effect sizes, checklist pass/flag table) and §9
  (overlay figure + agreement metrics + the three mandatory caveats).
- **recommend**: confident case (mirror/Lagos/4096 on model.joblib -> cdr,
  p=0.99, exit 0) and abstain case (layered/Manila@x0.25/256 on
  model_significant.joblib -> top p 0.406 < 0.9, "No confident
  recommendation", exit 2).
- **overlay_selector_vs_theory** (model.joblib, Manila x {0.25,0.5,1.0,1.5,
  2.0} x shots {256,1024,4096}): 15 grid points, **agreement 86.7 %**,
  iou_help 0.0, theory_help_share 0.133, boundary_overlay.png written.
  Reading: the smoke selector never predicts ZNE (zne/zne_fr won no
  aggregated smoke group, so those classes aren't in the bundle) while the
  theory says help on the 2 highest-eps x 4096 points — the disagreement
  band is exactly where the real boundary run has to put its data. Gotcha
  for scripts: call the overlay with an ABSOLUTE out_dir — plot_path is
  echoed as passed, and report validation resolves relative paths against
  --out (a CWD-relative path double-resolves and is rejected).

### 3. V1 regressions (byte-level)

- Fresh `configs/tiny.yaml` run: results.csv **SHA256-identical** to
  `results\tiny\results.csv` (CBDF857D…, 6248 bytes); winners rem 11 / cdr 8 /
  zne 1 exactly.
- Re-train on `results\research\aggregated.csv` `--label both` (default V1
  path): metrics.json **full-dict identical** to the stored
  `results\research\metrics.json` — CV 0.796/F1 0.417/baseline 0.594, LOFO
  0.787, LOBO 0.893, LODO 0.865; cost-aware 0.728/0.583/0.437, LOFO 0.702,
  LOBO 0.783, LODO 0.704. The END_RESULT headline reproduces exactly, not
  merely "within noise".

### 4. boundary.yaml sizing (measured)

Benchmark (faithful per-unit work: circuit + ideal + features v2 + executor +
apply_technique x7): layered_random over the full (n {2,3,4} x d {4,8,16}) x
{Manila, Lagos} grid @4096 = **7.44 s/unit mean**; budget factor 0.966
(256/1024/4096 mix — shots barely matter, transpile + CDR training sims
dominate); family factor 1.076 (hw_efficient 1.23x); dialed backends measured
~0.78x (kept as free margin). Total: 2430 x 7.44 x 0.966 x 1.076 ~= 18,800 s
**~= 5.2 h expected, 7.8 h with a 1.5x contention factor** -> the grid ships
UNTRIMMED (trim paths documented in the header). Both boundary configs
re-validated post-edit (2430 / 24 units, list_mode, fv2, 7 techniques).

### 5. Open items handed forward

1. **Angle-2 configs missing:** INTERFACES V2.6 planned `cdr_regressor.yaml` +
   smoke (techniques `[raw, cdr, cdr_ridge, cdr_rf]`, 5 families, fv2); B4's
   narrowed task shipped only the boundary pair. Author them before the
   Angle-2 run (`cdr_rf` is implemented and tested; no config exercises it).
2. **CDR_RIDGE_ALPHA=1.0** (B1 flag): shrinks hard at CDR's ~11-point/1-feature
   scale (cdr_ridge 0.776 vs cdr 0.166 on B1's capture circuit). Consider
   RidgeCV or a smaller alpha BEFORE the Angle-2 sweep, else the ridge arm is
   handicapped by construction.
3. **Naming:** task + shipped file say `boundary.yaml`; INTERFACES V2.6 said
   `boundary_sweep.yaml`. The file on disk wins; update INTERFACES on its next
   edit.
4. `results\boundary_smoke_prev\` is disposable once the determinism
   cross-check above is no longer wanted.
5. The smoke overlay is a plumbing artifact: for the paper, re-run
   `overlay_selector_vs_theory` against the bundle trained on the FULL
   boundary sweep, and quote agreement/IoU from that.
