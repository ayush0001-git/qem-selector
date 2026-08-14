# qemsel — INTERFACES (architect-owned; builders implement these VERBATIM)

Stub files under `src/qemsel/` contain the exact signatures + full docstring
contracts. **Implement the signatures verbatim — do not rename, reorder, add, or
remove parameters.** Replace only the `raise NotImplementedError` bodies (and add
private `_helpers` as needed). This document is the cross-team summary; the stub
docstrings are the authoritative fine print.

## Ownership map (edit ONLY your files)

| Builder | Owns (relative to project root) |
|---|---|
| builder-circuits | `src/qemsel/circuits.py`, `tests/test_circuits.py` |
| builder-backends | `src/qemsel/backends.py`, `src/qemsel/ideal.py`, `tests/test_backends.py`, `tests/test_ideal.py` |
| builder-mitigation | `src/qemsel/mitigation.py`, `tests/test_mitigation.py` |
| builder-features | `src/qemsel/features.py`, `tests/test_features.py` |
| builder-experiment | `src/qemsel/experiment.py`, `configs/` (all files), `scripts/run_experiment.py`, `tests/test_experiment.py` |
| builder-model | `src/qemsel/model.py`, `scripts/train_model.py`, `tests/test_model.py` |
| builder-recommend | `src/qemsel/recommend.py`, `src/qemsel/report.py`, `scripts/recommend.py`, `scripts/make_report.py`, `tests/test_recommend.py`, `tests/test_report.py` |
| builder-docs | `README.md`, `docs/` (all files) |
| architect (done) | `pyproject.toml`, `src/qemsel/__init__.py`, all stub signatures, `tests/conftest.py`, this file |

Shared read-only for everyone: `PROJECT_STATE.md`, `requirements.txt`,
`scripts/verify_env.py`. Each agent writes its own `notes/<label>.md`.

## Conventions (binding on all builders)

1. **Pauli-string convention:** `pauli[i]` acts on **qubit i** (leftmost char =
   q0). This is the REVERSE of qiskit's `quantum_info.Pauli` label; convert with
   `Pauli(pauli[::-1])`. Qiskit **counts bitstrings are little-endian**
   (rightmost bit = q0) — mind parity computations.
2. **Circuits carry NO final measurements** and no classical registers.
   Executors add measurements to a **copy**; never mutate a caller's circuit.
3. **Seeds everywhere.** Any randomness uses an explicit seed parameter via
   `numpy.random.default_rng(seed)` (never global numpy state); simulators get
   `seed_simulator=seed`, transpiler `seed_transpiler=seed`.
4. **pathlib everywhere.** All path parameters are `pathlib.Path`; create
   output dirs with `mkdir(parents=True, exist_ok=True)`.
5. **Transpile:** pass ONLY the AerSimulator built with the noise model (no
   `basis_gates=` — qiskit 2.5 UserWarning); use `optimization_level=0` inside
   executors so mitiq's folded gates survive.
6. **Cross-module imports go ONLY through the public functions listed here**
   (e.g. features calls `backends.get_backend_info`; mitigation reuses
   `ideal.ideal_expectation` for CDR). Never import another module's private
   helpers. `qemsel/__init__.py` stays version-only — no submodule imports.
7. **Environment:** run everything with
   `"E:\quatum  computiiing\qem-selector\.venv\Scripts\python.exe"` (parent dir
   has DOUBLE spaces — always quote). Package is installed editable; new
   modules are picked up automatically. Don't `pip install` anything new.
8. **Windows/PowerShell gotcha:** inline `python -c "..."` strips inner double
   quotes — write a temp `.py` file and run it instead.
9. **Tests:** put them in your own `tests/test_<module>.py`. Use the conftest
   fixtures (`tiny_circuit`, `tiny_identity_circuit`, `fake_executor`,
   `out_dir`, `tiny_results_df`) so unit tests stay fast and noiseless. Keep
   any noisy-simulation test tiny (<= 3 qubits, <= 256 shots) or mark it
   `@pytest.mark.slow`.

## Interface summary (authoritative details in stub docstrings)

### circuits.py (builder-circuits)
- `class CircuitSpec` — dataclass: `family: str, n_qubits: int, depth: int, seed: int, params: dict = {}`; property `circuit_id -> str` = `"{family}_q{n}_d{d}_s{s}"`.
- `layered_random(n_qubits: int, depth: int, seed: int) -> QuantumCircuit` — brick-work random 1q rotations + CX layers.
- `near_clifford(n_qubits: int, depth: int, seed: int, non_clifford_fraction: float = 0.15) -> QuantumCircuit` — mostly-Clifford random circuit.
- `ghz_plus(n_qubits: int, depth: int, seed: int) -> QuantumCircuit` — GHZ prep + identity-equivalent padding to target depth.
- `hw_efficient_ansatz(n_qubits: int, depth: int, seed: int) -> QuantumCircuit` — ry/rz + linear CX blocks, angles bound numerically.
- `mirror_circuit(n_qubits: int, depth: int, seed: int) -> QuantumCircuit` — U then U†; ideal `<'Z'*n>` EXACTLY +1.0.
- `FAMILIES: dict[str, Callable]` — name -> generator registry (5 entries above).
- `generate_suite(config: dict) -> list[tuple[QuantumCircuit, CircuitSpec]]` — cartesian product of `families x n_qubits x depths x seeds` (+ optional per-family `params`), deterministic order.

### backends.py + ideal.py (builder-backends)
- `BACKENDS: list[str]` = `["FakeManilaV2", "FakeJakartaV2", "FakeLagosV2", "FakeSherbrooke"]`.
- `get_backend_info(name: str) -> dict` — keys EXACTLY: `name, n_qubits, avg_1q_error, avg_2q_error, avg_readout_error, max_readout_error`; cached per name.
- `make_executor(backend_name: str, shots: int, seed: int) -> Callable[[QuantumCircuit, str], float]` — returns `executor(circuit_without_measurements, pauli) -> expectation`; noise model + simulator built once at make time; basis rotation for X/Y; `optimization_level=0`; little-endian parity per convention 1.
- `class RealHardwareBackend` — `__init__(self, name: str = "ibm_brisbane")` raises `NotImplementedError('add IBM token in configs/hardware.yaml to enable real hardware')`.
- `ideal_expectation(circuit: QuantumCircuit, pauli: str) -> float` — exact statevector expectation, same pauli convention.

### mitigation.py (builder-mitigation)
- `TECHNIQUES: list[str]` = `['raw', 'zne', 'cdr', 'rem']` (these are the ML class labels — never rename).
- `SHOT_MULTIPLIER: dict[str, int]` = `{'raw': 1, 'zne': 3, 'cdr': 11, 'rem': 3}` — tune values if implementation differs, but keep the name/keys and keep it truthful to what `apply_technique` executes.
- `apply_technique(name: str, circuit, pauli: str, executor, backend_name: str, shots: int, seed: int) -> float` — uniform dispatch: raw = direct executor call; zne = mitiq Richardson, scale factors (1,2,3); cdr = mitiq with 10 near-Clifford training circuits + `ideal.ideal_expectation` as the noiseless simulator; rem = inverse confusion matrix from backend readout errors. Exceptions may propagate (experiment catches). Do not clip outputs.
- `shots_consumed(name: str, base_shots: int) -> int` — `base_shots * SHOT_MULTIPLIER[name]`.

### features.py (builder-features)
- `FEATURE_NAMES: list[str]` — EXACT order: `n_qubits, depth, n_1q_gates, n_2q_gates, n_cnot, n_non_clifford, clifford_fraction, depth_per_qubit, backend_avg_2q_error, backend_avg_readout_error`.
- `extract_features(circuit: QuantumCircuit, backend_name: str) -> dict[str, float]` — keys exactly FEATURE_NAMES, all floats; computed on the as-generated (pre-transpile) circuit; rx/ry/rz Clifford iff angle ≡ 0 mod π/2 (tol 1e-9); backend numbers via `get_backend_info`.

### experiment.py (builder-experiment)
- `run_experiment(config: dict, out_dir: Path) -> pd.DataFrame` — config keys: `circuits` (generate_suite config), `backends`, `shots`, `pauli` (`'auto'` => `'Z'*n_qubits`), optional `techniques`. Crash-safe: append to `results.csv` after every (circuit, backend) unit; resume by skipping existing `(circuit_id, backend)` pairs; `run_meta.json` sidecar (config + versions + timestamp); per-technique try/except -> NaN + `errors.log` line. Row columns EXACTLY: `circuit_id, family, n_qubits, depth, seed, backend, pauli, ideal, feat_<name>..., <tech>_value, <tech>_abs_error, <tech>_shots..., best_technique` (best = argmin non-NaN abs_error; `''` if all failed).
- Also owns `scripts/run_experiment.py` (CLI: `--config configs/experiment.yaml --out results/run1`) and `configs/experiment.yaml` (a small default config: 2-3 qubit circuits, FakeManilaV2 + FakeLagosV2, 2048 shots) plus `configs/hardware.yaml` placeholder (`ibm_token: null`).

### model.py (builder-model)
- `train_and_eval(df: pd.DataFrame, out_dir: Path) -> dict` — RandomForest + GradientBoosting (random_state=0); StratifiedKFold with `n_splits = min(5, smallest class count)` (smallest class < 2 => fit-all fallback with `cv_folds: 0`); out-of-fold metrics; majority-class baseline; saves `model.joblib` BUNDLE `{'model', 'feature_names', 'classes', 'model_name', 'qemsel_version'}` + `metrics.json`. Returns keys EXACTLY: `best_model_name, accuracy, macro_f1, baseline_accuracy, labels, confusion_matrix, feature_importances, per_model, n_samples, cv_folds`.
- Also owns `scripts/train_model.py` (CLI: `--results results/run1/results.csv --out results/run1`).

### recommend.py + report.py (builder-recommend)
- `recommend(model_path: Path, circuit: QuantumCircuit, backend_name: str) -> dict` — loads the joblib bundle; features via `extract_features`, ordered by the bundle's `feature_names`; returns keys EXACTLY `technique, probabilities, features`.
- `generate_report(df: pd.DataFrame, model_metrics: dict, out_dir: Path) -> Path` — writes `report.md` + PNGs (`error_by_technique.png, win_rate.png, confusion_matrix.png, feature_importances.png`); matplotlib 'Agg'; returns path to report.md. Sections: overview, technique comparison, cost-normalized view (uses `<tech>_shots`), win rates, model evaluation, reproducibility.
- Also owns `scripts/recommend.py` and `scripts/make_report.py` CLIs.

### tests/conftest.py (architect — do not edit)
Fixtures: `tiny_circuit` (2q Bell, no measurements), `tiny_identity_circuit`
(3q identity-equivalent), `fake_executor` (noiseless statevector executor
matching the make_executor contract), `out_dir` (tmp Path),
`tiny_results_df` (16-row synthetic DataFrame in the exact experiment schema,
≥4 rows per class).

## Data-flow overview

```
circuits.generate_suite ─┐
                         ├─> experiment.run_experiment ──> results.csv (+ run_meta.json)
backends.make_executor ──┤        │  uses: ideal.ideal_expectation,
mitigation.apply_technique        │        features.extract_features,
                                  │        mitigation.shots_consumed
                                  v
                     model.train_and_eval ──> model.joblib + metrics.json
                                  │
              recommend.recommend │  report.generate_report ──> report.md + PNGs
```

---
---

# ============================ V2 SECTION ============================
# (added 2026-07-22 by architect-v2 — everything above is the FROZEN V1
# record; do NOT edit it. Builders B1-B8 implement the V2 stubs below.)

## V2.0 Mission (why these upgrades exist — full rationale in docs/RESEARCH_ANGLES.md, END_RESULT.md §3/§5)

Make the codebase capable of the paper's two experiments plus the
reviewer-bar upgrades:

* **Angle 3 (headline):** selector's learned ZNE-refusal boundary vs
  Scavino's analytic help-harm boundary (arXiv:2605.08251) — needs a SHOTS
  axis (`experiment`), a fixed-Richardson ZNE variant (`zne_fr` in
  `mitigation`), a boundary module (`boundary.py`), shots-aware features
  (`features` v2) and the overlay report section.
* **Angle 2 (supporting, sim-only):** CDR regressor choice as selectable
  techniques (`cdr_ridge`, `cdr_rf`) + overfitting map.
* **Reviewer bar:** statistical hygiene (`stats.py`, Koester checklist),
  significance-aware 'tie' labels, calibrated/abstaining model, low-noise
  dial coverage (`@x0.25`, `@x0.5` — the Heron-like regime, END_RESULT F7).

## V2.1 Ownership map (edit ONLY your files; each Bx also owns its test-file ADDITIONS)

| Builder | Owns (V2 changes only — do not rewrite V1 behavior) |
|---|---|
| **B1** builder-mitigation | `src/qemsel/mitigation.py`, `tests/test_mitigation.py` additions |
| **B2** builder-backends | `src/qemsel/backends.py`, `tests/test_backends.py` + `tests/test_noise_scaling.py` additions |
| **B3** builder-boundary | `src/qemsel/boundary.py` (NEW), `tests/test_boundary.py` (NEW) |
| **B4** builder-experiment | `src/qemsel/experiment.py`, `configs/` NEW files only (`boundary.yaml`, `boundary_smoke.yaml`, `cdr_regressor.yaml`, `cdr_regressor_smoke.yaml`), `scripts/run_experiment.py`, `tests/test_experiment.py` additions |
| **B5** builder-features | `src/qemsel/features.py`, `tests/test_features.py` additions |
| **B6** builder-stats | `src/qemsel/stats.py` (NEW), `scripts/compute_stats.py` (NEW), `tests/test_stats.py` (NEW) |
| **B7** builder-model | `src/qemsel/model.py`, `scripts/train_model.py`, `tests/test_model.py` additions |
| **B8** builder-recommend | `src/qemsel/recommend.py`, `src/qemsel/report.py`, `scripts/recommend.py`, `scripts/make_report.py`, `tests/test_recommend*.py` + `tests/test_report*.py` additions |
| architect-v2 (done) | all V2 stub signatures, this section, `notes/architect-v2.md` |

Existing configs (`tiny/small/full/research/research_smoke/experiment/
hardware/hw_first_run`) are byte-frozen — NOBODY touches them.

## V2.2 Binding conventions (in addition to V1 conventions 1-9)

10. **FROZEN V1 surface** (regression tests pin these — never extend them):
    `mitigation.TECHNIQUES` (5 entries), `mitigation.SHOT_MULTIPLIER`
    (5 keys), `features.FEATURE_NAMES` (10 entries), the V1 results.csv /
    aggregated.csv column lists for scalar-shots configs, the V1
    `recommend()` 3-key return for V1 bundles, default-path `metrics.json`
    keys. All new capability lives behind NEW names (`*_V2`) or NEW
    keyword-only parameters whose defaults reproduce V1 byte-identically.
11. **Stub discipline:** architect-v2 stubs raise `NotImplementedError`
    with a "V2 stub — builder-X implements" message. Implement the
    signature VERBATIM, replace only the raise (and the marked guard
    blocks). The stub docstrings are the authoritative fine print.
12. **Import DAG (V2):** `stats` imports nothing from qemsel; `boundary`
    imports `mitigation`/`backends`/`ideal`/`features`/`circuits` (never
    `model`/`report`/`experiment`); `model` may import `stats` and
    `features`; `report` consumes stats/boundary OUTPUT dicts only (never
    imports those modules); `mitigation` NEVER imports `boundary`.
    `richardson_coefficients` lives in `mitigation` (single source of
    truth for nodes/coefficients shared with the theory side).
13. **Sim-only:** nothing in V2 may touch `ibm_*` backends. `boundary`
    raises ValueError on any `ibm_*` name. No builder edits
    `hardware.py`, `hardware.yaml`, or the budget gates.
14. **Every builder's definition of done:** full suite green (the 424
    legacy tests unweakened + your additions), plus the specific
    regression duty named in your module's V2 entry below.
15. **Cross-builder stubs:** while another builder's stub is unimplemented,
    your unit tests must monkeypatch it (V1 pattern) — never block on
    landing order. Integration tests that need >= 2 builders' code go in
    the LAST-landing builder's test file, marked `@pytest.mark.slow`.

## V2.3 mitigation.py (B1)

- `TECHNIQUES_V2: list[str]` = `TECHNIQUES + ['zne_fr', 'cdr_ridge', 'cdr_rf']` (defined; class labels — never rename; NOT the experiment default).
- `SHOT_MULTIPLIER_V2: dict[str, int]` = superset of frozen `SHOT_MULTIPLIER`; `zne_fr` 1 (equal-split budget), `cdr_ridge`/`cdr_rf` 11 (defined; derived — must stay truthful to executed calls).
- Settings constants (defined; SPIKE MAY ADJUST values, never names): `ZNE_FR_SCALE_FACTORS=(1.0,2.0,3.0)`, `ZNE_FR_SHOT_ALLOCATION='equal_split'`, `ZNE_FR_FOLD_METHOD='global'`, `CDR_RIDGE_ALPHA=1.0`, `CDR_RF_N_ESTIMATORS=100`, `CDR_RF_MAX_DEPTH=None`, `CDR_SKLEARN_NUM_TRAINING_CIRCUITS=CDR_NUM_TRAINING_CIRCUITS`.
- `richardson_coefficients(scale_factors: tuple[float, ...]) -> tuple[float, ...]` — STUB: fixed Lagrange-at-zero coefficients; shared with boundary.py.
- `apply_technique` / `shots_consumed` — signatures UNCHANGED; now validate against `TECHNIQUES_V2` and dispatch `zne_fr` -> `_apply_zne_fr(circuit, pauli, executor, backend_name, base_shots, seed)` (STUB: fixed coefficients x equal-split rebuilt executors x deterministic global folding), `cdr_ridge`/`cdr_rf` -> `_apply_cdr_sklearn(circuit, pauli, executor, seed, regressor)` (STUB: same compile + SAME three fail-loud guards as `_apply_cdr`, same training-set settings, sklearn Ridge/RandomForestRegressor noisy->ideal fit — the generate_training_circuits bypass; spike may swap Ridge to the fit_function route only if proven equivalent).
- Regression duty: V1 names' dispatch path byte-identical; executor-call-count tests for all three new techniques mirroring the existing cost-model tests.

## V2.4 backends.py (B2)

- NO new signatures. `LOW_NOISE_SCALES: tuple[float, ...] = (0.25, 0.5)` (defined). Grammar already accepts any finite scale > 0.
- Duty: VERIFY the scaled path at 0.25/0.5 end to end and pin with tests: (a) raw |error| monotone over {0.25, 0.5, 1.0} on >= 1 device; (b) `get_backend_info` scales linearly below 1.0 (caps never bind); (c) plain-name path byte-identical; (d) executor determinism at low scales; docstring updated (done in part — extend as needed).

## V2.5 boundary.py (B3, NEW module)

- `BoundaryParams` dataclass (frozen): `d_p, k_q, p, q, scale_factors, shot_allocation, source` (defined).
- Constants (defined): `DEFAULT_EPS_FEATURE='avg_2q_error'`, `DEFAULT_ZNE_LABELS=('zne','zne_fr')`, `OVERLAY_PNG='boundary_overlay.png'`.
- `variance_k_q(scale_factors=None, shot_allocation=None) -> float` — STUB: a-priori K_q from `mitigation.richardson_coefficients` (Mohammadipour-Li; spike fixes expression).
- `estimate_params(circuit, pauli, backend_name, *, seed=0) -> BoundaryParams` — STUB: sim-side D_p estimate using mu_0 (`ideal`); ValueError on `ibm_*`.
- `delta_mse(eps: float, shots: float, params: BoundaryParams) -> float` — STUB: FROZEN formula `d_p*eps**(2p) - k_q*eps**q/shots`; POSITIVE = zne_fr helps (sign convention frozen).
- `regime(eps, shots, params, *, tol=0.0) -> str` — STUB: 'help' | 'harm'.
- `boundary_eps(shots, params) -> float | None`, `boundary_shots(eps, params) -> float | None` — STUBS: zero crossings; None when no crossing (all three regime shapes handled).
- `overlay_selector_vs_theory(model_bundle: Path|str|dict, grid_spec: dict, out_dir: Path) -> dict` — STUB: THE Angle 3 figure; grid_spec keys `backends`/`shots_list`/`circuits` (+ optional `pauli`/`params`/`eps_feature`/`zne_labels`); eps axis = REALIZED `get_backend_info` numbers; returns JSON-serializable dict with keys exactly `agreement_pct, iou_help, n_points, selector_help_share, theory_help_share, eps_feature, zne_labels, plot_path, grid`; writes `OVERLAY_PNG` into out_dir (matplotlib 'Agg').
- Duty: unit tests with synthetic params (all three regime shapes, sign convention, None cases) + a monkeypatched-bundle overlay smoke test.

## V2.6 experiment.py + configs (B4)

- `BASE_SHOTS_COLUMN='base_shots'`, `_AGG_KEY_COLUMNS_V2=[family, n_qubits, depth, backend, base_shots]` (defined).
- `_normalize_shots(shots_cfg) -> tuple[list[int], bool]` — STUB carrying the FULL shots-axis contract (read it): scalar => `([s], False)` and EVERYTHING byte-identical to V1; list => cross-product units (circuit x backend x base_shots, budgets innermost, config order), `base_shots` int column between `pauli` and `ideal`, resume key `(circuit_id, backend, base_shots)`, per-unit executor at the unit's budget, `<tech>_shots` from the unit's budget, low-signal screen once per (circuit, backend), errors.log lines gain `s{base_shots}` field in list mode only, aggregated.csv grouped by `_AGG_KEY_COLUMNS_V2`.
- Config keys: `shots: int | list[int]`; `feature_version: 1|2` (default 1) => feat_* columns from `features.FEATURE_NAMES_BY_VERSION[v]`, extract_features called with `version=v, base_shots=<unit budget>`; `techniques` validated against `mitigation.TECHNIQUES_V2` (default unchanged: `mitigation.TECHNIQUES`).
- NEW configs (+ nothing else in configs/): `boundary.yaml` (Angle 3: 3 devices x scales {0.25,0.5,1.0,1.5,2.0} x shots [256,1024,4096,16384], `feature_version: 2`, techniques incl. `zne_fr`; B4 sizes the circuit grid for a <= ~12 h sweep), `boundary_smoke.yaml` (same shape, minutes), `cdr_regressor.yaml` (Angle 2: techniques `[raw, cdr, cdr_ridge, cdr_rf]`, all 5 families, `feature_version: 2`), `cdr_regressor_smoke.yaml`.
- Regression duty: fresh `tiny.yaml` run byte-identical to the stored reference; a kill-resume test in list mode; a schema-mismatch (old CSV + list config) ValueError test.

## V2.7 features.py (B5)

- `FEATURE_NAMES_V2` = `FEATURE_NAMES + [log2_shots, n_2q_layers, entangling_density, mean_rz_angle_dist, backend_avg_1q_error]`; `FEATURE_NAMES_BY_VERSION={1:.., 2:..}` (defined; definitions in the constant's docstring — authoritative).
- `extract_features(circuit, backend_name, *, version=1, base_shots=None) -> dict` — signature updated; version=1 path byte-identical (base_shots ignored); version=2 STUB: exact V1 values plus the five new features; base_shots required (> 0) for log2_shots.
- Duty: hand-computed value tests for each new feature (incl. unbound-parameter -> 1.0 rule for mean_rz_angle_dist, empty-circuit zeros), order test `list(keys) == FEATURE_NAMES_V2`, and a version=1 regression test against stored V1 outputs.

## V2.8 stats.py (B6, NEW module) + scripts/compute_stats.py

- Constants (defined): `DEFAULT_N_BOOT=10000`, `DEFAULT_N_PERM=10000`, `DEFAULT_K_SIGMA=2.0`.
- `sigma_shot(value, shots) -> float` — STUB: `sqrt((1 - min(v^2,1))/shots)`.
- `win_shares(labels, techniques=None) -> dict[str, float]` — STUB: empty/NaN labels excluded from denominator.
- `bootstrap_ci(values, statistic=None, *, n_boot, ci=0.95, seed=0) -> dict` — STUB: percentile bootstrap; keys `estimate, lo, hi, ci, n, n_dropped_nan, n_boot, seed`.
- `win_share_ci(labels, technique, *, n_boot, ci, seed) -> dict` — STUB: bootstrap_ci shape + `technique`.
- `paired_permutation_test(err_a, err_b, *, n_perm, seed, alternative='two-sided') -> dict` — STUB: sign-flip test on mean paired diff; NaN pairs dropped pairwise; add-one p; keys `mean_diff, p_value, n_pairs, n_dropped_nan, n_perm, alternative, seed`.
- `cliffs_delta(a, b) -> float` — STUB: in [-1,1]; negative = a smaller.
- `summarize_folds(fold_scores) -> dict` — STUB: `mean, std(ddof=1), min, max, n_folds`.
- `koester_checklist(df, *, k_sigma=2.0) -> dict` — STUB: both schemas; checks `overshoot_beyond_physical_max, error_beyond_physical_max, nan_rate, label_argmin_consistent, winner_margin_below_k_sigma, partial_coverage_winners`; top-level `schema, n_rows, techniques, checks, passed`.
- `scripts/compute_stats.py` — NEW CLI `--data <csv> --out <dir>` writing `stats.json` (schema in the stats module docstring) = exactly report's `stats_results` input.
- Everything deterministic (`default_rng(seed)`), JSON-serializable, no qemsel imports. Duty: known-answer tests (hand-checkable tiny inputs) incl. NaN policies; run checklist against `results/research/results.csv` once and record the outcome in notes.

## V2.9 model.py (B7) + scripts/train_model.py

- `SIGNIFICANT_LABEL='best_technique_significant'`, `TIE_CLASS='tie'` (defined).
- `derive_significant_label(df, k_sigma=2.0, *, techniques=None) -> pd.Series` — STUB: winner unless runner-up within `k_sigma * sqrt(sigma_w^2 + sigma_r^2)` (sigmas via `stats.sigma_shot`; per-schema routes in the docstring) else `TIE_CLASS`; '' when all failed.
- `train_and_eval(..., *, feature_version=1, calibrate=False, abstain_threshold=None, extended_stats=False)` — new keyword-only params (defaults byte-identical; guard-stubbed): feature_version selects the feat_* matrix via `FEATURE_NAMES_BY_VERSION`; calibrate wraps the refit model in `CalibratedClassifierCV` (sigmoid, GROUPED folds) -> bundle `'calibrated': True` + metrics `'calibration'`; abstain_threshold stored verbatim in the bundle for recommend; extended_stats adds `fold_accuracies` + `fold_summary` (via `stats.summarize_folds`). New automatic metrics key `'loso'` (leave-one-shot-budget-out, lobo-shaped) whenever df has `base_shots` with >= 2 values.
- `train_and_eval_all(..., *, <same four kwargs>)` — forwards verbatim (done).
- V2 bundle keys (additive): `feature_version` (int), `calibrated` (bool), `abstain_threshold` (float|None). V1 bundles unchanged.
- `scripts/train_model.py`: `--label` gains `significant` (derive column via `derive_significant_label`, train to `model_significant.joblib` + `metrics_significant.json`); new flags `--k-sigma 2.0`, `--calibrate`, `--abstain-threshold`, `--feature-version`, `--stats` (-> extended_stats). Defaults reproduce V1 CLI byte-identically.
- Duty: 'tie' class flows through CV/LOFO/LODO like any class; calibration test (probabilities change, argmax accuracy within noise); abstain-rate metric test; loso test on a synthetic two-budget df.

## V2.10 recommend.py + report.py (B8) + scripts

- `recommend(model_path, circuit, backend_name, *, base_shots=None) -> dict` — signature updated (guard-stubbed): V1 bundle => EXACT V1 3-key return (base_shots ignored); V2 bundle (has `feature_version`) => extract_features with the bundle's version + base_shots (required for v2 features), abstain when max proba < bundle threshold => `'technique': 'abstain'`; return adds `abstained, abstain_threshold, feature_version`. `scripts/recommend.py` gains `--shots`.
- `generate_report(..., *, stats_results=None, boundary_overlay=None) -> Path` — signature updated (guard-stubbed): section 8 "Statistical hygiene" from the compute_stats dict (win-share CIs, permutation tests, Cliff's delta, checklist pass/flag table); section 9 "ZNE help-harm boundary overlay" from the boundary dict (figure by relative `plot_path` — must already be inside out_dir, ValueError otherwise; agreement/IoU/shares + mandatory caveats: realized-eps axis, zne_fr alignment, sim-only). Omitted args => report byte-identical to V1. `_CANONICAL_TECHNIQUES` extended (done). `scripts/make_report.py` gains `--stats-json`, `--boundary-json`.
- Duty: golden-file test that V1 calls produce byte-identical report.md; section 8/9 rendering tests from fixture dicts; V1-bundle recommend regression test.

## V2.11 Data-flow (V2 additions only)

```
configs/boundary.yaml ──> experiment (shots LIST, feature_version 2)
        ──> results.csv(+base_shots) / aggregated.csv(V2 keys)
        ──> model --label both/significant [--calibrate --abstain-threshold]
                ──> model*.joblib (V2 bundles) + metrics*.json (loso, ...)
        ──> stats: scripts/compute_stats.py ──> stats.json ─┐
        ──> boundary.overlay_selector_vs_theory ──> overlay dict + boundary_overlay.png ─┤
                                                                                         v
                                report.generate_report(..., stats_results, boundary_overlay)
                                        ──> report.md sections 8 + 9
```
