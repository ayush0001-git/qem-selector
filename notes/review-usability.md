# Usability review (review-usability agent) — 2026-07-21

Persona test: a final-year student uses and extends the project alone, following
README verbatim in a fresh PowerShell. Everything below was actually executed
with `"E:\quatum  computiiing\qem-selector\.venv\Scripts\python.exe"`.

## What I ran (all verbatim README commands unless noted)

| Command | Result |
| --- | --- |
| `scripts\verify_env.py` | exit 0, ALL CHECKS PASSED |
| `python -m pytest -q` | 251 passed, 3 benign mitiq warnings, 3:19 |
| `run_experiment.py --config configs\experiment.yaml --out <dir>` (quickstart step 1) | works; 80 units, 643 s (~11 min) total on this machine, no errors.log; excellent `[i/N] ... (unit Xs, elapsed Ys)` progress lines |
| resume (re-run same command) | works — prints "80 already in results.csv", skips all 80, exits cleanly |
| `train_model.py --results ...` (quickstart step 2, alias flag) | works; honest summary (model acc 0.713 < baseline 0.800, printed side by side; 3-fold CV auto-reduction announced) |
| `make_report.py --data ... --metrics ... --out ...` | works, report.md + 4 PNGs |
| `recommend.py --model ... --backend ... --demo ghz_plus` | works, clean JSON output |
| `recommend.py --qasm <file>` | works incl. stripping final measurements |
| README Python snippet (`from qemsel.circuits import ghz_plus; recommend(...)`) | works as printed |
| Error paths: unknown backend in config, missing CSV, missing metrics, unknown demo family | clear, actionable messages (unknown backend lists the valid names) |

Overall: this is an unusually usable research codebase. Configs are commented
with unit counts and expected runtimes, progress output is excellent, the
report.md is genuinely readable, resume works exactly as documented, and
LEARNING_GUIDE.md is accurate on every physics/API claim I checked (Lagos
readout ~27%/max 46% verified against `get_backend_info`; CDR/REM/ZNE cost
multipliers match `SHOT_MULTIPLIER`; endianness description matches the code).

## Findings

### 1. MEDIUM — Real-hardware switch is NOT config-only, and the stub/docs contradict each other
Files: `src/qemsel/backends.py` (RealHardwareBackend), `configs/hardware.yaml`, `README.md` ("Switching to real IBM hardware")

- The stub raises `NotImplementedError('add IBM token in configs/hardware.yaml
  to enable real hardware')` and `hardware.yaml` says "Set ibm_token ... to
  enable real-hardware runs" — but **nothing in the codebase reads
  hardware.yaml or ibm_token at all** (grep confirms zero references outside
  the stub's docstring). A student who follows the error message creates an
  IBM account, pastes a token, and still gets `NotImplementedError`.
- README step 4 claims that after implementing `RealHardwareBackend` "the rest
  of the pipeline needs zero changes". False: `experiment._validate_config`
  rejects any backend not in the 4-name fake `BACKENDS` list;
  `features.extract_features` -> `backends.get_backend_info` raises ValueError
  for e.g. "ibm_brisbane"; `run_experiment` and `recommend` both hardcode the
  fake-backend paths (`make_executor`, `get_backend_info`). `RealHardwareBackend`
  is referenced by no other module — there is no dispatch seam.
- Fix: (a) change the stub message and hardware.yaml comment to say the class
  must be *implemented* (token alone does nothing); (b) in README step 4, list
  the actual touch points (BACKENDS / get_backend_info / make_executor
  dispatch in backends.py, plus config validation in experiment.py), or add a
  real dispatch seam (e.g. `make_executor` consults hardware.yaml when the
  name is not a fake backend).

### 2. MEDIUM — No documented plug-in point for the custom CDR regressor (the student's planned original contribution)
Files: `src/qemsel/mitigation.py` (`_apply_cdr`), `README.md` (roadmap item 5), `docs/LEARNING_GUIDE.md` (section 4)

- Roadmap item 5 and the LEARNING_GUIDE both name "swap the CDR linear fit for
  ridge/polynomial/random-forest regressors" as the thesis's novel angle, but
  no document says *where* that swap happens. `_apply_cdr` hardcodes mitiq
  defaults and exposes no hook.
- Verified: mitiq 1.0.0 `execute_with_cdr` has `fit_function=` /
  `num_fit_parameters=` kwargs (default `linear_fit_function`) — the natural
  seam — but these are mentioned nowhere in mitigation.py, README,
  LEARNING_GUIDE, or notes/spike-cdr.md (the spike's signature listing omits
  them).
- Important nuance the docs should state: `fit_function` is a scipy
  `curve_fit`-style *parametric function*, so ridge (regularized) and
  random-forest regressors **cannot** be passed through it. Those need the
  lower-level route: `mitiq.cdr.generate_training_circuits` + run executor /
  ideal simulator yourself + fit any sklearn regressor + apply to the target's
  noisy value — i.e. replacing `execute_with_cdr` inside `_apply_cdr`.
- Fix: add a short "plugging in your own CDR regressor" note (README roadmap 5
  or mitigation.py module docstring): polynomial fits go via
  `fit_function=`/`num_fit_parameters=` pass-through; sklearn regressors via
  `generate_training_circuits`. Optionally expose a module-level
  `CDR_FIT_FUNCTION` constant next to the other CDR_* settings so the ablation
  is a one-line change.

### 3. MEDIUM — README quickstart mislabels its size and tells the student to overwrite a shipped config; graded configs are never mentioned
Files: `README.md` (Quickstart + Roadmap), `configs/tiny.yaml`, `configs/small.yaml`

- "Quickstart (tiny run)" actually runs `configs\experiment.yaml` = **80
  units** (643 s ≈ 11 min measured here; longer on a slower laptop) with no
  duration hint. The genuinely tiny shipped config (`configs/tiny.yaml`, 20 units,
  ~3.5 min, and run_experiment's *default* `--config`) is never mentioned.
- Instead the README says "Save this as `configs\tiny.yaml`" and prints YAML
  that **differs from the shipped tiny.yaml** (2 families/1 backend/1024 shots
  vs 5 families/2 backends/2000 shots). Following the instruction verbatim
  overwrites a shipped, integration-tested config.
- Roadmap step 2 describes the "small run" as "{2,3,4,5} qubits ... 3
  backends", but shipped `configs/small.yaml` is {2,3} qubits x 2 backends
  (120 units). Nothing in README points at small.yaml/full.yaml although they
  exist, are well-commented, and are the intended graded runs
  (PROJECT_STATE next step is literally "Run configs/small.yaml").
- Fix: quickstart should say "run the shipped `configs\tiny.yaml` (20 units,
  a few minutes)"; delete the inline YAML (or present it as "how to write your
  own config"); add one line listing tiny/small/full with unit counts and
  rough durations; align roadmap step 2 text with small.yaml or vice versa.

### 4. LOW — `recommend.py` prints a raw traceback for a missing --model (CLI inconsistency)
File: `scripts/recommend.py`

Confirmed tester A's M1 independently: `--qasm` missing gives a clean
`error: QASM file not found: ...` but `--model` missing dumps a full traceback
(message at the bottom is good: "run scripts/train_model.py first"). Fix: wrap
the `recommend(...)` call in `main()` in try/except
FileNotFoundError/ValueError -> `raise SystemExit(f"error: {exc}")`.

### 5. LOW — LEARNING_GUIDE: "variance blows up ~4x" should be "standard deviation ~4.4x"
File: `docs/LEARNING_GUIDE.md` (ZNE section)

With Richardson coefficients [3, -3, 1] (correct as stated), the *variance*
multiplier is 3^2+3^2+1^2 = 19; ~4.4x is the *standard-deviation* blow-up.
An ML student reader will notice. Fix: "...so the standard deviation of the
estimate grows ~4x (variance ~19x)". Everything else I fact-checked in the
guide is accurate.

## Explicitly checked, no finding
- Progress output during long runs: good (per-unit line with winner + timing;
  crash-safe append verified by testers, resume re-verified by me).
- report.md readability: good structure, honest cv_folds=0 warning, plots
  referenced by relative names render in any markdown viewer.
- Config comments (tiny/small/full/experiment.yaml): self-explanatory, include
  unit counts, runtime expectations and per-backend rationale.
- Error messages: config validation errors are excellent (they list valid
  backend/technique/family names). Tracebacks are shown for uncaught
  ValueError in run_experiment.py, but the bottom-line message is clear.
- LEARNING_GUIDE claims spot-checked against code/backends: Lagos readout
  errors, CDR 11x / ZNE 3x / REM 3x costs, endianness convention, GHZ
  parity facts, Clifford-fraction feature semantics — all accurate.
