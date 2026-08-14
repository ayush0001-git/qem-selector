# Spike: CDR with swappable sklearn regressors (cdr_nonlinear route decision)

2026-07-22. Owner: cdr-nl spike agent. Files owned: `spikes/spike_cdr_nl.py`,
this note. Nothing in `src/` touched; 424-test suite untouched.

## Question

Angle 2 (docs/RESEARCH_ANGLES.md) needs CDR variants with swappable sklearn
regressors (Ridge with an alpha grid, RandomForest) as selectable techniques.
In installed mitiq 1.0.0, which route?

- **Route A** — `execute_with_cdr(fit_function=..., num_fit_parameters=...)`
- **Route B** — bypass mitiq's fit: `mitiq.cdr.generate_training_circuits`
  + run the noisy executor / ideal simulator on the training circuits
  ourselves + fit any sklearn regressor + predict the target.

## DECISION: Route B for sklearn regressors. Route A only for unregularized parametric curve shapes.

### Why Route A cannot express Ridge (structural, from installed source)

`.venv/Lib/site-packages/mitiq/cdr/cdr.py`, `execute_with_cdr`:

- Lines 160–166: the regression is
  `curve_fit(lambda x, *params: fit_function(x, params), noisy[1:].T, ideal,
  p0=np.zeros(num_fit_parameters))`. `curve_fit` minimizes a **hardwired
  sum-of-squared-residuals loss**; `fit_function` shapes only the *prediction*
  f(x; params), never the *loss*. Ridge = OLS + `alpha*||w||^2` — the penalty
  lives in the loss, so **no choice of fit_function is Ridge-equivalent**.
- The textbook workaround (augment X with `sqrt(alpha)*I` pseudo-rows, y with
  zeros) needs write access to the training matrix, which is built *inside*
  `execute_with_cdr` (lines 134–157) with no injection point.
- RandomForest has no finite parameter vector at all — curve_fit-incompatible
  by construction.
- Route A **does** work for parametric fits: the spike ran a quadratic
  `fit_function` (`num_fit_parameters=3`) successfully. That is the ceiling of
  the existing `CDR_FIT_FUNCTION` / `CDR_NUM_FIT_PARAMETERS` hooks in
  `qemsel.mitigation` — polynomial ablations yes, sklearn no.

### Route B facts (all verified by running the spike)

- `generate_training_circuits(circuit, num_training_circuits,
  fraction_non_clifford, method_select='uniform', method_replace='closest',
  random_state=<int>)` is seedable, and with the same args produces the same
  training set mitiq builds internally (defaults confirmed at cdr.py:104–105:
  `uniform`/`closest`). `_apply_cdr` already pre-generates this exact set for
  its degeneracy guard — the integration can extend from there.
- **Exact executor cost: 1 (target) + N (training) noisy calls — asserted for
  both routes at N=10 and N=30.** mitiq's own path executes the identical
  count (its `scale_noise(c, 1)` at scale factor 1 is a no-op fold). So a
  future `cdr_nonlinear` technique keeps the same truthful formula
  `SHOT_MULTIPLIER = 1 + num_training_circuits` — at the current N=10 that is
  the same 11x as `cdr`, and `RAW_PLUS_MULTIPLIER` (derived max) is unchanged.
- **Equivalence check:** Route B + `LinearRegression` reproduces mitiq's
  Route A linear-CDR value to `0.00e+00` (N=10) / `8.3e-17` (N=30). The qemsel
  executor is fully deterministic per (circuit, pauli) (fixed
  `seed_simulator`/`seed_transpiler`, not a call counter — backends.py), so
  both routes see identical noisy values; mitiq's qiskit->cirq->qiskit fold
  round-trip is value-benign here.
- **Seedability:** full Route B pipeline re-run with the same seed is
  **bit-identical** (asserted for Ridge-CDR and RF-CDR); seed+1 moves the
  result (~2e-3 here). RidgeCV alpha selection uses LOO-CV — deterministic, no
  RNG; RF seeded via `random_state`.

## Spike results (layered_random q3 d8 seed=0, FakeManilaV2, 4096 shots, pauli ZZZ)

ideal +0.2697; raw error 0.1105. Errors |value − ideal|:

| technique | N=10 | N=30 | calls (N=10 / N=30) |
|---|---|---|---|
| route A linear (mitiq `cdr`) | 0.0133 | 0.0232 | 11 / 31 |
| route A quadratic fit_function | 0.0143 | 0.0234 | 11 / 31 |
| route B linear (sklearn) | 0.0133 | 0.0232 | 11 / 31 |
| route B Ridge (LOO-CV alpha grid 1e-6..1e3) | 0.0133 | 0.0232 | 11 / 31 |
| route B RandomForest (100 trees) | 0.0966 | 0.0006 | 11 / 31 |
| raw | 0.1105 | — | 1 |

Reading (single circuit — direction, not statistics):

- **Ridge ≈ linear** here: LOO-CV picks a near-zero alpha on well-spread 1-D
  training data. Matches the Korolev "regularized-linear-usually-wins /
  Ridge-insensitive" anchor the paper reproduces.
- **RF overfits at N=10** (error 7x linear's) — exactly the small-N corner the
  Angle 2 overfitting map predicts nonlinear should lose.
- RF's 0.0006 at N=30 is a **luck-of-the-leaf anecdote** (piecewise-constant
  prediction happened to land near the ideal), not evidence RF wins at N=30.
  Do not quote it; the real experiment averages over circuits/seeds.
- Caveat to carry into the experiment: **RF cannot extrapolate** — its
  prediction is bounded by the training-set ideal range, while linear/Ridge
  can extrapolate beyond it. On circuits where the target's ideal sits outside
  the near-Clifford training ideals' range, RF is structurally handicapped.
  Worth a sentence in the paper's methods.
- Training ideal spread at N=10 was 0.47 (>> `CDR_MIN_TRAINING_IDEAL_SPREAD`)
  — the existing degeneracy + fully-Clifford guards in `_apply_cdr` transfer
  to the nonlinear variants unchanged (constant training target is degenerate
  for *any* regressor).

## Integration sketch (for the builder who lands cdr_nonlinear — ADDITIVE only)

- New technique names (e.g. `cdr_ridge`, `cdr_rf`) appended behind new config
  keys; the existing `cdr` path, its guards, `TECHNIQUES` ordering for current
  configs, and byte-identical results stay untouched.
- Implementation = `_apply_cdr`'s existing prelude (transpile to
  `CDR_BASIS_GATES`, `is_clifford` guard, pre-generated training set + spread
  guard) then Route B: noisy-execute the pre-generated training circuits +
  target, fit the regressor, predict. The pre-generated set is *reused* so the
  guard and the fit see the same circuits and the call count stays 1 + N.
- Regressors: `RidgeCV(alphas=np.logspace(-6, 3, 19))` (deterministic LOO-CV)
  and `RandomForestRegressor(n_estimators=100, random_state=seed)`.
- `SHOT_MULTIPLIER[new] = 1 + num_training_circuits` (same derivation as
  `cdr`; keep it derived so it cannot drift).
- For the Angle 2 map, training-set size N and `fraction_non_clifford` become
  the sweep axes; both are plain arguments on the Route B path (no mitiq
  constraint on either).

## Repro

```powershell
cd "E:\quatum  computiiing\qem-selector"
& ".\.venv\Scripts\python.exe" spikes\spike_cdr_nl.py
```

25.5 s wall, 149 noisy executor calls total, all assertions pass
(call-count == N+1 for every route, bit-identical re-run under fixed seed).

## Re-verification (2026-07-23)

Re-ran the spike end-to-end: 23.2 s wall, 149 noisy calls, all assertions
pass, and **every number in the results table above reproduced exactly**
(ideal +0.2697, raw |err| 0.1105, linear/Ridge 0.0133 @ N=10 and 0.0232 @
N=30, RF 0.0966 / 0.0006, call counts 11 / 31, Route B linear == Route A
linear to 0.0 / 8.3e-17, seed-0 re-run bit-identical, seed-1 ridge moved
2.21e-03). Also re-confirmed the structural Route A argument against the
installed source (`.venv/Lib/site-packages/mitiq/cdr/cdr.py`): curve_fit
loss hardwired at lines 160-166, training matrix built internally at
134-142 (no injection point), `uniform`/`closest` defaults at 104-105,
fully-Clifford short-circuit at 130-131. The integration this spike
recommended has since landed in `src/qemsel/mitigation.py` as
`_apply_cdr_sklearn` (`cdr_ridge` via RidgeCV per this note's Regressors
section — the fixed-alpha variant was reverted by findings-applier
2026-07-23 — and `cdr_rf`), with `SHOT_MULTIPLIER_V2` = 1 + N as verified
here.
