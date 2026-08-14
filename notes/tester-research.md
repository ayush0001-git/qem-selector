# Independent tester — research-run integration verification (2026-07-21)

Role: independent tester (wrote none of the code under test). All runs fresh,
all recomputations done with plain pandas/numpy in my own scripts (scratchpad:
`verify_smoke_tester.py`, `test_model_synthetic.py`, `probe_x2_monotonic.py`,
`compare_runs_tester.py`). **Verdict: PASS — zero critical/major findings.**

## 1. Full suite twice (flake check) — PASS
`pytest tests -q` (slow included), two runs back-to-back:
- run 1: **411 passed, 0 failed**, 3 benign mitiq warnings, 123.6 s
- run 2: **411 passed, 0 failed**, 3 benign mitiq warnings, 128.5 s
Zero flakes. Logs: scratchpad `pytest_run1.log` / `pytest_run2.log`.

Repro: `& ".\.venv\Scripts\python.exe" -m pytest tests -q`

## 2. research_smoke e2e, fresh dir + independent pandas recomputation — PASS
Fresh run of `configs\research_smoke.yaml` into a scratchpad dir (exit 0).
26/26 checks in my own verifier PASS:
- 45 rows, unique (circuit_id, backend), backends exactly
  {FakeManilaV2, FakeLagosV2, FakeLagosV2@x1.5}.
- **raw_plus sanity:** columns present; `raw_plus_value != raw_value` on
  45/45; `raw_plus_shots == 45056 == 11 x 4096`; pooled mean |err|
  raw_plus 0.5604 vs raw 0.5615 (matches integrator's numbers exactly).
- `<tech>_abs_error == |value - ideal|` for all 5 techniques.
- Per-row `best_technique` == my recomputed argmin on 45/45.
- **aggregated.csv:** 15 groups, n_seeds=3 everywhere; every
  `<tech>_mean_abs_error` matches my NaN-skipping recomputation (<1e-9);
  `best_technique` == argmin(means) 15/15; `best_technique_cost_aware` ==
  argmin(mean * sqrt(SHOT_MULTIPLIER)) 15/15.
- **Noise monotonicity across smoke rows:** mean raw abs_error
  Lagos@x1.5 = 0.7151 > plain Lagos = 0.7033; paired per-circuit the
  scaled row is worse on 87% (13/15).
- NaN audit: raw/raw_plus/zne 0%; cdr 15/45 (33.3%), rem 2/45 (4.4%);
  errors.log = 17 lines, all `[cdr]`/`[rem]` refusals, count == NaN count.
- Row winners rem 24 / cdr 20 / zne 1; aggregated winners cdr 11 / rem 4
  (the row->aggregate flip is the intended seed-averaging effect; both
  label sets verified against independent argmins).

NOTE (task-spec nit, not a defect): the task said "x2.0 > x1.0 across smoke
rows" but research_smoke.yaml contains x1.5, not x2.0 (only research.yaml has
x2.0). I verified x1.5 > x1.0 on the smoke rows AND ran a direct executor
probe for x2.0: fixed 20-CNOT circuit, 8192 shots —
Lagos |err| 0.60 (x1.0) < 0.78 (x1.5) < 0.91 (x2.0); Manila 0.30 (x1.0)
< 0.45 (x2.0). Monotonic as claimed.

Repro: `& ".\.venv\Scripts\python.exe" scripts\run_experiment.py --config
configs\research_smoke.yaml --out <fresh-dir>` then run
`verify_smoke_tester.py <fresh-dir>`.

## 3. Determinism (research_smoke twice) — PASS
Two independent fresh runs (smoke_a, smoke_b): `results.csv`
**byte-identical** (filecmp), therefore all estimate columns exactly equal
(also checked column-by-column incl. NaN positions); winner columns equal;
`aggregated.csv` byte-identical too.

## 4. tiny.yaml regression — PASS
Fresh run into a scratchpad dir: `results.csv` **byte-identical** to the
pre-existing reference `results\tiny\results.csv`; winners exactly
**rem 11 / cdr 8 / zne 1**; errors.log exactly 8 lines, all `[cdr]`,
byte-identical to the reference errors.log. The noise-scaling /raw_plus
changes did not perturb the legacy plain-backend path.

## 5. model.py synthetic probes — PASS
`test_model_synthetic.py` (scratchpad), calling `qemsel.model.train_and_eval`
directly on synthetic DataFrames in the exact experiment schema:

**Case A — 1-member class (49 rows: cdr 24 / rem 24 / zne 1):**
- WARNING printed; `dropped_classes == ['zne']`; **CV still runs**
  (cv_folds=5, stratified_group); `cv_n_samples == 48 == n_samples-1`;
  confusion-matrix zne row all-zero; refit bundle still carries the zne
  class (`'zne' in model.classes_`). Exactly the documented behavior.

**Case B — learnable rules, LOFO/LOBO plausibility:**
- Backend-keyed rule (readout>0.1 -> rem): CV acc 1.000 vs baseline 0.500;
  LOFO present, acc 0.980 over 5 families; LOBO present — acc 0.000, which
  is the CORRECT honest result (a backend-determined label cannot be
  predicted for a held-out backend; the model learns the inverse mapping).
- Backend-independent rule (clifford_fraction family rule, 3 backends incl.
  an @x1.5 name): CV 5-fold acc 1.000, **LOBO 1.000** across all 3 backend
  environments, LOFO 1.000 over 5 families, dropped_classes []. LOFO/LOBO
  respond exactly as they should to learnable vs unlearnable structure.

## Findings
- **None critical or major.** All integrator claims I could re-derive
  (411/411 twice, 45/45 smoke rows, 0.5604/0.5615 raw_plus/raw pooled
  errors, 0.7151/0.7033 Lagos scaled/plain, 33%/4.4% cdr/rem NaN,
  17 refusal-only errors.log lines, tiny rem 11/cdr 8/zne 1 byte-identical)
  reproduced exactly.
- MINOR/informational: (a) the tester task text referenced "x2.0" for the
  smoke monotonicity check but the smoke config uses x1.5 (x2.0 covered by
  direct probe above and by the 39-test noise-scaling suite); (b) shared
  scratchpad already contained other agents' scripts, incl. an unrelated
  `compare_runs.py` — mine are suffixed `_tester` to avoid collision;
  (c) PROJECT_STATUS §6.2 (no explicit CDR-refusal indicator feature)
  remains open, as the integrator stated.

Green light from the independent-test side for launching the research sweep
(`configs\research.yaml`, ~4.9 h estimate).
