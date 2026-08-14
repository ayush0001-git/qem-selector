# Suite+Experiment Builder — research-run upgrades (2026-07-21)

Ownership: `src/qemsel/circuits.py`, `src/qemsel/mitigation.py`,
`src/qemsel/experiment.py`, `configs/research.yaml` (new),
`configs/research_smoke.yaml` (new), `tests/test_circuits.py`,
`tests/test_mitigation.py`, `tests/test_experiment.py` (extended, none
weakened). Full suite at handoff: **411 passed, 0 failed** (`pytest tests -q`,
356 s, includes other agents' concurrent additions).

## 1. circuits.py — source-level `min_abs_ideal` (family-skew fix)

* `generate_suite` gained optional config key `min_abs_ideal` (float in
  [0,1), default 0.0 = off, validated). For every non-exempt suite slot it
  rejection-samples DETERMINISTICALLY: attempt k regenerates the circuit at
  `seed + k * SUB_SEED_STRIDE` (stride 1_000_003, a prime so bumped seeds
  can never collide with other config seeds) and accepts the first whose
  exact `|<Z^n>|` ≥ threshold; capped at `MIN_ABS_IDEAL_MAX_ATTEMPTS` (50),
  then best-so-far is kept with a RuntimeWarning. The accepted (possibly
  bumped) seed is recorded in `CircuitSpec.seed`, so the reproducibility
  contract (`FAMILIES[f](n, d, spec.seed)` recreates the circuit) and
  circuit_id uniqueness hold. Byte-identical suites across calls.
* Exempt families (`MIN_ABS_IDEAL_EXEMPT_FAMILIES`): `mirror_circuit`
  (always exactly +1) AND `ghz_plus` (state is exactly GHZ at every padding
  seed — the ideal is seed-independent, so sampling is provably a no-op;
  its odd-n rows are handled by the experiment layer's `ghz_plus: X` pauli).
  I first tried a generic "3 identical ideals -> constant" early exit and
  it FALSE-POSITIVED on near_clifford (quantized Clifford ideals hit 0
  several attempts in a row by chance) — reverted in favor of the explicit
  exempt list; comment in `_sample_above_threshold` records why.
* Measured on the research grid (5 fam × n{2..5} × d{4,8,16} × 3 seeds):
  **180/180 circuits pass, perfectly balanced 36/family, 0 warnings, 1.1 s**
  to generate, 83/180 slots needed a bumped seed. The small-run skew
  (ghz_plus/mirror 24 rows vs layered_random/near_clifford 8) is gone at
  the source.

## 2. mitigation.py — raw_plus + REM damping floor

* New technique **`raw_plus`**: the empirical equal-budget baseline —
  ONE unmitigated execution at `RAW_PLUS_MULTIPLIER * base_shots` where
  `RAW_PLUS_MULTIPLIER` is derived as max of the other multipliers
  (= CDR's 11). `TECHNIQUES` is now
  `["raw", "raw_plus", "zne", "cdr", "rem"]`; `SHOT_MULTIPLIER["raw_plus"]
  = 11`; `shots_consumed` follows. Implementation REBUILDS an executor via
  `qemsel.backends.make_executor(backend, 11*shots, seed)` (module-attr
  access, monkeypatchable) because the passed executor is seed-bound —
  calling it 11× returns 11 identical values (a fake 11x). Closes the
  rebuilt executor in a `finally` if it exposes `close()` (hardware Batch).
  NOTE: mitigation now imports backends at module level — no cycle
  (backends only imports hardware lazily; verified).
* **`REM_MIN_DAMPING` 1e-6 → 0.02** (reviewer item 8): near-singular
  readout inversions now refuse as `MitigationError` instead of amplifying
  shot noise ~29x. Observed live in the smoke run: one honest `[rem]`
  refusal on FakeLagosV2@x1.5 at damping 0.019.

## 3. experiment.py — aggregation, budget abort, close(), scaled names

* **`aggregated.csv`** (seed-averaged labels, reviewer/verifier item 4):
  rewritten at the end of EVERY run from the complete DataFrame. Grouped
  by (family, n_qubits, depth, backend); columns: keys, `n_seeds`,
  `<tech>_mean_abs_error` per technique (NaN-skipping mean; NaN if all
  seeds failed), `best_technique` + `best_technique_cost_aware` recomputed
  FROM THE MEANS (cost-aware uses `shots_consumed`-derived sqrt penalty).
* **HardwareBudgetExceededError → clean whole-sweep abort** (reviewer
  item; only reachable on `ibm_*`): recognized anywhere in the exception
  CAUSE CHAIN (`apply_technique` wraps failures in MitigationError), logged
  to errors.log as `SWEEP ABORTED`, incomplete unit dropped (resume
  recomputes it), completed rows preserved, function returns normally.
  Unit-tested with fakes (direct, wrapped, resume-after-abort, and a
  regression guard that ordinary failures still NaN-and-continue).
* **`executor.close()` called in a `finally`** per unit when exposed;
  close failures are printed and swallowed (tested incl. on abort).
* **Noise-scaled backend names**: `_validate_config` now delegates name
  parsing to `backends.parse_backend_name` (the backends agent's public
  `'<Base>@x<scale>'` grammar) so validation and runtime can never
  disagree; base name checked against `BACKENDS`, width check uses the
  base device. Full scaled name flows into the `backend` column as a
  distinct noise environment.

## 4. Configs (sized from a real benchmark)

Benchmark (one unit = all 5 techniques + executor build + ideal;
hw_efficient_ansatz d=8, 4096 shots; scripts were in scratchpad, numbers
preserved in the research.yaml header):

| backend            | n=3    | n=5    |
|--------------------|--------|--------|
| FakeManilaV2       | 7.2 s* | 6.1 s  |
| FakeManilaV2@x2.0  | 3.5 s  | 5.0 s  |
| FakeLagosV2        | 4.9 s  | 7.0 s  |
| FakeJakartaV2      | 6.0 s  | 7.7 s  |
| FakeSherbrooke     | 37.1 s | 49.4 s |

(*first-call JIT.) Also measured: near_clifford CDR refuses at ANY
`non_clifford_fraction` (0.15 and 0.3 both: its non-Clifford gates are all
diagonal T/rz, so training ideals collapse) — CDR signal must come from
layered_random / hw_efficient_ansatz / mirror, which the balanced suite
provides (~756 of 1260 units).

* **`configs/research.yaml`**: 5 families × n{2,3,4,5} × d{4,8,16} ×
  seeds{0,1,2} = 180 circuits × 7 backends (Manila@{1,x1.5,x2}, 
  Lagos@{1,x1.5,x2}, Jakarta) = **1260 units**, 4096 shots, all 5
  techniques, min_abs_ideal 0.25 at BOTH layers. Estimate: 5.4 s/unit
  blended → **~1.9 h measured basis, ~3.8 h with ×2 margin ≤ 6 h** (full
  math in the header). **FakeSherbrooke dropped** without guilt: 6–8× per
  unit; as an 8th environment it alone would add ~2.3 h measured and push
  the pessimistic total past 6 h.
* **`configs/research_smoke.yaml`**: same shape, 15 circuits × 2 backends
  (plain Manila + Lagos@x1.5) = 30 units — kept 3 seeds so aggregation is
  genuinely exercised (why it is 30, not 15 units).
  **Ran it end-to-end: 101.8 s wall**, results in `results/research_smoke/`:
  30 rows, aggregated.csv 10 rows (n_seeds=3, cdr NaN means on
  near_clifford), winners cdr 15 / rem 14 / zne 1 (cost-aware rem 13 /
  cdr 11 / raw 6), errors.log = 10 intentional [cdr] refusals + 1
  intentional [rem] damping-floor refusal, no skipped_low_signal.log,
  scaled backend + bumped seeds (e.g. `layered_random_q3_d8_s2000007`)
  visible in the data.

## 5. Notes for the integrator / other agents

1. **`report.py` (`_CANONICAL_TECHNIQUES`) and any technique-literal lists
   do not know `raw_plus` yet** — report currently intersects with found
   columns, so raw_plus columns are silently IGNORED in report.md. Whoever
   owns report/model/docs should add `raw_plus` (labels can now also take
   the value `raw_plus`).
2. `conftest.py::tiny_results_df` still builds 4-technique rows
   (architect-owned, untouched) — model/report tests pass because they are
   schema-driven; fine, but a raw_plus-aware fixture would strengthen them.
3. `hardware.estimate_config_qpu_seconds` picks up raw_plus automatically
   via `SHOT_MULTIPLIER` for configs without an explicit `techniques` list
   — cost estimates for default-technique hardware configs went UP by 11
   executor-equivalents/unit. `configs/hw_first_run.yaml` pins its list
   explicitly, so it is unaffected.
4. raw_plus on `ibm_*` would open a SECOND Batch per unit (its rebuilt
   executor); it closes it in a finally. Exclude raw_plus (and cdr) from
   hardware configs anyway — budget.
5. The smoke run left `results/research_smoke/` on disk as e2e evidence;
   safe to delete or re-run (resume-safe).
6. errors.log em-dash mojibake in the PowerShell console is the known
   cosmetic issue (file content is valid UTF-8) — PROJECT_STATUS §6.12.
