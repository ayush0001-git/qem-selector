# builder-mitigation — notes

## Status: DONE, tests green

- `src/qemsel/mitigation.py` — implemented; all architect signatures verbatim
  (`TECHNIQUES`, `SHOT_MULTIPLIER`, `apply_technique`, `shots_consumed`).
- `tests/test_mitigation.py` — 40 tests, all pass standalone in ~50 s:
  `"E:\quatum  computiiing\qem-selector\.venv\Scripts\python.exe" -m pytest "E:\quatum  computiiing\qem-selector\tests\test_mitigation.py" -q`

## LOUD FLAG 1 — REM does not call `mitiq.rem.execute_with_rem`

`execute_with_rem` requires an executor returning `mitiq.MeasurementResult`
raw bitstrings; the qemsel uniform executor contract returns a single float
expectation, so the measured distribution is unavailable inside
`apply_technique`. Implemented instead: the **calibration-circuit REM variant
proven in spike_rem.py (method B)**, specialized to the single measured Pauli
at the expectation level:

- 2 calibration circuits through the SAME executor: |0...0> and |1...1> on
  the pauli's support, measured with Z on the support.
- `damping = (f0 + (-1)^k * f1) / 2` = `prod_i (1 - p0_i - p1_i)` + O(c^2),
  mitigated = raw / damping. **Exact** for symmetric readout errors, first-
  order accurate in asymmetry `c_i = p1_i - p0_i`. Raises `MitigationError`
  if |damping| < 1e-6 (near-singular readout).
- Self-calibrating through the executor => captures the actual transpiled
  layout/noise; `backend_name` is metadata-only for REM (signature kept
  verbatim). Noiseless executor => damping == 1 => exact identity (tested).
- Cost is exactly 3 executor calls, matching `SHOT_MULTIPLIER['rem'] = 3`
  ("1 execution + 2 basis-state calibrations" per the architect's comment).

## LOUD FLAG 2 — CDR short-circuits on fully-Clifford circuits

Verified in installed mitiq 1.0.0 source (`cdr.py` line 130): if the circuit
is fully Clifford, `execute_with_cdr` returns the **ideal simulator value
directly with ZERO noisy executions** => `cdr_abs_error == 0` by construction.
Experiment/report caveat: `ghz_plus` / Clifford `mirror_circuit` instances
will label `best_technique = 'cdr'` trivially. builder-experiment /
builder-docs may want to note or filter this (I did not change any interface;
it is documented in the module docstring and covered by a test).

## Implementation details

- **zne**: `zne.execute_with_zne(circ.copy(), 1-arg wrapper,
  factory=RichardsonFactory([1.0, 2.0, 3.0]),
  scale_noise=functools.partial(fold_gates_at_random, seed=seed))` — seeded
  folding per spike gotcha (unseeded Richardson is flaky under shot noise).
- **cdr**: pre-transpiles to `CDR_BASIS_GATES = ("rz","sx","x","cx")`
  (`transpile(circuit, basis_gates=..., optimization_level=0,
  seed_transpiler=seed)` — no backend passed, so no qiskit 2.5 UserWarning;
  basis verified to survive mitiq's qasm2->cirq round-trip). Then
  `execute_with_cdr(..., simulator=1-arg wrap of
  qemsel.ideal.ideal_expectation, num_training_circuits=10,
  fraction_non_clifford=0.2, random_state=seed)`. Imported as
  `from qemsel import ideal as _ideal` + attribute access, so tests (and
  anyone) can monkeypatch `qemsel.ideal.ideal_expectation`.
- **Error wrapping**: any technique-internal exception is wrapped in
  `MitigationError(RuntimeError)` with `.technique` attr and the original as
  `__cause__`. Unknown technique name raises plain `ValueError` (unwrapped).
- **Cost model**: `SHOT_MULTIPLIER = {raw:1, zne:3, cdr:11, rem:3}` is
  *derived* from module constants (`ZNE_SCALE_FACTORS`,
  `CDR_NUM_TRAINING_CIRCUITS`, `REM_NUM_CALIBRATION_CIRCUITS`); a
  parametrized test asserts the actual executor call count equals the
  multiplier for every technique.

## Noisy end-to-end smoke (scratch script, not in test suite)

3q circuit (16 CNOTs + rz content), pauli ZZZ, ideal +0.7009, 20k shots,
local executor built to the make_executor contract:

| backend | raw err | zne err | cdr err | rem err |
|---|---|---|---|---|
| FakeManilaV2 | 0.240 | 0.147 | 0.027 | 0.060 |
| FakeLagosV2 | 0.674 | 0.658 | 0.248 | 0.110 |

All techniques improve on both backends; CDR wins on gate-noise-dominated
Manila, REM wins on readout-dominated Lagos, ZNE barely helps on Lagos —
exactly the per-backend differentiation the classifier should learn.

## For the integrator

- `qemsel.ideal.ideal_expectation` was still a stub when I built; my tests
  monkeypatch it with an exact statevector implementation, so they stay green
  regardless of builder-backends progress. Once ideal.py is real, CDR uses it
  automatically (module-attribute lookup).
- mitiq emits `UserWarning: The input circuit is very short` for ZNE on
  2-gate circuits (harmless) and scipy `OptimizeWarning` for CDR's perfect
  fit on noiseless data (filtered in my tests, will not appear with real
  noise).
- Nothing outside my ownership list was modified.
