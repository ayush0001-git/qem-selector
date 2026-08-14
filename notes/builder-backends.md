# Notes — builder-backends (2026-07-21)

## Delivered
- `src/qemsel/backends.py` — implemented `BACKENDS`, `get_backend_info`,
  `make_executor`, `RealHardwareBackend` (stub raises per interface), plus a
  PUBLIC helper `expectation_from_counts(counts, pauli)`.
- `src/qemsel/ideal.py` — `ideal_expectation` via exact Statevector,
  `Pauli(pauli[::-1])` conversion, ValueError on bad length/chars.
- `tests/test_backends_ideal.py` — 41 tests, all passing in ~4 s:
  `"E:\quatum  computiiing\qem-selector\.venv\Scripts\python.exe" -m pytest "E:\quatum  computiiing\qem-selector\tests\test_backends_ideal.py" -q`

## FLAG for architect/integrator
- **Test-file name mismatch:** INTERFACES.md ownership map lists
  `tests/test_backends.py` + `tests/test_ideal.py`, but my task assignment
  specified a single `tests/test_backends_ideal.py`. I followed the task
  assignment — one combined file. No `test_backends.py`/`test_ideal.py` exist.

## What integrators need to know
- **Executor contract:** `make_executor(backend_name, shots, seed)` returns
  `executor(circuit_without_meas, pauli) -> float`. Full {I,X,Y,Z} support
  (X: h, Y: sdg+h basis rotation before measure_all). All-'I' pauli returns
  1.0 WITHOUT simulating (0 shots consumed). Input circuit is never mutated.
  Noise model + AerSimulator built once per make_executor call (~1 s for the
  small backends); transpile(measured, sim, optimization_level=0,
  seed_transpiler=seed); run(seed_simulator=seed). Deterministic: same
  (circuit, pauli) twice -> identical value. ValueError on pauli length
  mismatch / invalid chars / unknown backend name (raised at make time).
- **Endianness:** all parity logic lives in `expectation_from_counts` — keys
  are reversed once so bits[i] = qubit i; hand-checked cases in tests
  ({'01': N}: 'ZI' -> -1, 'IZ' -> +1). It strips spaces from multi-register
  keys. On its own it is Z/I-only semantics (any non-'I' is treated as a
  Z-basis bit); X/Y are correct only via the executor's pre-rotation —
  documented in its docstring.
- **get_backend_info:** cached per name, returns a fresh copy each call
  (safe to mutate). Averages skip None/NaN target entries; excludes
  measure/reset/delay/barrier/id from gate averages; NaN if a category has
  no entries (never happens for the 4 BACKENDS). Measured snapshot values:
  Manila 2q=0.010 ro_avg=0.037; Jakarta 2q=0.009 ro_avg=0.031; Lagos
  2q=0.015 ro_avg=0.204 ro_max=0.464 (!); Sherbrooke n=127 2q=0.072
  (ECR-heavy) ro_max=0.500. Lagos noisy Bell ZZ ~ 0.44 vs ideal 1.0 —
  confirmed as the "REM should win" backend.
- Heavy imports (qiskit_aer, qiskit_ibm_runtime, quantum_info) are deferred
  inside functions, so `import qemsel.backends` stays cheap.
- No `@pytest.mark.slow` used: the one noisy fixture is module-scoped
  (FakeManilaV2, 2q, 256 shots) — whole file runs in ~4 s.
