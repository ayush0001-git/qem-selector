# Notes — architect (2026-07-21)

## What was done
- `pyproject.toml`: setuptools src-layout, package `qemsel` v0.1.0, NO runtime deps
  listed (env is pinned by requirements.txt; keeping [project] deps empty stops
  `pip install -e .` from re-resolving pins). `pip install -e .` succeeded;
  `import qemsel` resolves to `src/qemsel/__init__.py`.
- Stubs with full docstring contracts (body = `raise NotImplementedError`) for:
  circuits, backends, ideal, mitigation, features, experiment, model, recommend,
  report. `__init__.py` is version-string-only ON PURPOSE — no submodule imports,
  so `import qemsel` never breaks while builders' modules are half-done.
- `tests/conftest.py` FULLY IMPLEMENTED (not stubs): `tiny_circuit` (2q Bell),
  `tiny_identity_circuit` (3q identity), `fake_executor` (exact statevector,
  satisfies the make_executor contract), `out_dir`, `tiny_results_df` (16 rows,
  exact experiment schema, 4 rows/class). Verified: fixtures work, fake_executor
  gives Bell <ZZ>=<XX>=1, <ZI>=0; df argmin(abs_error)==best_technique per row.
- `INTERFACES.md`: all signatures, ownership map, binding conventions.
- Verification script run (scratchpad/check_skeleton.py): all 9 modules import,
  all stubs raise NotImplementedError, constants + dataclass + signatures checked.

## Key design decisions builders MUST notice
1. **Pauli convention: `pauli[i]` acts on qubit i** (leftmost = q0). REVERSED vs
   qiskit `Pauli` labels (`Pauli(pauli[::-1])` to convert) and qiskit counts are
   little-endian (rightmost bit = q0). Stated in backends.py module docstring,
   ideal.py, conftest, INTERFACES.md.
2. Executor contract is `(circuit_without_measurements, pauli_str) -> float`;
   heavy objects (noise model/simulator) built once in make_executor closure.
   mitiq wants 1-arg executors — mitigation wraps with `lambda c: executor(c, pauli)`.
3. `CircuitSpec.circuit_id` property = `{family}_q{n}_d{d}_s{s}` — the resume key
   in experiment.py together with backend name.
4. `SHOT_MULTIPLIER = {'raw':1,'zne':3,'cdr':11,'rem':3}` in mitigation.py —
   builder-mitigation may tune the VALUES but must keep them truthful to what
   apply_technique actually executes; shots_consumed = base_shots * multiplier.
5. experiment row schema (exact col names) is mirrored by conftest's
   `tiny_results_df`; model/report builders can develop entirely against that
   fixture without simulation. If builder-experiment must deviate from the
   schema, conftest (architect-owned) has to change in the same commit — flag in
   PROJECT_STATE.md rather than silently diverging.
6. model.joblib is a BUNDLE dict {'model','feature_names','classes','model_name',
   'qemsel_version'} — recommend.py depends on this shape, not on a bare estimator.
7. conftest duplicates TECHNIQUES/FEATURE_NAMES as literals deliberately, so it
   imports even while qemsel modules are stubs. If the canonical lists change,
   update conftest too (architect ownership).
8. mirror_circuit is the known-answer family (ideal exactly +1 for 'Z'*n).
   ghz_plus ideal <Z...Z> is 1 only for EVEN n — everyone must use
   ideal.ideal_expectation, never assume.

## Open items for builders
- builder-experiment additionally owns configs/experiment.yaml (small default:
  2-3 qubits, FakeManilaV2+FakeLagosV2, 2048 shots) and configs/hardware.yaml
  placeholder (`ibm_token: null`) — referenced by RealHardwareBackend's message.
- CDR needs `ply` (installed) and uses ideal.ideal_expectation as its noiseless
  simulator — do not spin up a second Aer statevector path.
- Keep noisy tests tiny (<=3 qubits, <=256 shots) or mark `@pytest.mark.slow`.
