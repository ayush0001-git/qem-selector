# builder-features — notes

## Status: DONE, tests green

- `src/qemsel/features.py`: `FEATURE_NAMES` (exact 10-name order) + `extract_features(circuit, backend_name)`.
- `tests/test_features.py`: 26 tests, all pass standalone (`.venv\Scripts\python.exe -m pytest tests/test_features.py -q`, 0.12s).

## Implementation decisions (integrator, read this)

1. **Deferred backend import.** `extract_features` imports `qemsel.backends` inside the function body and calls `backends.get_backend_info(backend_name)` at call time. backends.py was still a stub when I built this, so my tests monkeypatch `qemsel.backends.get_backend_info` with a contract-faithful fake (known name -> info dict, unknown -> ValueError). Once builder-backends lands, NO change is needed here — the attribute lookup picks up the real function automatically.
2. **Unknown backend_name validation is DELEGATED** to `get_backend_info` (single source of truth, per its contract it raises ValueError). features.py does not pre-check against `BACKENDS`.
3. **Clifford set**: docstring set plus `id` (identity is trivially Clifford; ghz_plus padding may emit it). Angle rule applied to `rx/ry/rz` AND `p` (p(pi/2)=S): Clifford iff angle is an integer multiple of pi/2 within 1e-9, via `abs(math.remainder(angle, pi/2)) <= 1e-9`. Unbound ParameterExpressions -> non-Clifford. `t/tdg/u/...` -> non-Clifford.
4. **barrier/delay excluded** from all counts; `measure` raises ValueError.
5. **Defensive 3+-qubit handling** (minor spec deviation, only fires if a family ever emits ccx etc., which none do): 3+-qubit gates go into neither n_1q nor n_2q but ARE counted in the clifford_fraction denominator (as non-Clifford) so the fraction stays in [0,1]. For all real qemsel circuits, total_gates == n_1q + n_2q exactly as specced.
6. Returned dict is built in FEATURE_NAMES insertion order (asserted in-module); all values are plain Python `float`.
7. `depth_per_qubit` guards n_qubits==0 -> 0.0; empty circuit -> clifford_fraction 1.0 per spec.

No interface signature was changed. Nothing else flagged.
