# Notes — environment setup agent (2026-07-21)

## What was done
- Reused the existing venv at `.venv` (Python 3.12.3) left by a previous partial attempt; only pip was installed in it. Did NOT delete/recreate.
- Installed in one resolver pass: mitiq qiskit qiskit-aer qiskit-ibm-runtime scikit-learn pandas matplotlib pyyaml pytest joblib. No dependency conflict — pip resolved mitiq 1.0.0 with qiskit 2.5.0 cleanly.
- Added `ply==3.11` afterwards: mitiq's qiskit frontend (`mitiq.interface.mitiq_qiskit.conversions`) imports `cirq.contrib.qasm_import`, whose parser needs ply, and nothing declares it. Symptom without it: `ModuleNotFoundError: No module named 'ply'` inside `execute_with_zne`.
- Wrote `requirements.txt` (exact pins) and `scripts/verify_env.py` (rerunnable check; exits 0 = healthy). A previous attempt had drafted verify_env.py; I fixed its backend preference order and a transpile warning.

## Verification results (all passed)
- Imports: qiskit 2.5.0, qiskit_aer 0.17.2, qiskit_ibm_runtime 0.48.0, mitiq 1.0.0, sklearn 1.9.0, pandas 2.3.3, matplotlib 3.11.1, yaml 6.0.3.
- 41 `Fake*V2` classes in `qiskit_ibm_runtime.fake_provider`.
- Noisy Bell on AerSimulator + `NoiseModel.from_backend(FakeManilaV2())`: counts {'11': 1955, '00': 1895, '10': 123, '01': 123}, P(00)+P(11)=0.94.
- ZNE smoke test on qiskit circuit: unmitigated 0.9657 -> mitigated 0.9572 (readout-dominated toy; machinery works).
- `mitiq.zne/.cdr/.rem` all import; `execute_with_zne/execute_with_cdr/execute_with_rem` present.

## Backend noise characterization (empirical, 4096 shots, measure-only vs Bell)
| Backend | readout P(00|prep 00) | Bell P(00)+P(11) |
|---|---|---|
| FakeManilaV2 (5q) | 0.939 | 0.938 |
| FakeJakartaV2 (7q) | 0.955 | 0.954 |
| FakeSherbrooke (127q, no V2 suffix) | 0.962 | 0.968 |
| FakeLagosV2 (7q) | 0.728 | 0.737 |

FakeLagosV2's snapshot has ~27% readout error on q0/q1 — deliberately keep it in the benchmark pool as the "REM should win here" case.

## Pitfalls for later agents
- Paths: parent dir "E:\quatum  computiiing" has DOUBLE spaces. Quote everything.
- Python `-c` with inline code through PowerShell strips inner double quotes (f-strings break). Write a .py file and run it instead.
- qiskit 2.5: passing `basis_gates=` AND a backend to `transpile()` raises a UserWarning; pass only the AerSimulator built with the noise model. Use `optimization_level=0` inside mitiq executors so folded gates survive.
- mitiq 1.0.0 API: top-level has `zne, cdr, rem, pec, ddd, lre, qse, pt, calibration, benchmarks, Executor, Observable, PauliString, execute_with_mitigation, qem_methods`. `ZNE_SETTINGS` and `Calibrator` exist for auto-calibration.
