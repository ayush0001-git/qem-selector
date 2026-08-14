# Spike notes: REM (mitiq.rem, mitiq 1.0.0)

Spike file: `spikes/spike_rem.py` — runs green.
Result (FakeLagosV2, layout [1,3,5], 20000 shots, seeded):
ideal <ZZZ> = +0.8776, raw = +0.2783 (err 0.599), REM-from-props = +0.8323
(err 0.045), REM-from-calibration = +0.8397 (err 0.038). Residual ~0.04 gap
is gate noise, which REM does not target — expected.

## Exact API (introspected from installed mitiq 1.0.0)

- `mitiq.rem.execute_with_rem(circuit, executor, observable, *, inverse_confusion_matrix) -> float`
  - `observable` is REQUIRED (positional), `inverse_confusion_matrix` is KEYWORD-ONLY.
  - Accepts a qiskit `QuantumCircuit` directly (needs `ply` installed for the
    qiskit->cirq conversion — already pinned in requirements.txt).
- `mitiq.rem.mitigate_executor(executor, *, inverse_confusion_matrix)` wraps an
  executor; `mitiq.rem.mitigate_measurements(measurement_result, inv_cm)` is the
  low-level per-result transform.
- `mitiq.rem.generate_inverse_confusion_matrix(num_qubits, p0, p1)` — single
  (p0, p1) pair applied to ALL qubits (p0 = P(read 1|true 0), p1 = P(read 0|true 1)).
- `mitiq.rem.generate_tensored_inverse_confusion_matrix(num_qubits, confusion_matrices)`
  — per-qubit (or per-subsystem) 2x2 matrices, kron'ed after pinv. USE THIS for
  per-qubit backend data.
- `mitiq.raw.execute(circuit, executor, observable)` gives the unmitigated
  expectation through the same pipeline — use it for the raw baseline so raw
  and mitigated share identical measurement handling.

## Gotchas (the important part)

1. **Executor contract is different from ZNE/CDR:** the executor must return a
   `mitiq.MeasurementResult` (raw bitstrings), NOT a float expectation. The
   return-type annotation `-> MeasurementResult` on the executor function is
   REQUIRED — mitiq's `Executor` inspects annotations to classify the executor.
2. **Pass the circuit WITHOUT measurements.** mitiq appends measurements itself
   (`Observable.measure_in`), so the executor receives a circuit that already
   has measure ops (qubit i -> clbit i). Do not add your own `measure_all`.
3. **Bit order — two conventions collide:**
   - Qiskit count keys are little-endian: key = "c2 c1 c0".
   - mitiq `MeasurementResult` bitstrings are big-endian per `qubit_indices`
     (default `(0..n-1)`): bit j of the string <-> qubit j.
   - Fix: REVERSE each qiskit counts key (`key[::-1]`) before building
     `MeasurementResult`. Also strip spaces (multi-creg keys contain spaces).
4. **Confusion-matrix convention is the TRANSPOSE of qiskit's.** Qiskit
   `ReadoutError.probabilities[true][measured]` (rows = prepared state); mitiq
   wants `A[measured][true]` (columns = true state, `p_noisy = A @ p_true`).
   Transpose qiskit's matrix before handing it to
   `generate_tensored_inverse_confusion_matrix`.
5. **Kron order:** element 0 of `confusion_matrices` corresponds to qubit 0 of
   the `MeasurementResult`, which is the MOST significant bit of mitiq's
   probability-vector index (`int("b0b1b2", 2)`). So order the list
   [virtual q0, virtual q1, ...] = [physical LAYOUT[0], LAYOUT[1], ...].
6. **Getting per-qubit (p0, p1) from a V2 fake backend:** BackendV2/Target only
   exposes an aggregate `measure` error, not the asymmetric pair. Cleanest
   source: `NoiseModel.from_backend(backend).to_dict()["errors"]`, entries with
   `type == "roerror"`; `gate_qubits[0][0]` is the physical qubit and
   `probabilities` is the 2x2 qiskit-convention matrix. (On FakeLagosV2 these
   happen to be symmetric p0 == p1.)
7. **Layout must be pinned and swap-free.** Readout errors are per PHYSICAL
   qubit, so the confusion matrix is only valid if clbit i really maps to
   LAYOUT[i]. Use `transpile(..., initial_layout=LAYOUT, optimization_level=0)`
   and choose LAYOUT as a connected line on the coupling map so no SWAPs are
   inserted (Lagos coupling: 0-1, 1-2, 1-3, 3-5, 4-5, 5-6; [1,3,5] is a line).
   If routing inserts swaps, the clbit->physical map silently breaks.
8. **Avoid near-singular readout qubits.** FakeLagosV2 q2 has a 46.4% flip
   probability — `pinv` amplification ~14x turns shot noise into garbage.
   Layout [1,3,5] (13.6% / 1.7% / 26.2%) is well-conditioned and still shows a
   dramatic REM win (raw error 0.60 -> mitigated 0.04).
9. **Observable choice:** GHZ + ZZZ has ideal expectation 0 (degenerate test).
   Use rx(0.5) q0 before the CX chain -> ideal <ZZZ> = cos(0.5) ~ 0.878.
10. **REM output is stochastic beyond shot noise:** `mitigate_measurements`
    resamples bitstrings from the corrected distribution via `np.random`
    (module-level). Set `np.random.seed(...)` for reproducibility (plus
    `seed_simulator` for Aer).
11. mitiq returns numpy complex scalars (0 imaginary part) from
    `raw.execute` / `execute_with_rem` for Pauli observables — cast with
    `float(np.real(...))` before printing/comparing.
12. Calibration-circuit route (option B in the spike) matches backend
    properties to ~3 decimal places; note the |111> calibration slightly
    overestimates p1 because X-gate noise contaminates the prep (fine at this
    error scale).
