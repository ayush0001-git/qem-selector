"""API spike: Readout Error Mitigation (REM) with mitiq 1.0.0.

Technique: readout confusion inversion (mitiq.rem.execute_with_rem).
The inverse confusion matrix is built TWO ways and both are demonstrated:
  (A) from the fake backend's stored readout-error properties
      (extracted from NoiseModel.from_backend(...).to_dict() "roerror" entries)
  (B) empirically, from two calibration circuits (|000> and |111>) run on the
      same noisy simulator + layout (per-qubit flip rates -> 2x2 matrices)

Circuit: 3-qubit GHZ-like state rx(0.5) q0; cx(0,1); cx(1,2)
Observable: ZZZ, ideal <ZZZ> = cos(0.5) ~= 0.8776 (exactly, via Statevector).

Backend: FakeLagosV2 (large readout errors -> strong REM signal).
Physical layout [1, 3, 5]: a connected line on Lagos' coupling map
(1-3, 3-5), so optimization_level=0 + initial_layout inserts no swaps and
the clbit -> physical-qubit mapping stays exactly [1, 3, 5]. Readout flip
probabilities there: q1 ~13.6%, q3 ~1.7%, q5 ~26.2% (symmetric).

Run:
  "E:\\quatum  computiiing\\qem-selector\\.venv\\Scripts\\python.exe" spikes/spike_rem.py
"""

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Pauli, Statevector
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime.fake_provider import FakeLagosV2

import mitiq.raw
from mitiq import MeasurementResult, Observable, PauliString
from mitiq.rem import (
    execute_with_rem,
    generate_tensored_inverse_confusion_matrix,
)

SHOTS = 20000
LAYOUT = [1, 3, 5]  # physical qubits (a coupled line on FakeLagosV2)
NUM_QUBITS = 3
SEED = 1234

np.random.seed(SEED)  # mitiq's mitigate_measurements resamples via np.random

backend = FakeLagosV2()
noise_model = NoiseModel.from_backend(backend)
sim = AerSimulator.from_backend(backend)

# ---------------------------------------------------------------- circuit ---
# NOTE: no measurements here. mitiq appends measurements itself via
# Observable.measure_in (executor receives a circuit WITH measurements).
circuit = QuantumCircuit(NUM_QUBITS)
circuit.rx(0.5, 0)
circuit.cx(0, 1)
circuit.cx(1, 2)

ideal = float(Statevector(circuit).expectation_value(Pauli("ZZZ")).real)

# --------------------------------------------------------------- executor ---
def counts_to_measurement_result(counts: dict) -> MeasurementResult:
    """Qiskit counts -> mitiq MeasurementResult.

    Qiskit count keys are little-endian ("c2 c1 c0"); mitiq bitstrings are
    big-endian per qubit_indices (bit j <-> qubit_indices[j], default
    (0..n-1)). So REVERSE each key: reversed key bit i = clbit i = virtual
    qubit i (mitiq measures qubit i into clbit i).
    """
    bitstrings = []
    for key, n in counts.items():
        bs = key.replace(" ", "")[::-1]
        bitstrings.extend([bs] * n)
    return MeasurementResult(bitstrings)


def executor(circ: QuantumCircuit) -> MeasurementResult:
    """Noisy executor. Return-type annotation MeasurementResult is REQUIRED:
    mitiq's Executor uses it to know this executor returns raw samples."""
    tqc = transpile(
        circ,
        backend=sim,
        initial_layout=LAYOUT,
        optimization_level=0,
    )
    counts = sim.run(tqc, shots=SHOTS, seed_simulator=SEED).result().get_counts()
    return counts_to_measurement_result(counts)


# mitiq PauliString "ZZZ" = Z(0)*Z(1)*Z(2) (left char -> qubit 0)
obs = Observable(PauliString("ZZZ"))

# ------------------- (A) confusion matrix from backend readout properties ---
# NoiseModel.from_backend stores per-qubit readout errors as "roerror"
# entries: probabilities[true][measured] (rows = prepared/true state).
# mitiq's confusion-matrix convention is columns = true state
# (p_noisy = A @ p_true), i.e. the TRANSPOSE of qiskit's matrix.
qiskit_ro = {
    e["gate_qubits"][0][0]: np.array(e["probabilities"])
    for e in noise_model.to_dict()["errors"]
    if e["type"] == "roerror"
}
# List order matters: element 0 <-> qubit 0 of the MeasurementResult, which is
# the MOST significant bit of mitiq's probability-vector index (mitiq indexes
# bitstrings big-endian: int("b0b1b2", 2)).
cms_props = [qiskit_ro[q].T for q in LAYOUT]
inv_cm_props = generate_tensored_inverse_confusion_matrix(
    NUM_QUBITS, confusion_matrices=cms_props
)

# ------------------- (B) confusion matrix from calibration circuits ---------
def calibration_confusion_matrices() -> list[np.ndarray]:
    """Per-qubit 2x2 confusion matrices measured with |000> and |111> preps
    on the same simulator/layout (assumes uncorrelated readout errors)."""
    p0 = np.zeros(NUM_QUBITS)  # P(read 1 | prepared 0)
    p1 = np.zeros(NUM_QUBITS)  # P(read 0 | prepared 1)
    for prep, arr in (("0", p0), ("1", p1)):
        qc = QuantumCircuit(NUM_QUBITS, NUM_QUBITS)
        if prep == "1":
            qc.x(range(NUM_QUBITS))
        qc.measure(range(NUM_QUBITS), range(NUM_QUBITS))
        tqc = transpile(
            qc, backend=sim, initial_layout=LAYOUT, optimization_level=0
        )
        counts = (
            sim.run(tqc, shots=SHOTS, seed_simulator=SEED + 1)
            .result()
            .get_counts()
        )
        for key, n in counts.items():
            bs = key.replace(" ", "")[::-1]
            for i, b in enumerate(bs):
                if b != prep:
                    arr[i] += n
    p0 /= SHOTS
    p1 /= SHOTS
    # mitiq convention: A[measured][true], columns sum to 1
    return [
        np.array([[1.0 - p0[i], p1[i]], [p0[i], 1.0 - p1[i]]])
        for i in range(NUM_QUBITS)
    ]


cms_cal = calibration_confusion_matrices()
inv_cm_cal = generate_tensored_inverse_confusion_matrix(
    NUM_QUBITS, confusion_matrices=cms_cal
)

# ------------------------------------------------------------------- runs ---
# mitiq returns numpy complex with 0 imaginary part for Pauli observables;
# cast to real float.
raw = float(np.real(mitiq.raw.execute(circuit, executor, obs)))
mitigated_props = float(np.real(execute_with_rem(
    circuit, executor, obs, inverse_confusion_matrix=inv_cm_props
)))
mitigated_cal = float(np.real(execute_with_rem(
    circuit, executor, obs, inverse_confusion_matrix=inv_cm_cal
)))

# ----------------------------------------------------------------- report ---
per_qubit_flips = {
    q: float(qiskit_ro[q][0, 1]) for q in LAYOUT
}  # P(1|0) from properties
print(f"backend                 : FakeLagosV2, physical layout {LAYOUT}")
print(f"readout flip probs (props): "
      + ", ".join(f"q{q}={p:.3f}" for q, p in per_qubit_flips.items()))
cal_flips = [float(cm[1, 0]) for cm in cms_cal]
print(f"readout flip probs (cal)  : "
      + ", ".join(f"q{q}={p:.3f}" for q, p in zip(LAYOUT, cal_flips)))
print(f"shots                   : {SHOTS}")
print()
print(f"ideal      <ZZZ>        : {ideal:+.4f}")
print(f"raw noisy  <ZZZ>        : {raw:+.4f}   (err {abs(raw - ideal):.4f})")
print(f"REM (backend props)     : {mitigated_props:+.4f}   "
      f"(err {abs(mitigated_props - ideal):.4f})")
print(f"REM (calibration circs) : {mitigated_cal:+.4f}   "
      f"(err {abs(mitigated_cal - ideal):.4f})")

err_raw = abs(raw - ideal)
err_props = abs(mitigated_props - ideal)
err_cal = abs(mitigated_cal - ideal)
assert err_props < err_raw, (
    f"REM (props) did not improve: {err_props:.4f} >= {err_raw:.4f}"
)
assert err_cal < err_raw, (
    f"REM (cal) did not improve: {err_cal:.4f} >= {err_raw:.4f}"
)
print()
print("SPIKE OK: both REM variants land closer to ideal than raw.")
