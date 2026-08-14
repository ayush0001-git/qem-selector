"""API spike: Clifford Data Regression (CDR) via mitiq.cdr.execute_with_cdr.

mitiq 1.0.0 / qiskit 2.5.0 / qiskit-aer 0.17.2.

Key API facts (verified against installed source in
.venv/Lib/site-packages/mitiq/cdr/cdr.py):

* Signature:
    execute_with_cdr(circuit, executor, observable=None, *,
                     simulator,                 # KEYWORD-ONLY, required
                     num_training_circuits=10,
                     fraction_non_clifford=0.1,
                     fit_function=linear_fit_function,
                     num_fit_parameters=None,
                     scale_factors=(1,),
                     scale_noise=fold_gates_at_random,
                     **kwargs)  # method_select, method_replace,
                                # sigma_select, sigma_replace, random_state
* Both `executor` (noisy) and `simulator` (ideal/near-Clifford-capable) are
  plain callables circuit -> float (they get wrapped in mitiq.Executor
  automatically). With observable=None BOTH must return an expectation value
  as a float.
* Training circuits are generated INTERNALLY by
  mitiq.cdr.generate_training_circuits (num_training_circuits copies of the
  input circuit with ~(1 - fraction_non_clifford) of the non-Clifford gates
  replaced by Cliffords; selection/replacement controlled by
  method_select/method_replace kwargs). You never pass them yourself.
  generate_training_circuits is wrapped in @atomic_one_to_many_converter, so a
  qiskit input circuit yields qiskit training circuits — our executors always
  receive qiskit QuantumCircuits (without measurements; executor must add
  them).
* The input circuit must have all its non-Clifford gates as RZ rotations
  (i.e. compiled to ~{rz, sx/h, cx}); otherwise gate replacement cannot work.
"""

import itertools

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime.fake_provider import FakeManilaV2

from mitiq.cdr import execute_with_cdr

SHOTS = 20_000
N_QUBITS = 2

# ---------------------------------------------------------------------------
# 1. Circuit of interest: 2 qubits, Clifford scaffolding (h, cx) plus
#    non-Clifford RZ rotations so CDR has something to replace.
#    All non-Clifford content is in the rz gates, as CDR requires.
# ---------------------------------------------------------------------------
qc = QuantumCircuit(N_QUBITS)
angles = [0.15, 0.2, 0.25, 0.1]  # deliberately not multiples of pi/2
for _ in range(3):  # repeat layers -> enough depth for visible gate noise
    # h-rz-h == Rx-style rotation (all non-Clifford content is in the rz)
    qc.h(0)
    qc.h(1)
    qc.rz(angles[0], 0)
    qc.rz(angles[1], 1)
    qc.h(0)
    qc.h(1)
    # entangling block with rz phases between the CNOTs
    qc.cx(0, 1)
    qc.rz(angles[2], 0)
    qc.rz(angles[3], 1)
    qc.cx(0, 1)

# ---------------------------------------------------------------------------
# 2. Observable: Z on every qubit ("Z...Z"). Computed from counts by parity.
#    NOTE qiskit count keys are little-endian (qubit 0 = RIGHTMOST char), but
#    for Z on ALL qubits the parity of the whole bitstring is endian-agnostic.
#    The helper below indexes little-endian explicitly so it also works for a
#    Z observable on a subset of qubits.
# ---------------------------------------------------------------------------


def z_expectation_from_counts(counts: dict, qubits=None) -> float:
    """<Z...Z> on `qubits` (default: all) from qiskit counts (little-endian)."""
    shots = sum(counts.values())
    total = 0.0
    for key, n in counts.items():
        bits = key.replace(" ", "")[::-1]  # reverse -> bits[i] == qubit i
        idx = range(len(bits)) if qubits is None else qubits
        parity = sum(int(bits[q]) for q in idx) % 2
        total += n if parity == 0 else -n
    return total / shots


# ---------------------------------------------------------------------------
# 3. Noisy executor: AerSimulator + NoiseModel.from_backend(FakeManilaV2).
#    mitiq hands us a qiskit circuit WITHOUT measurements -> add measure_all.
#    optimization_level=0 so any folded/replaced gates are not optimized away.
# ---------------------------------------------------------------------------
backend = FakeManilaV2()
noise_model = NoiseModel.from_backend(backend)
noisy_sim = AerSimulator(noise_model=noise_model)
_seed = itertools.count(1234)


def noisy_executor(circuit: QuantumCircuit) -> float:
    circ = circuit.copy()
    circ.measure_all()
    tqc = transpile(circ, noisy_sim, optimization_level=0)
    result = noisy_sim.run(
        tqc, shots=SHOTS, seed_simulator=next(_seed)
    ).result()
    return z_expectation_from_counts(result.get_counts())


# ---------------------------------------------------------------------------
# 4. Ideal (noiseless) simulator for the near-Clifford training circuits.
#    Exact statevector expectation -> no shot noise in the training labels.
#    ("ZZ" as a SparsePauliOp label is symmetric, so endianness is moot.)
# ---------------------------------------------------------------------------
ZOBS = SparsePauliOp("Z" * N_QUBITS)


def ideal_simulator(circuit: QuantumCircuit) -> float:
    circ = circuit.remove_final_measurements(inplace=False)
    return float(Statevector(circ).expectation_value(ZOBS).real)


# ---------------------------------------------------------------------------
# 5. Run: ideal / raw noisy / CDR-mitigated.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ideal = ideal_simulator(qc)
    raw = noisy_executor(qc)

    mitigated = execute_with_cdr(
        qc,
        noisy_executor,
        simulator=ideal_simulator,   # keyword-only
        num_training_circuits=30,
        fraction_non_clifford=0.2,   # keep ~20% of the 12 rz gates non-Clifford
        random_state=0,              # seed for training-circuit sampling
    )

    err_raw = abs(raw - ideal)
    err_mit = abs(mitigated - ideal)
    print(f"ideal      <ZZ> = {ideal:+.4f}")
    print(f"raw noisy  <ZZ> = {raw:+.4f}   |err| = {err_raw:.4f}")
    print(f"CDR mitig. <ZZ> = {mitigated:+.4f}   |err| = {err_mit:.4f}")
    print(f"error reduction factor = {err_raw / max(err_mit, 1e-12):.2f}x")
    if err_mit < err_raw:
        print("SUCCESS: CDR moved the estimate closer to the ideal value.")
    else:
        print("WARNING: CDR did not improve over raw (investigate).")
