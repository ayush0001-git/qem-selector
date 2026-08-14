"""API spike: Zero-Noise Extrapolation with mitiq 1.0.0 + qiskit 2.5 / qiskit-aer 0.17.

Verified API (introspected from installed mitiq 1.0.0):
    mitiq.zne.execute_with_zne(
        circuit,                      # qiskit QuantumCircuit or cirq Circuit
        executor,                     # Callable[[circuit], float] (or Executor)
        observable=None,              # optional mitiq Observable (we bake obs into executor)
        *,
        factory=None,                 # default RichardsonFactory([1.0, 2.0, 3.0])
        scale_noise=fold_gates_at_random,   # default gate folding
        num_to_average=1,
    ) -> float

    RichardsonFactory(scale_factors), LinearFactory(scale_factors)
    After use, factory holds data: get_scale_factors(), get_expectation_values(),
    get_zero_noise_limit().

Pattern here: build the circuit WITHOUT measurements; the executor copies it,
appends measure_all(), transpiles with optimization_level=0 (so folded gates
survive), runs on AerSimulator(noise_model=NoiseModel.from_backend(fake)),
and returns the Z...Z expectation from counts (parity of each bitstring --
endianness-irrelevant for an all-Z observable, but noted for general Paulis:
qiskit count keys are little-endian, leftmost char = highest qubit index).
"""

import functools
import itertools

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector, Pauli
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime.fake_provider import FakeManilaV2

from mitiq import zne
from mitiq.zne import RichardsonFactory, LinearFactory
from mitiq.zne.scaling import fold_gates_at_random

SHOTS = 100_000  # Richardson [1,2,3] amplifies per-point shot noise ~4.4x
SEED = 7

# Deterministic gate folding: fold_gates_at_random re-randomizes on every call
# by default, which (with shot noise) makes Richardson flaky run-to-run.
FOLD = functools.partial(fold_gates_at_random, seed=SEED)


def build_circuit() -> QuantumCircuit:
    """3-qubit circuit, gate-noise dominated (many CNOTs), known ideal <ZZZ>.

    rx(theta) on each qubit gives <Z> = cos(theta) per qubit; the CNOT pairs
    compose to identity, so ideal <ZZZ> = cos(theta)^3 exactly, while the
    32 physical CNOTs supply plenty of two-qubit gate noise for ZNE to remove.
    """
    theta = 0.6
    qc = QuantumCircuit(3)
    for q in range(3):
        qc.rx(theta, q)
    for _ in range(8):  # 8 * (2+2) = 32 CNOTs, each pair == identity
        qc.cx(0, 1)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.cx(1, 2)
    return qc


def zzz_expectation_from_counts(counts: dict) -> float:
    """<Z...Z> = sum_b (-1)^popcount(b) p(b).

    NOTE qiskit bitstrings are little-endian (leftmost char = highest qubit).
    For the all-Z parity observable the bit order is irrelevant; for a general
    Pauli like ZIZ you MUST index chars as bitstring[::-1][qubit_index].
    """
    total = sum(counts.values())
    return sum((-1) ** key.count("1") * n for key, n in counts.items()) / total


def make_executor(noise_model, shots=SHOTS):
    backend = AerSimulator(noise_model=noise_model) if noise_model else AerSimulator()
    # Distinct-but-deterministic simulator seed per executor call: repeat runs
    # of this script are reproducible, yet scale points stay independent.
    seed_stream = itertools.count(SEED)

    def executor(circuit: QuantumCircuit) -> float:
        # mitiq hands back a qiskit circuit (same frontend as input), already
        # gate-folded. It has no measurements because we built it without any.
        circ = circuit.copy()
        circ.measure_all()
        # optimization_level=0: folded G G^dag G sequences must NOT be
        # simplified away. Pass only the Aer backend (it carries the noise
        # model's basis gates); adding basis_gates= triggers a qiskit 2.5
        # UserWarning about invalidating error rates.
        compiled = transpile(circ, backend, optimization_level=0, seed_transpiler=SEED)
        result = backend.run(
            compiled, shots=shots, seed_simulator=next(seed_stream)
        ).result()
        return zzz_expectation_from_counts(result.get_counts())

    return executor


def readout_floor(noisy_executor) -> float:
    """Best value ZNE can reach: same rx layer, zero CNOTs.

    ZNE extrapolates gate noise to zero but readout error does NOT scale with
    gate folding, so it survives extrapolation. This measures that floor.
    """
    qc = QuantumCircuit(3)
    for q in range(3):
        qc.rx(0.6, q)
    return noisy_executor(qc)


def main() -> None:
    circuit = build_circuit()

    # --- ideal (exact, noiseless) ---
    ideal = float(Statevector(circuit).expectation_value(Pauli("ZZZ")).real)

    # --- raw noisy ---
    noise_model = NoiseModel.from_backend(FakeManilaV2())
    noisy_executor = make_executor(noise_model)
    raw = noisy_executor(circuit)

    # --- ZNE, gate folding (seeded fold_gates_at_random), two factories ---
    rich = RichardsonFactory(scale_factors=[1.0, 2.0, 3.0])
    zne_rich = zne.execute_with_zne(
        circuit, noisy_executor, factory=rich, scale_noise=FOLD
    )

    lin = LinearFactory(scale_factors=[1.0, 2.0, 3.0])
    zne_lin = zne.execute_with_zne(
        circuit, noisy_executor, factory=lin, scale_noise=FOLD
    )

    # also confirm the all-defaults call path:
    # factory=None -> RichardsonFactory([1.0, 2.0, 3.0]),
    # scale_noise -> unseeded fold_gates_at_random
    zne_default = zne.execute_with_zne(circuit, noisy_executor)

    # what ZNE can at best reach (readout error survives extrapolation)
    floor = readout_floor(noisy_executor)

    print(f"ideal <ZZZ>                : {ideal:+.4f}  (= cos(0.6)^3)")
    print(f"readout-limited floor      : {floor:+.4f}  (gate noise = 0, ZNE's best case)")
    print(f"raw noisy                  : {raw:+.4f}   |err| = {abs(raw - ideal):.4f}")
    print(f"ZNE Richardson [1,2,3]     : {zne_rich:+.4f}   |err| = {abs(zne_rich - ideal):.4f}")
    print(f"ZNE Linear     [1,2,3]     : {zne_lin:+.4f}   |err| = {abs(zne_lin - ideal):.4f}")
    print(f"ZNE default factory        : {zne_default:+.4f}   |err| = {abs(zne_default - ideal):.4f}")
    gap = floor - raw
    if gap > 0:
        print(f"gate-noise gap recovered   : Richardson {100 * (zne_rich - raw) / gap:.0f}%, "
              f"Linear {100 * (zne_lin - raw) / gap:.0f}%")
    print()
    print("Richardson factory internals:")
    print(f"  scale factors      : {rich.get_scale_factors()}")
    print(f"  expectation values : {np.round(rich.get_expectation_values(), 4)}")
    print(f"  zero-noise limit   : {rich.get_zero_noise_limit():+.4f}")

    assert abs(zne_rich - ideal) < abs(raw - ideal), "Richardson ZNE did not improve"
    assert abs(zne_lin - ideal) < abs(raw - ideal), "Linear ZNE did not improve"
    print()
    print("PASS: both factories moved the estimate closer to ideal than raw.")


if __name__ == "__main__":
    main()
