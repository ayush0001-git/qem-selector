"""Noiseless reference expectation values via statevector simulation."""

from __future__ import annotations

from qiskit import QuantumCircuit

#: Expectations closer than this to an integer (-1, 0, +1) are snapped to it.
#: Float statevector simulation of e.g. mirror circuits (U then U-dagger)
#: leaves ~1e-16 dust on the mathematically exact value; 1e-10 is far above
#: that dust and far below any physically meaningful deviation, so snapping
#: preserves the known-answer contract (mirror <Z...Z> EXACTLY +1.0) without
#: affecting generic continuum-valued expectations.
_INTEGER_SNAP_TOL: float = 1e-10


def ideal_expectation(circuit: QuantumCircuit, pauli: str) -> float:
    """Exact noiseless expectation value of a Pauli observable.

    Args:
        circuit: QuantumCircuit WITHOUT final measurements (input state
            |0...0>). Not mutated (the statevector is built from the
            circuit's instructions without touching the circuit itself).
        pauli: Pauli string with ``len(pauli) == circuit.num_qubits``,
            characters from {I, X, Y, Z}, qemsel convention: ``pauli[i]``
            acts on qubit i. This is REVERSED vs qiskit's Pauli label
            convention (rightmost char = qubit 0), so we convert with
            ``Pauli(pauli[::-1])`` before calling qiskit.quantum_info.

    Returns:
        float in [-1, +1]: <psi| P |psi> computed exactly via
        ``qiskit.quantum_info.Statevector`` (no shots, no noise). Only the
        real part is returned (the imaginary part of a Pauli expectation on
        a pure state is zero up to float roundoff).

    Raises:
        ValueError: if the pauli length does not match circuit.num_qubits or
            contains invalid characters.
    """
    if len(pauli) != circuit.num_qubits:
        raise ValueError(
            f"pauli length {len(pauli)} != circuit num_qubits "
            f"{circuit.num_qubits}"
        )
    invalid = set(pauli) - set("IXYZ")
    if invalid:
        raise ValueError(
            f"invalid pauli characters {sorted(invalid)!r} in {pauli!r}; "
            "allowed: I, X, Y, Z"
        )

    # Import here so `import qemsel.ideal` stays cheap for non-simulation use.
    from qiskit.quantum_info import Pauli, Statevector

    statevector = Statevector.from_instruction(circuit)
    # qemsel convention (pauli[i] = qubit i) -> qiskit label: reverse.
    value = statevector.expectation_value(Pauli(pauli[::-1]))
    result = float(value.real)
    # Snap float dust: identity-equivalent circuits (mirror family, ghz_plus
    # padding) are mathematically exact eigenstates; return the exact value.
    nearest_int = round(result)
    if result != nearest_int and abs(result - nearest_int) < _INTEGER_SNAP_TOL:
        result = float(nearest_int)
    return result
