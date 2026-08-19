"""Verify that the new high-level MitigatedExecutor API works correctly."""

import logging

logging.basicConfig(level=logging.WARNING, format="%(message)s")

from qiskit import QuantumCircuit

from qemsel.api import MitigatedExecutor


def main():
    print("--- Testing High-Level MitigatedExecutor API ---")

    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "..", "results", "boundary", "model_significant.joblib")
    print(f"Loading model from: {model_path}")
    executor = MitigatedExecutor(model_path)

    # 2. Create a simple 3-qubit GHZ circuit
    print("Creating a 3-qubit GHZ circuit...")
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)

    # 3. Execute with automatic mitigation selection
    print("Executing circuit on FakeLagosV2 with 1024 base shots...")
    result = executor.execute(
        circuit=qc,
        pauli="ZZZ",
        backend_name="FakeLagosV2",
        base_shots=1024,
        seed=42
    )

    # 4. Display results
    print("\n--- Execution Successful ---")
    print(f"AI Recommended Technique: {result['technique'].upper()}")
    print(f"Mitigated Expectation Value <ZZZ>: {result['value']:.5f}")
    print("\nProbabilities assigned by AI:")
    for tech, prob in result['probabilities'].items():
        print(f"  - {tech}: {prob:.2%}")

    print("\nExtracted features used by AI:")
    for feat, val in list(result['features'].items())[:5]: # Show first 5 features
        print(f"  - {feat}: {val}")
    print("  - ...")

if __name__ == "__main__":
    main()
