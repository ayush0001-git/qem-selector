"""Production-grade high-level API for qemsel.

Exposes MitigatedExecutor, which automatically extracts features, queries the
AI model for a cost-aware QEM recommendation, and executes the mitigated circuit
under the planned shot budget.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from qiskit import QuantumCircuit

_log = logging.getLogger(__name__)

from qemsel.backends import make_executor
from qemsel.mitigation import apply_technique
from qemsel.recommend import recommend


class MitigatedExecutor:
    """Production-ready QEM Selector runner.

    Encapsulates a trained model bundle and exposes a simple interface to execute
    circuits with automatically-chosen optimal error mitigation.

    Example::

        from qiskit import QuantumCircuit
        from qemsel.api import MitigatedExecutor

        # Initialize with the path to the trained model
        executor = MitigatedExecutor("results/boundary/model_significant.joblib")

        # Define circuit and parameters
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(1, 2)

        result = executor.execute(
            circuit=qc,
            pauli="ZZZ",
            backend_name="FakeLagosV2",
            base_shots=1024,
            seed=42
        )

        print(f"Mitigated expectation value: {result['value']}")
        print(f"AI Recommended technique: {result['technique']}")
    """

    def __init__(self, model_path: str | Path):
        """Initialize the executor with a trained model bundle.

        Args:
            model_path: Path to the trained `model_significant.joblib` bundle.
        """
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {self.model_path}. "
                "Ensure you have trained the model using scripts/train_model.py."
            )

    def execute(
        self,
        circuit: QuantumCircuit,
        pauli: str,
        backend_name: str,
        base_shots: int,
        seed: int = 42,
        fallback_technique: str = "raw",
    ) -> dict[str, Any]:
        """Query the model for the best QEM technique and execute the circuit.

        Args:
            circuit: QuantumCircuit WITHOUT final measurements.
            pauli: Pauli string representing the observable (e.g. 'ZZZ').
            backend_name: Target backend name (e.g. 'FakeLagosV2').
            base_shots: The planned base shot budget.
            seed: Seed for reproducibility of simulations.
            fallback_technique: Default technique to use if the model abstains
                or fails.

        Returns:
            Dict containing the mitigated value, recommended technique,
            extracted features, and recommendation probabilities.
        """
        # 1. Query the recommender for the best technique
        recommendation = recommend(
            self.model_path,
            circuit,
            backend_name,
            base_shots=base_shots
        )

        tech = recommendation["technique"]
        abstained = recommendation.get("abstained", False)

        # Handle non-executable outcomes (abstain or tie)
        if abstained or tech in ("abstain", "tie"):
            tech = fallback_technique

        # 2. Build the noisy backend executor
        raw_executor = make_executor(backend_name, base_shots, seed)

        # 3. Apply the recommended mitigation technique
        try:
            mitigated_value = apply_technique(
                name=tech,
                circuit=circuit,
                pauli=pauli,
                executor=raw_executor,
                backend_name=backend_name,
                shots=base_shots,
                seed=seed
            )
        except Exception as e:
            # Safe production fallback
            _log.warning(
                f"Mitigation technique '{tech}' failed with: {e!r}. "
                f"Falling back to '{fallback_technique}'."
            )
            tech = fallback_technique
            mitigated_value = apply_technique(
                name=tech,
                circuit=circuit,
                pauli=pauli,
                executor=raw_executor,
                backend_name=backend_name,
                shots=base_shots,
                seed=seed
            )

        return {
            "value": float(mitigated_value),
            "technique": tech,
            "abstained": abstained,
            "probabilities": recommendation["probabilities"],
            "features": recommendation["features"]
        }


def run(
    circuit: QuantumCircuit,
    pauli: str,
    backend_name: str,
    base_shots: int,
    model_path: str | Path = "results/boundary/model_significant.joblib",
    seed: int = 42,
    fallback_technique: str = "raw",
) -> float:
    """Convenience function to execute a mitigated circuit in a single line.

    Args:
        circuit: QuantumCircuit WITHOUT final measurements.
        pauli: Pauli string representing the observable (e.g. 'ZZZ').
        backend_name: Target backend name (e.g. 'FakeLagosV2').
        base_shots: The planned base shot budget.
        model_path: Path to the trained model bundle.
        seed: Seed for reproducibility of simulations.
        fallback_technique: Default technique if the model abstains or fails.

    Returns:
        The mitigated expectation value.
    """
    executor = MitigatedExecutor(model_path)
    res = executor.execute(
        circuit=circuit,
        pauli=pauli,
        backend_name=backend_name,
        base_shots=base_shots,
        seed=seed,
        fallback_technique=fallback_technique
    )
    return res["value"]

