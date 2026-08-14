"""CLI: recommend the best QEM technique for a circuit.

Usage (from the project root, with the project venv):

    python scripts/recommend.py --model results/run1/model.joblib \
        --backend FakeManilaV2 --qasm my_circuit.qasm

    python scripts/recommend.py --model results/run1/model.joblib \
        --backend FakeLagosV2 --demo ghz_plus --qubits 3 --depth 4 --seed 0

Exactly one of --qasm / --demo must be given. --qasm loads an OpenQASM 2.0
file (final measurements, if any, are stripped — the qemsel convention is
measurement-free circuits). --demo builds a fresh benchmark circuit from one
of the qemsel.circuits families.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from qiskit import QuantumCircuit


def _load_qasm_circuit(path: Path) -> QuantumCircuit:
    """Load an OpenQASM 2.0 file and strip any final measurements."""
    from qiskit import qasm2

    if not path.exists():
        raise SystemExit(f"error: QASM file not found: {path}")
    circuit = qasm2.load(str(path))
    circuit.remove_final_measurements(inplace=True)
    return circuit


def _build_demo_circuit(
    family: str, n_qubits: int, depth: int, seed: int
) -> QuantumCircuit:
    """Build a demo circuit from a qemsel.circuits family generator."""
    from qemsel.circuits import FAMILIES

    if family not in FAMILIES:
        raise SystemExit(
            f"error: unknown demo family {family!r}; choose one of "
            f"{sorted(FAMILIES.keys())}"
        )
    return FAMILIES[family](n_qubits, depth, seed)


def main(argv: list[str] | None = None) -> int:
    """Parse args, run the recommender, print the result as JSON."""
    parser = argparse.ArgumentParser(
        description="Recommend the best QEM technique for a circuit."
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="path to model.joblib bundle saved by scripts/train_model.py",
    )
    parser.add_argument(
        "--backend",
        type=str,
        required=True,
        help="backend name, e.g. FakeManilaV2 (see qemsel.backends.BACKENDS)",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--qasm", type=Path, help="OpenQASM 2.0 file with the circuit"
    )
    source.add_argument(
        "--demo",
        type=str,
        help="build a demo circuit from this qemsel.circuits family "
        "(e.g. layered_random, ghz_plus, mirror_circuit)",
    )
    parser.add_argument(
        "--qubits", type=int, default=3, help="demo circuit qubits (default 3)"
    )
    parser.add_argument(
        "--depth", type=int, default=4, help="demo circuit depth (default 4)"
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="demo circuit seed (default 0)"
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=None,
        help="planned shot budget (base_shots). REQUIRED for a model bundle "
        "trained with feature_version 2 (its features include log2_shots); "
        "ignored for V1 bundles",
    )
    args = parser.parse_args(argv)

    if args.qasm is not None:
        circuit = _load_qasm_circuit(args.qasm)
    else:
        circuit = _build_demo_circuit(args.demo, args.qubits, args.depth, args.seed)

    from qemsel.recommend import recommend

    result = recommend(args.model, circuit, args.backend, base_shots=args.shots)
    print(json.dumps(result, indent=2, sort_keys=False))
    # Abstain path: the model has no confident recommendation (V2 bundle with
    # an abstain_threshold the top probability did not clear). Signal it with
    # a distinct message and a non-zero exit code so a wrapper can branch on
    # it (falling back to a safe default such as 'raw', or a human).
    if result["technique"] == "abstain":
        top = max(result["probabilities"].values()) if result["probabilities"] else 0.0
        print(
            f"\nNo confident recommendation for this circuit on {args.backend}: "
            f"the model ABSTAINED (top probability {top:.3g} < abstain "
            f"threshold {result.get('abstain_threshold')}). Fall back to a "
            "safe default (e.g. 'raw') or escalate to a human."
        )
        return 2
    print(
        f"\nRecommended technique for this circuit on {args.backend}: "
        f"{result['technique']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
