"""Unit tests for qemsel.features.

backends.py may still be a stub when these run, so every test monkeypatches
``qemsel.backends.get_backend_info`` with a deterministic fake that mirrors
the real contract (known-name -> info dict, unknown -> ValueError). The
integrator wires the live implementation; these tests stay hermetic.
"""

from __future__ import annotations

import math

import pytest
from qiskit import QuantumCircuit

import qemsel.backends
from qemsel.features import FEATURE_NAMES, extract_features

_FAKE_INFO = {
    "FakeManilaV2": {
        "name": "FakeManilaV2",
        "n_qubits": 5,
        "avg_1q_error": 3.0e-4,
        "avg_2q_error": 8.0e-3,
        "avg_readout_error": 0.025,
        "max_readout_error": 0.04,
    },
    "FakeLagosV2": {
        "name": "FakeLagosV2",
        "n_qubits": 7,
        "avg_1q_error": 2.0e-4,
        "avg_2q_error": 7.0e-3,
        "avg_readout_error": 0.15,
        "max_readout_error": 0.27,
    },
}


@pytest.fixture(autouse=True)
def patched_backend_info(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Replace get_backend_info with a contract-faithful deterministic fake."""

    def _fake(name: str) -> dict:
        if name not in _FAKE_INFO:
            raise ValueError(f"unknown backend: {name!r}")
        return _FAKE_INFO[name]

    monkeypatch.setattr(qemsel.backends, "get_backend_info", _fake)
    return _FAKE_INFO


def _known_circuit() -> QuantumCircuit:
    """H(0); CX(0,1); T(1) — hand-checkable counts, depth 3."""
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.t(1)
    return qc


class TestKnownCircuitCounts:
    def test_exact_expected_values(self) -> None:
        feats = extract_features(_known_circuit(), "FakeManilaV2")
        assert feats["n_qubits"] == 2.0
        assert feats["depth"] == 3.0
        assert feats["n_1q_gates"] == 2.0  # h, t
        assert feats["n_2q_gates"] == 1.0  # cx
        assert feats["n_cnot"] == 1.0
        assert feats["n_non_clifford"] == 1.0  # t only
        assert feats["clifford_fraction"] == pytest.approx(2.0 / 3.0)
        assert feats["depth_per_qubit"] == pytest.approx(1.5)
        assert feats["backend_avg_2q_error"] == pytest.approx(8.0e-3)
        assert feats["backend_avg_readout_error"] == pytest.approx(0.025)

    def test_backend_features_follow_backend_name(self) -> None:
        feats = extract_features(_known_circuit(), "FakeLagosV2")
        assert feats["backend_avg_2q_error"] == pytest.approx(7.0e-3)
        assert feats["backend_avg_readout_error"] == pytest.approx(0.15)

    def test_barriers_excluded_from_counts(self) -> None:
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.barrier()
        qc.cx(0, 1)
        feats = extract_features(qc, "FakeManilaV2")
        assert feats["n_1q_gates"] == 1.0
        assert feats["n_2q_gates"] == 1.0
        assert feats["n_non_clifford"] == 0.0
        assert feats["clifford_fraction"] == 1.0

    def test_input_circuit_not_mutated(self) -> None:
        qc = _known_circuit()
        n_ops_before = len(qc.data)
        extract_features(qc, "FakeManilaV2")
        assert len(qc.data) == n_ops_before


class TestCliffordAngleRule:
    @pytest.mark.parametrize(
        "angle",
        [0.0, math.pi / 2, math.pi, -math.pi, 3 * math.pi / 2, -7 * math.pi / 2,
         math.pi / 2 + 5e-10],  # within the 1e-9 tolerance
    )
    def test_pi_half_multiples_are_clifford(self, angle: float) -> None:
        qc = QuantumCircuit(1)
        qc.rz(angle, 0)
        feats = extract_features(qc, "FakeManilaV2")
        assert feats["n_non_clifford"] == 0.0

    @pytest.mark.parametrize(
        "angle", [math.pi / 4, 0.3, math.pi / 2 + 1e-6, 1.0]
    )
    def test_other_angles_are_non_clifford(self, angle: float) -> None:
        qc = QuantumCircuit(1)
        qc.rx(angle, 0)
        feats = extract_features(qc, "FakeManilaV2")
        assert feats["n_non_clifford"] == 1.0

    def test_t_and_tdg_are_non_clifford(self) -> None:
        qc = QuantumCircuit(1)
        qc.t(0)
        qc.tdg(0)
        feats = extract_features(qc, "FakeManilaV2")
        assert feats["n_non_clifford"] == 2.0
        assert feats["clifford_fraction"] == 0.0

    def test_mixed_rotations_counted_per_gate(self) -> None:
        qc = QuantumCircuit(2)
        qc.ry(math.pi, 0)       # Clifford
        qc.rz(math.pi / 4, 1)   # non-Clifford
        qc.rx(0.0, 0)           # Clifford
        feats = extract_features(qc, "FakeManilaV2")
        assert feats["n_non_clifford"] == 1.0
        assert feats["clifford_fraction"] == pytest.approx(2.0 / 3.0)


class TestMirrorCircuit:
    def test_non_clifford_count_is_even(self) -> None:
        """U then U-dagger: every non-Clifford gate appears with its inverse."""
        u = QuantumCircuit(2)
        u.rz(0.3, 0)
        u.ry(0.7, 1)
        u.cx(0, 1)
        u.t(0)
        mirror = u.compose(u.inverse())
        feats = extract_features(mirror, "FakeManilaV2")
        assert feats["n_non_clifford"] == 6.0  # 3 in U + 3 inverses
        assert int(feats["n_non_clifford"]) % 2 == 0
        assert feats["n_2q_gates"] == 2.0
        assert feats["n_cnot"] == 2.0


class TestOutputShape:
    def test_key_order_stable_and_exact(self) -> None:
        f1 = extract_features(_known_circuit(), "FakeManilaV2")
        f2 = extract_features(_known_circuit(), "FakeManilaV2")
        assert list(f1.keys()) == FEATURE_NAMES
        assert list(f2.keys()) == FEATURE_NAMES
        assert f1 == f2  # deterministic

    def test_all_values_plain_floats(self) -> None:
        feats = extract_features(_known_circuit(), "FakeLagosV2")
        for name, value in feats.items():
            assert type(value) is float, f"{name} is {type(value)}"

    def test_feature_names_constant(self) -> None:
        assert FEATURE_NAMES == [
            "n_qubits",
            "depth",
            "n_1q_gates",
            "n_2q_gates",
            "n_cnot",
            "n_non_clifford",
            "clifford_fraction",
            "depth_per_qubit",
            "backend_avg_2q_error",
            "backend_avg_readout_error",
        ]

    def test_empty_circuit_clifford_fraction_one(self) -> None:
        qc = QuantumCircuit(3)
        feats = extract_features(qc, "FakeManilaV2")
        assert feats["clifford_fraction"] == 1.0
        assert feats["n_1q_gates"] == 0.0
        assert feats["n_2q_gates"] == 0.0
        assert feats["depth"] == 0.0

    def test_works_with_conftest_fixtures(
        self, tiny_circuit: QuantumCircuit, tiny_identity_circuit: QuantumCircuit
    ) -> None:
        bell = extract_features(tiny_circuit, "FakeManilaV2")
        assert bell["n_1q_gates"] == 1.0  # h
        assert bell["n_cnot"] == 1.0
        assert bell["n_non_clifford"] == 0.0
        ident = extract_features(tiny_identity_circuit, "FakeLagosV2")
        assert ident["n_qubits"] == 3.0
        assert ident["n_cnot"] == 4.0
        assert ident["clifford_fraction"] == 1.0


class TestErrors:
    def test_measurement_raises(self) -> None:
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.measure(0, 0)
        with pytest.raises(ValueError, match="measure"):
            extract_features(qc, "FakeManilaV2")

    def test_measure_all_raises(self) -> None:
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()
        with pytest.raises(ValueError):
            extract_features(qc, "FakeManilaV2")

    def test_unknown_backend_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown backend"):
            extract_features(_known_circuit(), "NoSuchBackend")
