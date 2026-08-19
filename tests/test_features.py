"""Unit tests for qemsel.features V2 (builder-features / B5).

Covers the additive ``version=2`` feature set (``FEATURE_NAMES_V2``) and pins
the frozen ``version=1`` surface byte-identically. Reference V1 dicts here were
CAPTURED from the pre-V2 code (capture-first regression rule) so any drift in
the shared V1 computation fails loudly.

backends.py is monkeypatched with the same contract-faithful fake as
test_features.py so these stay hermetic (known name -> info dict, unknown ->
ValueError).
"""

from __future__ import annotations

import math

import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

import qemsel.backends
from qemsel.features import (
    FEATURE_NAMES,
    FEATURE_NAMES_BY_VERSION,
    FEATURE_NAMES_V2,
    extract_features,
    convert_circuit_to_graph,
)

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


# --------------------------------------------------------------------------- #
# Circuits with hand-checkable structure.
# --------------------------------------------------------------------------- #
def _known_circuit() -> QuantumCircuit:
    """H(0); CX(0,1); T(1) — depth 3, 2q-layers 1, no rotation gates."""
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.t(1)
    return qc


def _mixed_circuit() -> QuantumCircuit:
    """3q circuit: depth 4, two disjoint CX layers, one rz(0.3) + one rx(pi/4).

    Hand values: n_2q_layers=2 (cx(0,1) then cx(1,2) share q1),
    entangling_density = 2/(3*4), mean_rz over {rz(0.3), rx(pi/4)}.
    """
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.rz(0.3, 2)
    qc.cx(1, 2)
    qc.rx(math.pi / 4, 0)
    qc.h(0)
    return qc


# Reference V1 outputs CAPTURED from the pre-V2 code (do not hand-edit values).
_V1_REFERENCE = {
    ("known", "FakeManilaV2"): {
        "n_qubits": 2.0,
        "depth": 3.0,
        "n_1q_gates": 2.0,
        "n_2q_gates": 1.0,
        "n_cnot": 1.0,
        "n_non_clifford": 1.0,
        "clifford_fraction": 0.6666666666666666,
        "depth_per_qubit": 1.5,
        "backend_avg_2q_error": 0.008,
        "backend_avg_readout_error": 0.025,
    },
    ("known", "FakeLagosV2"): {
        "n_qubits": 2.0,
        "depth": 3.0,
        "n_1q_gates": 2.0,
        "n_2q_gates": 1.0,
        "n_cnot": 1.0,
        "n_non_clifford": 1.0,
        "clifford_fraction": 0.6666666666666666,
        "depth_per_qubit": 1.5,
        "backend_avg_2q_error": 0.007,
        "backend_avg_readout_error": 0.15,
    },
    ("mixed", "FakeManilaV2"): {
        "n_qubits": 3.0,
        "depth": 4.0,
        "n_1q_gates": 4.0,
        "n_2q_gates": 2.0,
        "n_cnot": 2.0,
        "n_non_clifford": 2.0,
        "clifford_fraction": 0.6666666666666666,
        "depth_per_qubit": 1.3333333333333333,
        "backend_avg_2q_error": 0.008,
        "backend_avg_readout_error": 0.025,
    },
}

_CIRCUIT_BUILDERS = {"known": _known_circuit, "mixed": _mixed_circuit}


# --------------------------------------------------------------------------- #
# Constants / schema.
# --------------------------------------------------------------------------- #
class TestConstants:
    def test_v2_extends_v1_by_five(self) -> None:
        assert FEATURE_NAMES_V2 == FEATURE_NAMES + [
            "log2_shots",
            "n_2q_layers",
            "entangling_density",
            "mean_rz_angle_dist",
            "backend_avg_1q_error",
        ]
        assert len(FEATURE_NAMES_V2) == 15

    def test_v1_prefix_preserved(self) -> None:
        # The V1 block must remain the exact prefix of V2 (column-order stable).
        assert FEATURE_NAMES_V2[: len(FEATURE_NAMES)] == FEATURE_NAMES

    def test_by_version_map(self) -> None:
        assert FEATURE_NAMES_BY_VERSION == {1: FEATURE_NAMES, 2: FEATURE_NAMES_V2}


# --------------------------------------------------------------------------- #
# V1 regression (byte-identical, base_shots ignored).
# --------------------------------------------------------------------------- #
class TestV1Regression:
    @pytest.mark.parametrize(("label", "backend"), list(_V1_REFERENCE))
    def test_v1_matches_captured_reference(self, label: str, backend: str) -> None:
        feats = extract_features(_CIRCUIT_BUILDERS[label](), backend, version=1)
        assert list(feats) == FEATURE_NAMES
        assert feats == _V1_REFERENCE[(label, backend)]

    def test_default_version_is_one(self) -> None:
        # No version kwarg == version=1 exactly.
        default = extract_features(_known_circuit(), "FakeManilaV2")
        explicit = extract_features(_known_circuit(), "FakeManilaV2", version=1)
        assert default == explicit == _V1_REFERENCE[("known", "FakeManilaV2")]

    def test_v1_ignores_base_shots(self) -> None:
        without = extract_features(_mixed_circuit(), "FakeManilaV2", version=1)
        with_shots = extract_features(
            _mixed_circuit(), "FakeManilaV2", version=1, base_shots=999
        )
        assert without == with_shots

    def test_v2_first_ten_equal_v1(self) -> None:
        v1 = extract_features(_mixed_circuit(), "FakeLagosV2", version=1)
        v2 = extract_features(
            _mixed_circuit(), "FakeLagosV2", version=2, base_shots=4096
        )
        assert {k: v2[k] for k in FEATURE_NAMES} == v1


# --------------------------------------------------------------------------- #
# V2 additive feature values (hand-checked).
# --------------------------------------------------------------------------- #
class TestV2Values:
    def test_mixed_circuit_exact_values(self) -> None:
        feats = extract_features(
            _mixed_circuit(), "FakeManilaV2", version=2, base_shots=1024
        )
        assert list(feats) == FEATURE_NAMES_V2
        # V1 block unchanged.
        for k, v in _V1_REFERENCE[("mixed", "FakeManilaV2")].items():
            assert feats[k] == v
        # V2 additions, hand-computed.
        assert feats["log2_shots"] == 10.0  # log2(1024)
        assert feats["n_2q_layers"] == 2.0  # cx(0,1) then cx(1,2)
        assert feats["entangling_density"] == pytest.approx(2.0 / 12.0)
        d_rz = abs(math.remainder(0.3, math.pi / 2)) / (math.pi / 4)
        assert feats["mean_rz_angle_dist"] == pytest.approx((d_rz + 1.0) / 2.0)
        assert feats["backend_avg_1q_error"] == pytest.approx(3.0e-4)

    def test_backend_1q_error_follows_backend(self) -> None:
        feats = extract_features(
            _mixed_circuit(), "FakeLagosV2", version=2, base_shots=512
        )
        assert feats["backend_avg_1q_error"] == pytest.approx(2.0e-4)

    @pytest.mark.parametrize(
        ("shots", "expected"),
        [(256, 8.0), (1024, 10.0), (2048, 11.0), (16384, 14.0)],
    )
    def test_log2_shots(self, shots: int, expected: float) -> None:
        feats = extract_features(
            _known_circuit(), "FakeManilaV2", version=2, base_shots=shots
        )
        assert feats["log2_shots"] == pytest.approx(expected)

    def test_log2_shots_non_power_of_two_and_float(self) -> None:
        feats = extract_features(
            _known_circuit(), "FakeManilaV2", version=2, base_shots=3000
        )
        assert feats["log2_shots"] == pytest.approx(math.log2(3000))
        # float base_shots (e.g. a scaled budget) is accepted too.
        feats_f = extract_features(
            _known_circuit(), "FakeManilaV2", version=2, base_shots=1500.0
        )
        assert feats_f["log2_shots"] == pytest.approx(math.log2(1500.0))

    def test_n_2q_layers_no_two_qubit_gates(self) -> None:
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.rz(0.5, 1)
        qc.h(0)
        feats = extract_features(qc, "FakeManilaV2", version=2, base_shots=1024)
        assert feats["n_2q_layers"] == 0.0

    def test_n_2q_layers_serial_chain(self) -> None:
        # Three CX all sharing a qubit -> three distinct 2q layers.
        qc = QuantumCircuit(2)
        qc.cx(0, 1)
        qc.cx(0, 1)
        qc.cx(0, 1)
        feats = extract_features(qc, "FakeManilaV2", version=2, base_shots=1024)
        assert feats["n_2q_layers"] == 3.0

    def test_n_2q_layers_parallel(self) -> None:
        # Two disjoint CX in one layer -> single 2q layer.
        qc = QuantumCircuit(4)
        qc.cx(0, 1)
        qc.cx(2, 3)
        feats = extract_features(qc, "FakeManilaV2", version=2, base_shots=1024)
        assert feats["n_2q_layers"] == 1.0
        assert feats["n_2q_gates"] == 2.0

    def test_entangling_density_denominator_zero(self) -> None:
        # Empty circuit: depth 0 -> denominator 0 -> defined as 0.0.
        empty = QuantumCircuit(3)
        feats = extract_features(empty, "FakeManilaV2", version=2, base_shots=2048)
        assert feats["entangling_density"] == 0.0

    def test_entangling_density_value(self) -> None:
        qc = QuantumCircuit(2)
        qc.cx(0, 1)  # depth 1, n_2q 1, n_qubits 2 -> 1/2
        feats = extract_features(qc, "FakeManilaV2", version=2, base_shots=1024)
        assert feats["entangling_density"] == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
# mean_rz_angle_dist rules.
# --------------------------------------------------------------------------- #
class TestMeanRzAngleDist:
    def test_no_rotation_gates_is_zero(self) -> None:
        feats = extract_features(
            _known_circuit(), "FakeManilaV2", version=2, base_shots=1024
        )  # only h, cx, t -> no rx/ry/rz/p
        assert feats["mean_rz_angle_dist"] == 0.0

    def test_pi_half_multiple_contributes_zero(self) -> None:
        qc = QuantumCircuit(1)
        qc.rz(math.pi / 2, 0)  # Clifford rotation -> distance 0.0
        qc.rz(math.pi, 0)
        feats = extract_features(qc, "FakeManilaV2", version=2, base_shots=1024)
        assert feats["mean_rz_angle_dist"] == 0.0

    def test_pi_quarter_is_one(self) -> None:
        qc = QuantumCircuit(1)
        qc.rx(math.pi / 4, 0)  # exact midpoint -> distance 1.0
        feats = extract_features(qc, "FakeManilaV2", version=2, base_shots=1024)
        assert feats["mean_rz_angle_dist"] == pytest.approx(1.0)

    def test_unbound_parameter_is_one(self) -> None:
        theta = Parameter("theta")
        qc = QuantumCircuit(1)
        qc.rz(theta, 0)  # unbound -> conservatively 1.0
        feats = extract_features(qc, "FakeManilaV2", version=2, base_shots=1024)
        assert feats["mean_rz_angle_dist"] == pytest.approx(1.0)

    def test_unbound_and_clifford_average(self) -> None:
        theta = Parameter("theta")
        qc = QuantumCircuit(1)
        qc.rz(theta, 0)  # 1.0
        qc.rx(0.0, 0)  # Clifford -> 0.0
        feats = extract_features(qc, "FakeManilaV2", version=2, base_shots=256)
        assert feats["mean_rz_angle_dist"] == pytest.approx(0.5)

    def test_phase_gate_counted(self) -> None:
        # p (phase) gate is an angle-checked rotation; p(pi/4) -> distance 1.0.
        qc = QuantumCircuit(1)
        qc.p(math.pi / 4, 0)
        feats = extract_features(qc, "FakeManilaV2", version=2, base_shots=1024)
        assert feats["mean_rz_angle_dist"] == pytest.approx(1.0)

    def test_averages_over_all_rotations(self) -> None:
        qc = QuantumCircuit(1)
        qc.rz(0.0, 0)  # 0.0
        qc.ry(math.pi / 4, 0)  # 1.0
        qc.rx(math.pi / 4, 0)  # 1.0
        feats = extract_features(qc, "FakeManilaV2", version=2, base_shots=1024)
        assert feats["mean_rz_angle_dist"] == pytest.approx(2.0 / 3.0)


# --------------------------------------------------------------------------- #
# Empty / edge circuits.
# --------------------------------------------------------------------------- #
class TestEmptyCircuit:
    def test_all_v2_structure_zero(self) -> None:
        empty = QuantumCircuit(3)
        feats = extract_features(empty, "FakeManilaV2", version=2, base_shots=2048)
        assert feats["n_2q_layers"] == 0.0
        assert feats["entangling_density"] == 0.0
        assert feats["mean_rz_angle_dist"] == 0.0
        assert feats["log2_shots"] == pytest.approx(math.log2(2048))
        # V1 block still consistent.
        assert feats["clifford_fraction"] == 1.0
        assert feats["depth"] == 0.0


# --------------------------------------------------------------------------- #
# Output shape / types / determinism.
# --------------------------------------------------------------------------- #
class TestOutputShape:
    def test_key_order_exact(self) -> None:
        feats = extract_features(
            _mixed_circuit(), "FakeManilaV2", version=2, base_shots=1024
        )
        assert list(feats.keys()) == FEATURE_NAMES_V2

    def test_all_values_plain_floats(self) -> None:
        feats = extract_features(
            _mixed_circuit(), "FakeLagosV2", version=2, base_shots=1024
        )
        for name, value in feats.items():
            assert type(value) is float, f"{name} is {type(value)}"

    def test_deterministic(self) -> None:
        a = extract_features(
            _mixed_circuit(), "FakeManilaV2", version=2, base_shots=1024
        )
        b = extract_features(
            _mixed_circuit(), "FakeManilaV2", version=2, base_shots=1024
        )
        assert a == b

    def test_input_not_mutated(self) -> None:
        qc = _mixed_circuit()
        n_before = len(qc.data)
        extract_features(qc, "FakeManilaV2", version=2, base_shots=1024)
        assert len(qc.data) == n_before

    def test_works_with_conftest_fixtures(
        self, tiny_circuit: QuantumCircuit, tiny_identity_circuit: QuantumCircuit
    ) -> None:
        bell = extract_features(
            tiny_circuit, "FakeManilaV2", version=2, base_shots=1024
        )
        assert list(bell) == FEATURE_NAMES_V2
        assert bell["n_2q_layers"] == 1.0  # single cx
        assert bell["mean_rz_angle_dist"] == 0.0  # h + cx only
        ident = extract_features(
            tiny_identity_circuit, "FakeLagosV2", version=2, base_shots=512
        )
        assert ident["n_2q_layers"] == 4.0  # four serial cx sharing qubits


# --------------------------------------------------------------------------- #
# Errors / validation.
# --------------------------------------------------------------------------- #
class TestErrors:
    @pytest.mark.parametrize("bad", [None, 0, -5, -1.0])
    def test_version2_requires_positive_base_shots(self, bad: object) -> None:
        with pytest.raises(ValueError, match="base_shots"):
            extract_features(
                _known_circuit(), "FakeManilaV2", version=2, base_shots=bad
            )

    def test_unknown_version_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown feature version"):
            extract_features(_known_circuit(), "FakeManilaV2", version=3)

    def test_version2_measurement_raises(self) -> None:
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()
        with pytest.raises(ValueError, match="measure"):
            extract_features(qc, "FakeManilaV2", version=2, base_shots=1024)

    def test_version2_unknown_backend_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown backend"):
            extract_features(
                _known_circuit(), "NoSuchBackend", version=2, base_shots=1024
            )

    def test_version2_missing_base_shots_before_backend_lookup(self) -> None:
        # base_shots validation should fire even with an otherwise-bad backend.
        with pytest.raises(ValueError, match="base_shots"):
            extract_features(
                _known_circuit(), "NoSuchBackend", version=2, base_shots=None
            )


class TestConvertCircuitToGraph:
    def test_convert_simple_circuit(self) -> None:
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)

        graph = convert_circuit_to_graph(qc)

        assert "nodes" in graph
        assert "edge_index" in graph

        nodes = graph["nodes"]
        edge_index = graph["edge_index"]

        # There should be exactly two operation nodes (h and cx)
        assert len(nodes) == 2
        assert nodes[0]["op"] == "h"
        assert nodes[1]["op"] == "cx"

        # Edge index should represent the dependency flow from gate 0 to gate 1
        assert len(edge_index) >= 1
