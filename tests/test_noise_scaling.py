"""Tests for noise-scaled backend variants "<FakeName>@x<scale>" (backends.py).

Covers both upward (scale >= 1.0) and downward/sub-unity (0 < scale < 1.0) noise dials.
"""

from __future__ import annotations

import math
import pytest
from qiskit import QuantumCircuit

from qemsel.backends import (
    BACKENDS,
    LOW_NOISE_SCALES,
    _SCALED_GATE_ERROR_CAP,
    _SCALED_READOUT_ERROR_CAP,
    get_backend_info,
    make_executor,
    parse_backend_name,
)

#: Hardcoded PRE-CHANGE executor value (captured on the code before noise
#: scaling existed): make_executor("FakeManilaV2", shots=256, seed=7) on the
#: Bell circuit h(0); cx(0,1), pauli "ZZ". 222/256 — exactly representable.
_PRE_CHANGE_BELL_ZZ = 0.8671875

#: Sub-unity determinism anchors: SAME (256 shots, seed 7) Bell <ZZ>, captured
#: on the current code at scales 0.5 and 0.25. 240/256 and 248/256.
_SUBUNITY_BELL_ZZ_X0_5 = 0.9375
_SUBUNITY_BELL_ZZ_X0_25 = 0.96875

_INFO_KEYS = [
    "name",
    "n_qubits",
    "avg_1q_error",
    "avg_2q_error",
    "avg_readout_error",
    "max_readout_error",
]
_SCALING_INFO_KEYS = (
    "avg_1q_error",
    "avg_2q_error",
    "avg_readout_error",
    "max_readout_error",
)


def _bell() -> QuantumCircuit:
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    return qc


def _deep_cx_error(backend_name: str, shots: int = 8000, seed: int = 0) -> float:
    """|raw error| of <ZZ> on a FIXED 10-CNOT circuit (ideal <ZZ> = +1)."""
    qc = QuantumCircuit(2)
    qc.x(0)
    qc.x(1)
    for _ in range(10):
        qc.cx(0, 1)
    executor = make_executor(backend_name, shots=shots, seed=seed)
    return abs(1.0 - executor(qc, "ZZ"))


def _readout_only_error(backend_name: str, shots: int = 8000, seed: int = 5) -> float:
    """|raw error| of <ZZ> on |11> (two x gates only) — a readout-dominated probe."""
    qc = QuantumCircuit(2)
    qc.x(0)
    qc.x(1)
    executor = make_executor(backend_name, shots=shots, seed=seed)
    return abs(1.0 - executor(qc, "ZZ"))


# ---------------------------------------------------------------------------
# 1. Name parsing
# ---------------------------------------------------------------------------

class TestParseBackendName:
    def test_plain_name_is_scale_one(self):
        assert parse_backend_name("FakeManilaV2") == ("FakeManilaV2", 1.0)

    def test_plain_ibm_name_passes_through(self):
        assert parse_backend_name("ibm_brisbane") == ("ibm_brisbane", 1.0)

    @pytest.mark.parametrize(
        "name,base,scale",
        [
            ("FakeManilaV2@x1.5", "FakeManilaV2", 1.5),
            ("FakeLagosV2@x2.0", "FakeLagosV2", 2.0),
            ("FakeJakartaV2@x3", "FakeJakartaV2", 3.0),
            ("FakeSherbrooke@x0.5", "FakeSherbrooke", 0.5),
            ("FakeManilaV2@x1.0", "FakeManilaV2", 1.0),
            ("FakeManilaV2@x0.25", "FakeManilaV2", 0.25),
            ("FakeManilaV2@x.5", "FakeManilaV2", 0.5),
        ],
    )
    def test_suffix_parsed(self, name, base, scale):
        assert parse_backend_name(name) == (base, scale)

    @pytest.mark.parametrize(
        "bad",
        [
            "FakeManilaV2@1.5",  # missing 'x'
            "FakeManilaV2@x",  # empty scale
            "FakeManilaV2@xfoo",  # not a number
            "FakeManilaV2@y2.0",  # wrong marker letter
            "FakeManilaV2@x0",  # zero scale
            "FakeManilaV2@x0.0",
            "FakeManilaV2@x-2",  # negative
            "FakeManilaV2@x-0.25",
            "FakeManilaV2@x2@x3",  # more than one '@'
            "FakeManilaV2@xinf",  # non-finite
            "FakeManilaV2@xnan",
            "@x2.0",  # empty base
            "ibm_brisbane@x2.0",  # NEVER scale real hardware
            "ibm_brisbane@x1.0",  # not even at scale 1
        ],
    )
    def test_bad_suffix_raises(self, bad):
        with pytest.raises(ValueError):
            parse_backend_name(bad)

    def test_bad_suffix_raises_through_get_backend_info(self):
        with pytest.raises(ValueError):
            get_backend_info("FakeManilaV2@1.5")

    def test_bad_suffix_raises_through_make_executor(self):
        with pytest.raises(ValueError):
            make_executor("FakeManilaV2@x0", shots=16, seed=0)

    def test_unknown_base_with_valid_suffix_raises(self):
        with pytest.raises(ValueError, match="unknown backend"):
            get_backend_info("FakeNopeV2@x1.5")
        with pytest.raises(ValueError, match="unknown backend"):
            make_executor("FakeNopeV2@x1.5", shots=16, seed=0)

    def test_backends_list_holds_plain_names_only(self):
        assert BACKENDS == [
            "FakeManilaV2",
            "FakeJakartaV2",
            "FakeLagosV2",
            "FakeSherbrooke",
        ]
        assert not any("@" in name for name in BACKENDS)


# ---------------------------------------------------------------------------
# 2. get_backend_info scaling
# ---------------------------------------------------------------------------

class TestScaledBackendInfo:
    def test_ratio_is_exactly_the_scale_when_no_cap_engages(self):
        base = get_backend_info("FakeManilaV2")
        scaled = get_backend_info("FakeManilaV2@x1.5")
        for key in _SCALING_INFO_KEYS:
            assert scaled[key] == pytest.approx(1.5 * base[key], rel=1e-12), key

    @pytest.mark.parametrize("scale", LOW_NOISE_SCALES)
    def test_ratio_is_exactly_the_scale_subunity(self, scale):
        base = get_backend_info("FakeManilaV2")
        scaled = get_backend_info(f"FakeManilaV2@x{scale}")
        for key in _SCALING_INFO_KEYS:
            assert scaled[key] == pytest.approx(scale * base[key], rel=1e-12), key

    def test_schema_and_name_echo(self):
        scaled = get_backend_info("FakeManilaV2@x1.5")
        assert sorted(scaled) == sorted(_INFO_KEYS)
        assert scaled["name"] == "FakeManilaV2@x1.5"
        assert scaled["n_qubits"] == get_backend_info("FakeManilaV2")["n_qubits"]

    def test_x1_suffix_returns_plain_numbers_verbatim(self):
        base = get_backend_info("FakeLagosV2")
        suffixed = get_backend_info("FakeLagosV2@x1.0")
        assert suffixed["name"] == "FakeLagosV2@x1.0"
        for key in _INFO_KEYS[1:]:
            assert suffixed[key] == base[key], key
        assert base["max_readout_error"] > 0.45

    @pytest.mark.parametrize("scale", LOW_NOISE_SCALES)
    def test_caps_never_bind_below_unity_on_saturated_lagos(self, scale):
        base = get_backend_info("FakeLagosV2")
        assert base["max_readout_error"] > _SCALED_READOUT_ERROR_CAP
        scaled = get_backend_info(f"FakeLagosV2@x{scale}")
        for key in _SCALING_INFO_KEYS:
            assert scaled[key] == pytest.approx(scale * base[key], rel=1e-12), key
        assert scaled["max_readout_error"] < _SCALED_READOUT_ERROR_CAP

    def test_readout_cap_engages(self):
        scaled = get_backend_info("FakeLagosV2@x50")
        assert scaled["max_readout_error"] == pytest.approx(0.45)
        assert scaled["avg_readout_error"] == pytest.approx(0.45)

    def test_gate_cap_engages(self):
        scaled = get_backend_info("FakeManilaV2@x100000")
        assert scaled["avg_2q_error"] == pytest.approx(0.9)
        assert scaled["avg_1q_error"] <= 0.9
        assert scaled["avg_readout_error"] == pytest.approx(0.45)

    def test_cached_copy_is_safe_to_mutate(self):
        first = get_backend_info("FakeManilaV2@x1.5")
        first["avg_2q_error"] = 999.0
        assert get_backend_info("FakeManilaV2@x1.5")["avg_2q_error"] != 999.0

    def test_scaled_numbers_flow_through_extract_features(self):
        from qemsel.features import extract_features
        qc = _bell()
        base = extract_features(qc, "FakeManilaV2")
        scaled = extract_features(qc, "FakeManilaV2@x2.0")
        assert scaled["backend_avg_2q_error"] == pytest.approx(
            2.0 * base["backend_avg_2q_error"], rel=1e-12
        )
        assert scaled["backend_avg_readout_error"] == pytest.approx(
            2.0 * base["backend_avg_readout_error"], rel=1e-12
        )


# ---------------------------------------------------------------------------
# 3. Plain-name regression
# ---------------------------------------------------------------------------

class TestPlainNameRegression:
    def test_plain_name_matches_prechange_value_exactly(self):
        value = make_executor("FakeManilaV2", shots=256, seed=7)(_bell(), "ZZ")
        assert value == _PRE_CHANGE_BELL_ZZ

    def test_x1_suffix_matches_prechange_value_exactly(self):
        value = make_executor("FakeManilaV2@x1.0", shots=256, seed=7)(_bell(), "ZZ")
        assert value == _PRE_CHANGE_BELL_ZZ


# ---------------------------------------------------------------------------
# 4. Scaled executor: determinism, readout scaling, monotone noise dial
# ---------------------------------------------------------------------------

class TestScaledExecutor:
    def test_deterministic_within_and_across_builds(self):
        ex_a = make_executor("FakeManilaV2@x2.0", shots=512, seed=11)
        ex_b = make_executor("FakeManilaV2@x2.0", shots=512, seed=11)
        qc = _bell()
        first = ex_a(qc, "ZZ")
        assert ex_a(qc, "ZZ") == first
        assert ex_b(qc, "ZZ") == first

    def test_subunity_deterministic_bell_anchors(self):
        val_05 = make_executor("FakeManilaV2@x0.5", shots=256, seed=7)(_bell(), "ZZ")
        assert val_05 == _SUBUNITY_BELL_ZZ_X0_5
        val_025 = make_executor("FakeManilaV2@x0.25", shots=256, seed=7)(_bell(), "ZZ")
        assert val_025 == _SUBUNITY_BELL_ZZ_X0_25

    def test_readout_scaling_fires(self):
        qc = QuantumCircuit(2)
        qc.x(0)
        qc.x(1)
        plain = make_executor("FakeManilaV2", shots=8000, seed=5)
        scaled = make_executor("FakeManilaV2@x2.0", shots=8000, seed=5)
        err_x1 = abs(1.0 - plain(qc, "ZZ"))
        err_x2 = abs(1.0 - scaled(qc, "ZZ"))
        assert err_x2 > err_x1 + 0.05

    def test_subunity_is_cleaner_than_plain(self):
        err_x1 = _readout_only_error("FakeLagosV2@x1.0")
        err_x025 = _readout_only_error("FakeLagosV2@x0.25")
        assert err_x025 < err_x1 - 0.1

    def test_monotone_x2_noisier_than_x1(self):
        err_x1 = _deep_cx_error("FakeManilaV2")
        err_x2 = _deep_cx_error("FakeManilaV2@x2.0")
        assert err_x2 > err_x1 + 0.05

    def test_monotone_x3_noisier_than_x2(self):
        err_x2 = _deep_cx_error("FakeManilaV2@x2.0")
        err_x3 = _deep_cx_error("FakeManilaV2@x3.0")
        assert err_x3 > err_x2 + 0.05

    def test_downward_dial_monotone_lagos_readout(self):
        err_025 = _readout_only_error("FakeLagosV2@x0.25")
        err_05 = _readout_only_error("FakeLagosV2@x0.5")
        err_10 = _readout_only_error("FakeLagosV2@x1.0")
        assert err_05 > err_025 + 0.05
        assert err_10 > err_05 + 0.05

    def test_monotone_across_unity_both_directions(self):
        err_05 = _deep_cx_error("FakeManilaV2@x0.5")
        err_10 = _deep_cx_error("FakeManilaV2")
        err_15 = _deep_cx_error("FakeManilaV2@x1.5")
        assert err_10 > err_05 + 0.05
        assert err_15 > err_10 + 0.05

    def test_scaled_path_still_routes_noise_on_non_device_edges(self):
        executor = make_executor("FakeLagosV2@x2.0", shots=2000, seed=0)
        pauli = "IIIZZ"
        ref = QuantumCircuit(5)
        ref.x(3)
        ref.x(4)
        deep = ref.copy()
        for _ in range(40):
            deep.cx(3, 4)
        ref_val = executor(ref, pauli)
        deep_val = executor(deep, pauli)
        assert ref_val > 0.5
        assert ref_val - deep_val > 0.3

    def test_input_circuit_not_mutated_on_scaled_path(self):
        qc = _bell()
        n_ops_before = len(qc.data)
        make_executor("FakeManilaV2@x1.5", shots=128, seed=3)(qc, "ZZ")
        assert len(qc.data) == n_ops_before
        assert qc.num_clbits == 0
