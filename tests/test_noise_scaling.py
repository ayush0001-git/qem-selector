"""Tests for noise-scaled backend variants "<FakeName>@x<scale>" (backends.py).

Four contracts under test:

1. Name parsing: the "@x<scale>" grammar, with ValueError on every malformed
   suffix (including scale suffixes on ibm_* hardware names).
2. get_backend_info on a scaled name returns the SCALED per-entry averages
   (correct ratio when no cap engages; capped values when one does), and the
   scaled numbers flow through features.extract_features automatically.
3. Plain-name REGRESSION: plain names (and "@x1.0") must be byte-identical
   to the pre-change behavior. Anchored by a hardcoded executor value
   captured by running the PRE-CHANGE code: FakeManilaV2, 256 shots, seed 7,
   Bell <ZZ> = 0.8671875 (= 222/256 exactly).
4. Physics + determinism of the scaled executor: readout scaling fires, the
   noise dial is monotone (|raw error| at x2.0 > x1.0 > generous margins on
   a fixed 10-CNOT circuit), device routing still applies noise on
   non-device-edge pairs, and same (name, shots, seed) -> identical results.
"""

from __future__ import annotations

import pytest
from qiskit import QuantumCircuit

from qemsel.backends import (
    BACKENDS,
    get_backend_info,
    make_executor,
    parse_backend_name,
)

#: Hardcoded PRE-CHANGE executor value (captured on the code before noise
#: scaling existed): make_executor("FakeManilaV2", shots=256, seed=7) on the
#: Bell circuit h(0); cx(0,1), pauli "ZZ". 222/256 — exactly representable,
#: so the comparison below is EXACT equality, not approx.
_PRE_CHANGE_BELL_ZZ = 0.8671875


def _bell() -> QuantumCircuit:
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    return qc


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

_INFO_KEYS = [
    "name",
    "n_qubits",
    "avg_1q_error",
    "avg_2q_error",
    "avg_readout_error",
    "max_readout_error",
]


class TestScaledBackendInfo:
    def test_ratio_is_exactly_the_scale_when_no_cap_engages(self):
        # FakeManilaV2 errors are all far below the caps at x1.5 (worst
        # readout 9.6% -> 14.5%), so every average must scale by EXACTLY 1.5.
        base = get_backend_info("FakeManilaV2")
        scaled = get_backend_info("FakeManilaV2@x1.5")
        for key in (
            "avg_1q_error",
            "avg_2q_error",
            "avg_readout_error",
            "max_readout_error",
        ):
            assert scaled[key] == pytest.approx(1.5 * base[key], rel=1e-12), key

    def test_schema_and_name_echo(self):
        scaled = get_backend_info("FakeManilaV2@x1.5")
        assert sorted(scaled) == sorted(_INFO_KEYS)
        assert scaled["name"] == "FakeManilaV2@x1.5"
        assert scaled["n_qubits"] == get_backend_info("FakeManilaV2")["n_qubits"]

    def test_x1_suffix_returns_plain_numbers_verbatim(self):
        # Scale 1.0 must NOT apply the caps (FakeLagosV2 q2 stores 46.4%
        # readout error, above the 0.45 cap) — plain behavior verbatim.
        base = get_backend_info("FakeLagosV2")
        suffixed = get_backend_info("FakeLagosV2@x1.0")
        assert suffixed["name"] == "FakeLagosV2@x1.0"
        for key in _INFO_KEYS[1:]:
            assert suffixed[key] == base[key], key  # exact, not approx
        assert base["max_readout_error"] > 0.45  # the cap would have cut this

    def test_readout_cap_engages(self):
        # At x50 every FakeLagosV2 per-qubit readout error saturates 0.45.
        scaled = get_backend_info("FakeLagosV2@x50")
        assert scaled["max_readout_error"] == pytest.approx(0.45)
        assert scaled["avg_readout_error"] == pytest.approx(0.45)

    def test_gate_cap_engages(self):
        # At an absurd scale every nonzero 2q entry saturates the 0.9 cap.
        # (avg_1q may stay below 0.9: virtual-rz entries with calibrated
        # error 0.0 scale to 0.0 and stay in the average, as in the plain
        # path.)
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
        # Circuit-side features must be identical — only the backend
        # features change with the scale suffix.
        for key in (
            "n_qubits",
            "depth",
            "n_1q_gates",
            "n_2q_gates",
            "n_cnot",
            "n_non_clifford",
            "clifford_fraction",
            "depth_per_qubit",
        ):
            assert scaled[key] == base[key], key


# ---------------------------------------------------------------------------
# 3. Plain-name regression (byte-identical to the pre-change code)
# ---------------------------------------------------------------------------


class TestPlainNameRegression:
    def test_plain_name_matches_hardcoded_prechange_value_exactly(self):
        value = make_executor("FakeManilaV2", shots=256, seed=7)(_bell(), "ZZ")
        assert value == _PRE_CHANGE_BELL_ZZ  # EXACT — any drift is a break

    def test_x1_suffix_matches_hardcoded_prechange_value_exactly(self):
        value = make_executor("FakeManilaV2@x1.0", shots=256, seed=7)(
            _bell(), "ZZ"
        )
        assert value == _PRE_CHANGE_BELL_ZZ


# ---------------------------------------------------------------------------
# 4. Scaled executor: determinism, readout scaling, monotone noise dial
# ---------------------------------------------------------------------------


def _deep_cx_error(backend_name: str, shots: int = 8000, seed: int = 0) -> float:
    """|raw error| of <ZZ> on a FIXED 10-CNOT circuit (ideal <ZZ> = +1).

    Prepare |11> then apply 10 cx(0,1) (even count -> identity on the
    state), measure ZZ. Any deviation from +1 is noise.
    """
    qc = QuantumCircuit(2)
    qc.x(0)
    qc.x(1)
    for _ in range(10):
        qc.cx(0, 1)
    executor = make_executor(backend_name, shots=shots, seed=seed)
    return abs(1.0 - executor(qc, "ZZ"))


class TestScaledExecutor:
    def test_deterministic_within_and_across_builds(self):
        ex_a = make_executor("FakeManilaV2@x2.0", shots=512, seed=11)
        ex_b = make_executor("FakeManilaV2@x2.0", shots=512, seed=11)
        qc = _bell()
        first = ex_a(qc, "ZZ")
        assert ex_a(qc, "ZZ") == first  # same executor, twice
        assert ex_b(qc, "ZZ") == first  # freshly built executor

    def test_readout_scaling_fires(self):
        # Measure-only workload: prepare |11> (two x gates, negligible gate
        # error) and read ZZ. Manila readout ~3.5%/2.2% on q0/q1 gives
        # error ~0.11 at x1 and ~0.23 at x2 — far beyond the ~0.008 shot
        # sd at 8000 shots.
        qc = QuantumCircuit(2)
        qc.x(0)
        qc.x(1)
        plain = make_executor("FakeManilaV2", shots=8000, seed=5)
        scaled = make_executor("FakeManilaV2@x2.0", shots=8000, seed=5)
        err_x1 = abs(1.0 - plain(qc, "ZZ"))
        err_x2 = abs(1.0 - scaled(qc, "ZZ"))
        assert err_x2 > err_x1 + 0.05, (
            f"readout scaling did not fire: |err| x1={err_x1:.4f} "
            f"x2={err_x2:.4f}"
        )

    def test_monotone_x2_noisier_than_x1(self):
        # Fixed 10-CNOT circuit; probe values while building: x1 |err|
        # ~0.21, x2 ~0.36. Margin 0.05 >> shot sd (~0.011 at 8000 shots).
        err_x1 = _deep_cx_error("FakeManilaV2")
        err_x2 = _deep_cx_error("FakeManilaV2@x2.0")
        assert err_x2 > err_x1 + 0.05, (
            f"noise dial not monotone: |err| x1={err_x1:.4f} x2={err_x2:.4f}"
        )

    def test_monotone_x3_noisier_than_x2(self):
        # Both on the scaled path (clean apples-to-apples): probe x2 ~0.36,
        # x3 ~0.49.
        err_x2 = _deep_cx_error("FakeManilaV2@x2.0")
        err_x3 = _deep_cx_error("FakeManilaV2@x3.0")
        assert err_x3 > err_x2 + 0.05, (
            f"noise dial not monotone: |err| x2={err_x2:.4f} x3={err_x3:.4f}"
        )

    def test_scaled_path_still_routes_noise_on_non_device_edges(self):
        # Counterpart of the plain-path regression test (review 2026-07-21):
        # (3,4) is NOT a FakeLagosV2 edge; the scaled path must keep the
        # from_backend coupling map so routed 2q gates still pick up
        # (scaled) noise. A 40-cx chain from |11> must decay hard at x2.
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
        assert ref_val > 0.5, f"reference state already broken: {ref_val}"
        assert ref_val - deep_val > 0.3, (
            f"deep cx chain on scaled Lagos pair (3,4) decayed only "
            f"{ref_val - deep_val:.4f} — scaled noise model lost the "
            "device routing/coverage"
        )

    def test_input_circuit_not_mutated_on_scaled_path(self):
        qc = _bell()
        n_ops_before = len(qc.data)
        make_executor("FakeManilaV2@x1.5", shots=128, seed=3)(qc, "ZZ")
        assert len(qc.data) == n_ops_before
        assert qc.num_clbits == 0
