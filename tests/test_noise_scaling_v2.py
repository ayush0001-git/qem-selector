"""Tests for the SUB-UNITY (downward) noise dial: ``"<FakeName>@x<scale>"`` with
``0 < scale < 1`` (B2 / INTERFACES.md V2).

Companion to ``test_noise_scaling.py`` (which pins the UPWARD dial, scale >= 1).
The downward dial (``LOW_NOISE_SCALES = (0.25, 0.5)``) extends the noise axis
DOWN into the clean, Heron-like regime the research grid lacked (PROJECT_STATUS
finding F7 / END_RESULT: the selector's one hardware miss sits exactly there).
Same synthetic parametric model as the upward dial — ``scale`` MULTIPLIES every
calibrated gate/readout error — now exercised below 1.0.

Contracts pinned here:

1. **Parsing.** Sub-unity suffixes (``x0.25``, ``x0.5``, ``x.5``, ``x0.1``)
   parse to their float. Zero / negative / non-finite scales still raise even
   though they are "below 1". ``ibm_*`` hardware still refuses every suffix.
2. **``get_backend_info`` below 1.0 scales EXACTLY linearly** — the caps
   (0.9 gate / 0.45 readout) never engage below 1.0. Proven on the
   cap-saturated device FakeLagosV2 (plain q2 readout 46.4% > 0.45 cap, yet
   ``@x0.5`` reports exactly 23.19%, ratio 0.5, strictly under the cap — this
   is the property that distinguishes the sub-unity path from the upward dial,
   where Lagos saturates). Scaled numbers flow through ``extract_features``.
3. **Monotone DOWNWARD dial.** ``|raw error|`` strictly decreases
   ``x1.0 > x0.5 > x0.25`` on a fixed workload (mirror of the upward
   monotonicity test), and the dial stays monotone continuously ACROSS unity
   (``x0.5 < x1.0 < x1.5`` — "both directions" in a single chain). Device
   routing still applies (scaled) noise on non-device-edge pairs.
4. **Plain-name / scale>=1 REGRESSION (capture-first).** Byte-identical to the
   pre-change code — the sub-unity capability is purely additive. Anchored by
   hardcoded values captured from the current code (plain Bell <ZZ> = 0.8671875,
   unchanged upward ratios), plus exact determinism of the sub-unity executor
   with hardcoded sub-unity anchors.
"""

from __future__ import annotations

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

# --- Capture-first anchors (exact values from the CURRENT code) -------------
#: Pre-scaling regression anchor (identical to test_noise_scaling.py): the plain
#: FakeManilaV2 executor on the Bell circuit h(0);cx(0,1), pauli "ZZ", 256 shots,
#: seed 7 -> 222/256, exactly representable so the comparison is EXACT.
_PLAIN_BELL_ZZ = 0.8671875
#: Sub-unity determinism anchors: SAME (256 shots, seed 7) Bell <ZZ>, captured
#: on the current code at scales 0.5 and 0.25. 240/256 and 248/256 — both
#: exactly representable, so these are EXACT-equality regression pins that also
#: guard cross-process reproducibility of the scaled noise model.
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
    """|raw error| of <ZZ> on a FIXED 10-CNOT circuit (ideal <ZZ> = +1).

    Prepare |11> then apply 10 cx(0,1) (even count -> identity on the state),
    measure ZZ. Any deviation from +1 is noise. Same helper as the upward
    test so the two dials are measured on an identical workload.
    """
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
# 1. Sub-unity name parsing
# ---------------------------------------------------------------------------


class TestParseSubUnity:
    @pytest.mark.parametrize(
        "name,base,scale",
        [
            ("FakeManilaV2@x0.25", "FakeManilaV2", 0.25),
            ("FakeManilaV2@x0.5", "FakeManilaV2", 0.5),
            ("FakeManilaV2@x.5", "FakeManilaV2", 0.5),  # leading zero optional
            ("FakeLagosV2@x0.1", "FakeLagosV2", 0.1),
            ("FakeJakartaV2@x0.75", "FakeJakartaV2", 0.75),
        ],
    )
    def test_subunity_suffix_parsed(self, name, base, scale):
        assert parse_backend_name(name) == (base, scale)

    def test_low_noise_scales_constant(self):
        # The declared downward-dial coverage — every entry is a valid
        # sub-unity scale and parses on a real backend name.
        assert LOW_NOISE_SCALES == (0.25, 0.5)
        for scale in LOW_NOISE_SCALES:
            assert 0.0 < scale < 1.0
            assert parse_backend_name(f"FakeManilaV2@x{scale}") == (
                "FakeManilaV2",
                scale,
            )

    @pytest.mark.parametrize(
        "bad",
        [
            "FakeManilaV2@x0",  # zero is "below 1" but still invalid
            "FakeManilaV2@x0.0",
            "FakeManilaV2@x-0.25",  # negative sub-unity
            "FakeManilaV2@x-0.5",
            "FakeManilaV2@xnan",  # non-finite
        ],
    )
    def test_invalid_subunity_like_still_raises(self, bad):
        with pytest.raises(ValueError):
            parse_backend_name(bad)

    def test_ibm_subunity_suffix_refused(self):
        # Noise scaling is simulation-only — refused on real hardware at ANY
        # scale, including sub-unity.
        with pytest.raises(ValueError):
            parse_backend_name("ibm_brisbane@x0.5")
        with pytest.raises(ValueError):
            get_backend_info("ibm_brisbane@x0.25")

    def test_unknown_base_with_subunity_suffix_raises(self):
        with pytest.raises(ValueError, match="unknown backend"):
            get_backend_info("FakeNopeV2@x0.5")
        with pytest.raises(ValueError, match="unknown backend"):
            make_executor("FakeNopeV2@x0.25", shots=16, seed=0)


# ---------------------------------------------------------------------------
# 2. get_backend_info scales exactly linearly below 1.0 (caps never bind)
# ---------------------------------------------------------------------------


class TestSubUnityBackendInfo:
    @pytest.mark.parametrize("scale", LOW_NOISE_SCALES)
    def test_ratio_is_exactly_the_scale_manila(self, scale):
        # FakeManilaV2 errors are all far below the caps, so every average must
        # scale by EXACTLY the sub-unity factor.
        base = get_backend_info("FakeManilaV2")
        scaled = get_backend_info(f"FakeManilaV2@x{scale}")
        for key in _SCALING_INFO_KEYS:
            assert scaled[key] == pytest.approx(scale * base[key], rel=1e-12), key

    @pytest.mark.parametrize("scale", LOW_NOISE_SCALES)
    def test_caps_never_bind_below_unity_on_saturated_lagos(self, scale):
        # THE key sub-unity contract. FakeLagosV2 q2 stores 46.4% readout error
        # — ABOVE the 0.45 readout cap — so the UPWARD dial saturates it. Below
        # 1.0 the caps must NEVER engage: the ratio is exactly the scale and
        # every scaled value stays strictly under its cap.
        base = get_backend_info("FakeLagosV2")
        assert base["max_readout_error"] > _SCALED_READOUT_ERROR_CAP  # cap would cut it
        scaled = get_backend_info(f"FakeLagosV2@x{scale}")
        for key in _SCALING_INFO_KEYS:
            assert scaled[key] == pytest.approx(scale * base[key], rel=1e-12), key
        # Strictly under the caps -> proof no min(...) clamp engaged.
        assert scaled["max_readout_error"] < _SCALED_READOUT_ERROR_CAP
        assert scaled["avg_readout_error"] < _SCALED_READOUT_ERROR_CAP
        assert scaled["avg_2q_error"] < _SCALED_GATE_ERROR_CAP
        assert scaled["avg_1q_error"] < _SCALED_GATE_ERROR_CAP

    def test_third_device_jakarta_ratio(self):
        base = get_backend_info("FakeJakartaV2")
        scaled = get_backend_info("FakeJakartaV2@x0.5")
        for key in _SCALING_INFO_KEYS:
            assert scaled[key] == pytest.approx(0.5 * base[key], rel=1e-12), key

    def test_schema_and_name_echo(self):
        scaled = get_backend_info("FakeManilaV2@x0.5")
        assert sorted(scaled) == sorted(_INFO_KEYS)
        assert scaled["name"] == "FakeManilaV2@x0.5"
        assert scaled["n_qubits"] == get_backend_info("FakeManilaV2")["n_qubits"]

    def test_cached_copy_is_safe_to_mutate(self):
        first = get_backend_info("FakeManilaV2@x0.25")
        first["avg_2q_error"] = 999.0
        assert get_backend_info("FakeManilaV2@x0.25")["avg_2q_error"] != 999.0

    def test_scaled_numbers_flow_through_extract_features(self):
        from qemsel.features import extract_features

        qc = _bell()
        base = extract_features(qc, "FakeManilaV2")
        scaled = extract_features(qc, "FakeManilaV2@x0.5")
        assert scaled["backend_avg_2q_error"] == pytest.approx(
            0.5 * base["backend_avg_2q_error"], rel=1e-12
        )
        assert scaled["backend_avg_readout_error"] == pytest.approx(
            0.5 * base["backend_avg_readout_error"], rel=1e-12
        )
        # Circuit-side features are backend-independent -> identical.
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
# 3. Plain-name / scale>=1 regression: byte-identical (capture-first)
# ---------------------------------------------------------------------------


class TestPlainAndUpwardRegression:
    def test_plain_name_matches_prechange_value_exactly(self):
        value = make_executor("FakeManilaV2", shots=256, seed=7)(_bell(), "ZZ")
        assert value == _PLAIN_BELL_ZZ  # EXACT — adding sub-unity must not drift it

    def test_x1_suffix_matches_prechange_value_exactly(self):
        value = make_executor("FakeManilaV2@x1.0", shots=256, seed=7)(_bell(), "ZZ")
        assert value == _PLAIN_BELL_ZZ

    def test_upward_dial_ratios_unchanged(self):
        # The >=1 path must be untouched by the downward extension: uncapped
        # Manila ratios are still exactly the nominal scale.
        base = get_backend_info("FakeManilaV2")
        for scale in (1.5, 2.0):
            scaled = get_backend_info(f"FakeManilaV2@x{scale}")
            for key in _SCALING_INFO_KEYS:
                assert scaled[key] == pytest.approx(scale * base[key], rel=1e-12), (
                    scale,
                    key,
                )

    def test_x1_suffix_info_is_uncapped_plain_verbatim(self):
        # Scale 1.0 (suffix form) must return the UNSCALED, UNCAPPED Lagos
        # numbers verbatim — regression guard shared with the upward tests.
        base = get_backend_info("FakeLagosV2")
        suffixed = get_backend_info("FakeLagosV2@x1.0")
        assert suffixed["name"] == "FakeLagosV2@x1.0"
        for key in _INFO_KEYS[1:]:
            assert suffixed[key] == base[key], key
        assert base["max_readout_error"] > _SCALED_READOUT_ERROR_CAP


# ---------------------------------------------------------------------------
# 4. Sub-unity executor: determinism, monotone downward dial, routing
# ---------------------------------------------------------------------------


class TestSubUnityExecutor:
    def test_deterministic_within_and_across_builds(self):
        ex_a = make_executor("FakeManilaV2@x0.5", shots=512, seed=11)
        ex_b = make_executor("FakeManilaV2@x0.5", shots=512, seed=11)
        qc = _bell()
        first = ex_a(qc, "ZZ")
        assert ex_a(qc, "ZZ") == first  # same executor, twice
        assert ex_b(qc, "ZZ") == first  # freshly built executor

    def test_subunity_bell_anchor_x0_5(self):
        value = make_executor("FakeManilaV2@x0.5", shots=256, seed=7)(_bell(), "ZZ")
        assert value == _SUBUNITY_BELL_ZZ_X0_5  # EXACT cross-process pin

    def test_subunity_bell_anchor_x0_25(self):
        value = make_executor("FakeManilaV2@x0.25", shots=256, seed=7)(_bell(), "ZZ")
        assert value == _SUBUNITY_BELL_ZZ_X0_25

    def test_subunity_is_cleaner_than_plain(self):
        # The dial's PURPOSE: sub-unity must be measurably cleaner than x1.0 on
        # a readout-dominated probe (Lagos |11> <ZZ>: ~0.52 err at x1.0 vs
        # ~0.15 at x0.25). Margin 0.1 >> shot sd (~0.011 at 8000 shots).
        err_x1 = _readout_only_error("FakeLagosV2@x1.0")
        err_x025 = _readout_only_error("FakeLagosV2@x0.25")
        assert err_x025 < err_x1 - 0.1, (
            f"downward dial did not reduce noise: |err| x0.25={err_x025:.4f} "
            f"x1.0={err_x1:.4f}"
        )

    def test_downward_dial_monotone_lagos_readout(self):
        # Full 3-point downward chain on a readout-dominated probe (Lagos has
        # the largest, cleanly separated gaps): |err| ~0.145 / 0.280 / 0.516 at
        # x0.25 / x0.5 / x1.0. Margin 0.05 << the ~0.13/0.24 gaps.
        err_025 = _readout_only_error("FakeLagosV2@x0.25")
        err_05 = _readout_only_error("FakeLagosV2@x0.5")
        err_10 = _readout_only_error("FakeLagosV2@x1.0")
        assert err_05 > err_025 + 0.05, (err_025, err_05)
        assert err_10 > err_05 + 0.05, (err_05, err_10)

    def test_downward_dial_monotone_lagos_deepcx(self):
        # Same 3-point chain on the 10-CNOT (gate-dominated) probe: |err|
        # ~0.172 / 0.318 / 0.573 at x0.25 / x0.5 / x1.0.
        err_025 = _deep_cx_error("FakeLagosV2@x0.25")
        err_05 = _deep_cx_error("FakeLagosV2@x0.5")
        err_10 = _deep_cx_error("FakeLagosV2")
        assert err_05 > err_025 + 0.05, (err_025, err_05)
        assert err_10 > err_05 + 0.05, (err_05, err_10)

    def test_downward_dial_manila_robust_steps(self):
        # Second device: Manila |err| ~0.047 / 0.094 / 0.209 at x0.25/x0.5/x1.0.
        # Assert only the wide steps (x0.25<x1.0, x0.5<x1.0) to stay well clear
        # of the ~0.011 shot sd; the tight x0.25->x0.5 step is covered on Lagos.
        err_025 = _deep_cx_error("FakeManilaV2@x0.25")
        err_05 = _deep_cx_error("FakeManilaV2@x0.5")
        err_10 = _deep_cx_error("FakeManilaV2")
        assert err_10 > err_05 + 0.05, (err_05, err_10)
        assert err_10 > err_025 + 0.05, (err_025, err_10)

    def test_monotone_across_unity_both_directions(self):
        # "Both directions": one monotone chain spanning the unity boundary,
        # Manila 10-CNOT |err| ~0.094 / 0.209 / 0.276 at x0.5 / x1.0 / x1.5.
        # Confirms the sub-unity extension joins the upward dial continuously.
        err_05 = _deep_cx_error("FakeManilaV2@x0.5")
        err_10 = _deep_cx_error("FakeManilaV2")
        err_15 = _deep_cx_error("FakeManilaV2@x1.5")
        assert err_10 > err_05 + 0.05, (err_05, err_10)
        assert err_15 > err_10 + 0.05, (err_10, err_15)

    def test_subunity_path_still_routes_noise_on_non_device_edges(self):
        # Sub-unity counterpart of the upward routing test: (3,4) is NOT a
        # FakeLagosV2 edge; the scaled path must keep the from_backend coupling
        # map so routed 2q gates still pick up (scaled) noise. A 40-cx chain
        # from |11> decays ref 0.954 -> deep 0.410 at x0.5 (drop 0.54).
        executor = make_executor("FakeLagosV2@x0.5", shots=2000, seed=0)
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
            f"{ref_val - deep_val:.4f} — sub-unity model lost device routing"
        )

    def test_input_circuit_not_mutated_on_subunity_path(self):
        qc = _bell()
        n_ops_before = len(qc.data)
        make_executor("FakeManilaV2@x0.5", shots=128, seed=3)(qc, "ZZ")
        assert len(qc.data) == n_ops_before
        assert qc.num_clbits == 0


def test_backends_list_unchanged_by_v2():
    # The dial adds NO new plain names — sub-unity is reached purely by suffix.
    assert BACKENDS == [
        "FakeManilaV2",
        "FakeJakartaV2",
        "FakeLagosV2",
        "FakeSherbrooke",
    ]
