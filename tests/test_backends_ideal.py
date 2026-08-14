"""Unit tests for qemsel.backends and qemsel.ideal (builder-backends).

Fast by design: only ONE noise model is built (module-scoped executor on
FakeManilaV2, 2 qubits, 256 shots); everything else is exact statevector or
pure counts arithmetic. Endianness is the critical contract here:

* qemsel pauli convention: pauli[i] acts on qubit i (LEFTMOST char = q0).
* qiskit counts keys are little-endian (RIGHTMOST bit = q0).
"""

from __future__ import annotations

import math

import pytest
from qiskit import QuantumCircuit

from qemsel.backends import (
    BACKENDS,
    RealHardwareBackend,
    expectation_from_counts,
    get_backend_info,
    make_executor,
)
from qemsel.ideal import ideal_expectation

# ---------------------------------------------------------------------------
# expectation_from_counts — hand-computed endianness cases
# ---------------------------------------------------------------------------


class TestExpectationFromCounts:
    def test_key_01_pauli_zi_is_minus_one(self):
        # Key '01': rightmost bit = q0 = 1, q1 = 0. 'ZI' = Z on q0 -> -1.
        assert expectation_from_counts({"01": 100}, "ZI") == -1.0

    def test_key_01_pauli_iz_is_plus_one(self):
        # 'IZ' = Z on q1; q1 measured 0 -> +1.
        assert expectation_from_counts({"01": 100}, "IZ") == 1.0

    def test_key_01_pauli_zz_is_minus_one(self):
        # Joint parity of q0=1, q1=0 is odd -> -1.
        assert expectation_from_counts({"01": 100}, "ZZ") == -1.0

    def test_key_10_flips_the_two_locals(self):
        # Key '10': q0 = 0, q1 = 1 — mirror image of the '01' cases.
        assert expectation_from_counts({"10": 7}, "ZI") == 1.0
        assert expectation_from_counts({"10": 7}, "IZ") == -1.0

    def test_all_identity_returns_one(self):
        assert expectation_from_counts({"01": 3, "10": 5}, "II") == 1.0

    def test_weighted_mixture(self):
        # 'ZI' on {'00': 3, '01': 1}: q0 bits are 0,0,0,1 -> (3 - 1)/4 = 0.5.
        assert expectation_from_counts({"00": 3, "01": 1}, "ZI") == pytest.approx(0.5)

    def test_three_qubits_leftmost_key_bit_is_last_qubit(self):
        # Key '100': q2 = 1, q1 = 0, q0 = 0.
        counts = {"100": 11}
        assert expectation_from_counts(counts, "IIZ") == -1.0  # Z on q2
        assert expectation_from_counts(counts, "ZII") == 1.0  # Z on q0
        assert expectation_from_counts(counts, "ZZI") == 1.0  # q0,q1 even
        assert expectation_from_counts(counts, "ZZZ") == -1.0  # odd parity

    def test_bell_counts_zz(self):
        # Ideal-ish Bell counts: ZZ = P(00)+P(11)-P(01)-P(10).
        counts = {"00": 450, "11": 450, "01": 50, "10": 50}
        assert expectation_from_counts(counts, "ZZ") == pytest.approx(0.8)

    def test_keys_with_spaces_are_stripped(self):
        # Multi-register style key '0 1' == '01'.
        assert expectation_from_counts({"0 1": 10}, "ZI") == -1.0

    def test_empty_counts_raise(self):
        with pytest.raises(ValueError):
            expectation_from_counts({}, "ZZ")

    def test_short_key_raises(self):
        with pytest.raises(ValueError):
            expectation_from_counts({"0": 5}, "ZZ")


# ---------------------------------------------------------------------------
# ideal_expectation — exact statevector, endianness via asymmetric circuits
# ---------------------------------------------------------------------------


class TestIdealExpectation:
    def test_bell_zz_is_plus_one(self, tiny_circuit):
        assert ideal_expectation(tiny_circuit, "ZZ") == pytest.approx(1.0)

    def test_bell_locals_are_zero(self, tiny_circuit):
        assert ideal_expectation(tiny_circuit, "ZI") == pytest.approx(0.0, abs=1e-12)
        assert ideal_expectation(tiny_circuit, "IZ") == pytest.approx(0.0, abs=1e-12)

    def test_bell_xx_is_plus_one(self, tiny_circuit):
        assert ideal_expectation(tiny_circuit, "XX") == pytest.approx(1.0)

    def test_endianness_x_on_qubit0_only(self):
        # |psi> = |q1=0, q0=1>: Z on q0 -> -1, Z on q1 -> +1.
        qc = QuantumCircuit(2)
        qc.x(0)
        assert ideal_expectation(qc, "ZI") == pytest.approx(-1.0)
        assert ideal_expectation(qc, "IZ") == pytest.approx(1.0)

    def test_endianness_x_on_qubit1_only(self):
        qc = QuantumCircuit(2)
        qc.x(1)
        assert ideal_expectation(qc, "ZI") == pytest.approx(1.0)
        assert ideal_expectation(qc, "IZ") == pytest.approx(-1.0)

    def test_y_eigenstate(self):
        # h; s |0> = (|0> + i|1>)/sqrt(2), the +1 eigenstate of Y.
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.s(0)
        assert ideal_expectation(qc, "Y") == pytest.approx(1.0)

    def test_all_identity(self, tiny_circuit):
        assert ideal_expectation(tiny_circuit, "II") == pytest.approx(1.0)

    def test_identity_circuit_zzz(self, tiny_identity_circuit):
        assert ideal_expectation(tiny_identity_circuit, "ZZZ") == pytest.approx(1.0)

    def test_matches_fake_executor_fixture(self, tiny_circuit, fake_executor):
        for pauli in ("ZZ", "ZI", "IZ", "XX", "II"):
            assert ideal_expectation(tiny_circuit, pauli) == pytest.approx(
                fake_executor(tiny_circuit, pauli), abs=1e-9
            )

    def test_does_not_mutate_circuit(self, tiny_circuit):
        n_ops_before = len(tiny_circuit.data)
        ideal_expectation(tiny_circuit, "ZZ")
        assert len(tiny_circuit.data) == n_ops_before

    def test_length_mismatch_raises(self, tiny_circuit):
        with pytest.raises(ValueError):
            ideal_expectation(tiny_circuit, "ZZZ")

    def test_invalid_char_raises(self, tiny_circuit):
        with pytest.raises(ValueError):
            ideal_expectation(tiny_circuit, "ZA")


# ---------------------------------------------------------------------------
# get_backend_info
# ---------------------------------------------------------------------------

_EXPECTED_INFO_KEYS = [
    "name",
    "n_qubits",
    "avg_1q_error",
    "avg_2q_error",
    "avg_readout_error",
    "max_readout_error",
]


class TestGetBackendInfo:
    @pytest.mark.parametrize("name,n_qubits", [("FakeManilaV2", 5), ("FakeLagosV2", 7)])
    def test_keys_and_sane_values(self, name, n_qubits):
        info = get_backend_info(name)
        assert sorted(info) == sorted(_EXPECTED_INFO_KEYS)
        assert info["name"] == name
        assert info["n_qubits"] == n_qubits
        for key in ("avg_1q_error", "avg_2q_error", "avg_readout_error"):
            assert 0.0 < info[key] < 1.0, f"{name}.{key} = {info[key]}"
            assert not math.isnan(info[key])
        assert info["max_readout_error"] >= info["avg_readout_error"]
        # Calibration snapshots: 2q gates are noisier than 1q gates.
        assert info["avg_2q_error"] > info["avg_1q_error"]

    def test_lagos_has_extreme_readout(self):
        # PROJECT_STATE.md: FakeLagosV2 snapshot has ~27% readout on q0/q1.
        assert get_backend_info("FakeLagosV2")["max_readout_error"] > 0.2

    def test_cached_and_copy_safe(self):
        first = get_backend_info("FakeManilaV2")
        first["avg_2q_error"] = 999.0  # mutate the returned copy
        second = get_backend_info("FakeManilaV2")
        assert second["avg_2q_error"] != 999.0  # cache not corrupted
        del first["avg_2q_error"]
        del second["avg_2q_error"]
        assert first == second

    def test_unknown_name_raises(self):
        with pytest.raises(ValueError):
            get_backend_info("FakeNopeV2")

    def test_backends_constant(self):
        assert BACKENDS == [
            "FakeManilaV2",
            "FakeJakartaV2",
            "FakeLagosV2",
            "FakeSherbrooke",
        ]


# ---------------------------------------------------------------------------
# make_executor — ONE noise model built for the whole module (FakeManilaV2)
# ---------------------------------------------------------------------------

_SHOTS = 256
_SEED = 7


@pytest.fixture(scope="module")
def manila_executor():
    """Shared noisy executor: FakeManilaV2, 256 shots, seed 7."""
    return make_executor("FakeManilaV2", shots=_SHOTS, seed=_SEED)


class TestMakeExecutor:
    def test_noisy_bell_zz_close_to_ideal(self, manila_executor):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        value = manila_executor(qc, "ZZ")
        # Ideal is +1; Manila noise costs ~0.1, shot noise a bit more.
        assert 0.7 <= value <= 1.0

    def test_noisy_bell_xx_close_to_ideal(self, manila_executor):
        # Exercises the X basis-rotation path.
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        value = manila_executor(qc, "XX")
        assert 0.7 <= value <= 1.0

    def test_deterministic_given_seed(self, manila_executor):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        assert manila_executor(qc, "ZZ") == manila_executor(qc, "ZZ")

    def test_all_identity_shortcut(self, manila_executor):
        qc = QuantumCircuit(2)
        qc.h(0)
        assert manila_executor(qc, "II") == 1.0

    def test_does_not_mutate_input_circuit(self, manila_executor):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        n_ops_before = len(qc.data)
        manila_executor(qc, "ZZ")
        assert len(qc.data) == n_ops_before
        assert qc.num_clbits == 0  # no measurements were added to the input

    def test_length_mismatch_raises(self, manila_executor):
        with pytest.raises(ValueError):
            manila_executor(QuantumCircuit(2), "Z")

    def test_invalid_char_raises(self, manila_executor):
        with pytest.raises(ValueError):
            manila_executor(QuantumCircuit(2), "ZQ")

    def test_unknown_backend_raises_at_make_time(self):
        with pytest.raises(ValueError):
            make_executor("NotABackend", shots=16, seed=0)


class TestNoiselessExecutorMachinery:
    """The full measure->transpile->run->counts pipeline WITHOUT a noise
    model must approximate the exact statevector value within shot noise."""

    def _noiseless_run(self, circuit: QuantumCircuit, pauli: str, shots: int, seed: int) -> float:
        from qiskit import transpile
        from qiskit_aer import AerSimulator

        simulator = AerSimulator()  # NO noise model
        measured = circuit.copy()
        for qubit, char in enumerate(pauli):
            if char == "X":
                measured.h(qubit)
            elif char == "Y":
                measured.sdg(qubit)
                measured.h(qubit)
        measured.measure_all()
        transpiled = transpile(
            measured, simulator, optimization_level=0, seed_transpiler=seed
        )
        counts = (
            simulator.run(transpiled, shots=shots, seed_simulator=seed)
            .result()
            .get_counts()
        )
        return expectation_from_counts(counts, pauli)

    def test_bell_zz_matches_ideal_exactly(self, tiny_circuit):
        # Bell ZZ has zero variance (outcomes 00/11 only) — exact even at
        # finite shots.
        value = self._noiseless_run(tiny_circuit, "ZZ", shots=512, seed=3)
        assert value == pytest.approx(ideal_expectation(tiny_circuit, "ZZ"))

    def test_rotated_state_within_shot_tolerance(self):
        # <Z> on ry(pi/3)|0> = cos(pi/3) = 0.5; sd ~ 1/sqrt(shots) ~ 0.03.
        qc = QuantumCircuit(1)
        qc.ry(math.pi / 3, 0)
        value = self._noiseless_run(qc, "Z", shots=1024, seed=5)
        assert value == pytest.approx(ideal_expectation(qc, "Z"), abs=0.12)

    def test_asymmetric_endianness_through_sampling(self):
        # q0 flipped: sampled 'ZI' must be -1, 'IZ' must be +1 — the sampled
        # pipeline agrees with the statevector on WHICH qubit is which.
        qc = QuantumCircuit(2)
        qc.x(0)
        assert self._noiseless_run(qc, "ZI", shots=128, seed=1) == -1.0
        assert self._noiseless_run(qc, "IZ", shots=128, seed=1) == 1.0


# ---------------------------------------------------------------------------
# Regression: 2-qubit gate noise must fire on EVERY pair the suite uses
# (review 2026-07-21). Pre-fix, make_executor transpiled against a bare
# AerSimulator with no coupling map, so cx on non-device-edges (e.g. (2,3)
# and (3,4) on the H-topology Lagos/Jakarta) and wrong-direction ecr
# (Sherbrooke) executed with EXACTLY ZERO gate noise. The from_backend fix
# routes onto real edges, so a deep cx chain from |11> must now decay.
# ---------------------------------------------------------------------------


def _zz_decay(backend_name: str, pair: tuple[int, int], n_cx: int,
              n_qubits: int, shots: int = 4000) -> tuple[float, float]:
    """(reference, decayed) <Z_a Z_b> from |11> on ``pair``, 0 vs n_cx cx."""
    executor = make_executor(backend_name, shots=shots, seed=0)
    a, b = pair
    pauli = "".join("Z" if q in pair else "I" for q in range(n_qubits))
    ref = QuantumCircuit(n_qubits)
    ref.x(a)
    ref.x(b)
    deep = ref.copy()
    for _ in range(n_cx):
        deep.cx(a, b)
    return executor(ref, pauli), executor(deep, pauli)


def test_lagos_uncoupled_pair_gets_noise():
    # (3,4) is NOT a FakeLagosV2 edge; pre-fix the deep chain decayed by
    # EXACTLY 0.0 (bit-identical to the reference). Post-fix routing makes
    # the noise fire hard (probe: 0.91 -> 0.10 over 60 cx).
    ref, deep = _zz_decay("FakeLagosV2", (3, 4), n_cx=60, n_qubits=5)
    assert ref > 0.7, f"reference state already broken: {ref}"
    assert ref - deep > 0.3, (
        f"deep cx chain on Lagos pair (3,4) decayed only {ref - deep:.4f} "
        "— 2q noise is not firing on non-device-edge pairs again"
    )


@pytest.mark.slow
@pytest.mark.parametrize(
    "backend_name,pair,n_qubits",
    [
        ("FakeJakartaV2", (3, 4), 5),   # (3,4) not a Jakarta edge
        ("FakeSherbrooke", (0, 1), 2),  # ecr stored in ONE direction only
        ("FakeSherbrooke", (3, 4), 5),  # deep-chain pair on the 127q device
    ],
)
def test_every_suite_pair_gets_noise(backend_name, pair, n_qubits):
    ref, deep = _zz_decay(backend_name, pair, n_cx=40, n_qubits=n_qubits,
                          shots=2000)
    assert ref - deep > 0.05, (
        f"deep cx chain on {backend_name} pair {pair} decayed only "
        f"{ref - deep:.4f} — 2q noise silently dropped"
    )


def test_executor_rejects_circuit_wider_than_backend(manila_executor):
    # FakeManilaV2 has 5 qubits; qubits beyond the device would simulate
    # with NO noise at all (silent wrong data), so this must be an error.
    qc = QuantumCircuit(6)
    qc.h(0)
    for q in range(5):
        qc.cx(q, q + 1)
    with pytest.raises(ValueError, match="6 qubits"):
        manila_executor(qc, "Z" * 6)


# ---------------------------------------------------------------------------
# RealHardwareBackend stub
# ---------------------------------------------------------------------------


def test_real_hardware_backend_not_implemented():
    with pytest.raises(NotImplementedError, match="configs/hardware.yaml"):
        RealHardwareBackend()


# ---------------------------------------------------------------------------
# Integration regression (integrator): known-answer families must be EXACT.
# ideal_expectation snaps sub-1e-10 float dust to the nearest integer so the
# mirror-circuit contract (ideal <Z...Z> == +1.0 exactly) holds bit-exactly.
# ---------------------------------------------------------------------------


def test_mirror_circuit_ideal_is_exactly_one():
    from qemsel.circuits import mirror_circuit

    for seed in (0, 1, 2):
        qc = mirror_circuit(2, 4, seed)
        assert ideal_expectation(qc, "ZZ") == 1.0


def test_snap_does_not_touch_generic_values():
    qc = QuantumCircuit(1)
    qc.ry(math.pi / 3, 0)  # <Z> = cos(pi/3) = 0.5, nowhere near an integer
    assert ideal_expectation(qc, "Z") == pytest.approx(0.5, abs=1e-12)
