"""Unit tests for qemsel.circuits (builder-circuits).

Fast, simulation-light tests: statevectors only, <= 5 qubits.
"""

from __future__ import annotations

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli, Statevector

from qemsel.circuits import (
    FAMILIES,
    MIN_ABS_IDEAL_EXEMPT_FAMILIES,
    MIN_ABS_IDEAL_MAX_ATTEMPTS,
    SUB_SEED_STRIDE,
    CircuitSpec,
    generate_suite,
    ghz_plus,
    hw_efficient_ansatz,
    layered_random,
    mirror_circuit,
    near_clifford,
)

#: Gate names every generated circuit is allowed to contain (module contract).
ALLOWED_GATES = {
    "h", "x", "y", "z", "s", "sdg", "t", "tdg", "sx",
    "rx", "ry", "rz", "cx", "cz", "barrier",
}

FAMILY_NAMES = sorted(FAMILIES)


def _op_list(qc: QuantumCircuit) -> list[tuple]:
    """Structural fingerprint: (name, params, qubit indices) per instruction."""
    out = []
    for inst in qc.data:
        qubits = tuple(qc.find_bit(q).index for q in inst.qubits)
        params = tuple(float(p) for p in inst.operation.params)
        out.append((inst.operation.name, params, qubits))
    return out


def _zn_expectation(qc: QuantumCircuit) -> float:
    """Exact <Z...Z> using the qemsel pauli convention (pauli[i] -> qubit i)."""
    pauli = "Z" * qc.num_qubits
    sv = Statevector.from_instruction(qc)
    return float(np.real(sv.expectation_value(Pauli(pauli[::-1]))))


# ---------------------------------------------------------------------------
# CircuitSpec
# ---------------------------------------------------------------------------


def test_circuit_spec_id_format():
    spec = CircuitSpec(family="ghz_plus", n_qubits=3, depth=8, seed=42)
    assert spec.circuit_id == "ghz_plus_q3_d8_s42"


def test_circuit_spec_params_default_not_shared():
    a = CircuitSpec("layered_random", 2, 4, 0)
    b = CircuitSpec("layered_random", 2, 4, 1)
    assert a.params == {}
    a.params["x"] = 1
    assert b.params == {}, "default params dict must not be shared"


# ---------------------------------------------------------------------------
# Common contract for every family
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("family", FAMILY_NAMES)
@pytest.mark.parametrize("n_qubits", [2, 4])
@pytest.mark.parametrize("depth", [1, 6])
def test_family_common_contract(family, n_qubits, depth):
    qc = FAMILIES[family](n_qubits, depth, seed=7)
    assert isinstance(qc, QuantumCircuit)
    assert qc.num_qubits == n_qubits
    assert qc.num_clbits == 0, "no classical registers allowed"
    names = {inst.operation.name for inst in qc.data}
    assert "measure" not in names and "reset" not in names
    assert names <= ALLOWED_GATES, f"disallowed gates: {names - ALLOWED_GATES}"
    assert len(qc.data) > 0
    assert qc.num_parameters == 0, "no unbound qiskit Parameters allowed"


@pytest.mark.parametrize("family", FAMILY_NAMES)
def test_family_determinism_same_seed(family):
    a = FAMILIES[family](3, 6, seed=123)
    b = FAMILIES[family](3, 6, seed=123)
    assert _op_list(a) == _op_list(b)
    assert a == b  # qiskit structural equality


@pytest.mark.parametrize(
    "family",
    ["layered_random", "near_clifford", "hw_efficient_ansatz", "mirror_circuit"],
)
def test_family_seed_sensitivity(family):
    a = FAMILIES[family](3, 6, seed=0)
    b = FAMILIES[family](3, 6, seed=1)
    assert _op_list(a) != _op_list(b)


def test_ghz_plus_seed_sensitivity_in_padding():
    base = _op_list(ghz_plus(3, 14, seed=0))
    assert any(
        _op_list(ghz_plus(3, 14, seed=s)) != base for s in range(1, 6)
    ), "padding choices should depend on the seed"


# ---------------------------------------------------------------------------
# layered_random
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_qubits,depth", [(2, 1), (3, 4), (5, 7)])
def test_layered_random_structure(n_qubits, depth):
    qc = layered_random(n_qubits, depth, seed=3)
    ops = _op_list(qc)
    n_1q = sum(1 for name, _, qubits in ops if len(qubits) == 1)
    n_cx = sum(1 for name, _, _ in ops if name == "cx")
    assert n_1q == n_qubits * depth, "one rotation per qubit per layer"
    expected_cx = sum(
        len(range(layer % 2, n_qubits - 1, 2)) for layer in range(depth)
    )
    assert n_cx == expected_cx
    for name, _, qubits in ops:
        if len(qubits) == 1:
            assert name in ("rx", "ry", "rz")


def test_layered_random_invalid_args():
    with pytest.raises(ValueError):
        layered_random(0, 4, 0)
    with pytest.raises(ValueError):
        layered_random(3, 0, 0)


# ---------------------------------------------------------------------------
# near_clifford
# ---------------------------------------------------------------------------


def test_near_clifford_zero_fraction_is_all_clifford():
    qc = near_clifford(4, 10, seed=5, non_clifford_fraction=0.0)
    names = [inst.operation.name for inst in qc.data]
    assert "t" not in names and "rz" not in names and "tdg" not in names
    assert set(names) <= {"h", "s", "sdg", "x", "z", "cx"}


def test_near_clifford_full_fraction_all_slots_non_clifford():
    qc = near_clifford(3, 8, seed=5, non_clifford_fraction=1.0)
    non_cliff = [
        inst for inst in qc.data if inst.operation.name in ("t", "rz")
    ]
    assert len(non_cliff) == 3 * 8, "every 1q slot must be non-Clifford"


def test_near_clifford_rz_angles_are_not_clifford():
    qc = near_clifford(4, 20, seed=9, non_clifford_fraction=0.5)
    half_pi = np.pi / 2
    saw_rz = False
    for inst in qc.data:
        if inst.operation.name == "rz":
            saw_rz = True
            angle = float(inst.operation.params[0])
            residue = angle % half_pi
            assert min(residue, half_pi - residue) > 1e-9
    assert saw_rz, "expect at least one rz among 40 non-Clifford draws"


def test_near_clifford_contains_cx_and_mostly_clifford():
    qc = near_clifford(4, 12, seed=1)  # default fraction 0.15
    names = [inst.operation.name for inst in qc.data]
    assert "cx" in names
    n_slots = 4 * 12
    n_non_cliff = sum(1 for n in names if n in ("t", "rz"))
    assert n_non_cliff < n_slots / 2, "circuit should be mostly Clifford"


def test_near_clifford_invalid_fraction():
    with pytest.raises(ValueError):
        near_clifford(2, 4, 0, non_clifford_fraction=-0.1)
    with pytest.raises(ValueError):
        near_clifford(2, 4, 0, non_clifford_fraction=1.5)


# ---------------------------------------------------------------------------
# ghz_plus
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_qubits", [2, 3, 4, 5])
def test_ghz_plus_state_is_exactly_ghz(n_qubits):
    qc = ghz_plus(n_qubits, depth=12, seed=11)
    sv = Statevector.from_instruction(qc)
    target = np.zeros(2**n_qubits, dtype=complex)
    target[0] = 1 / np.sqrt(2)
    target[-1] = 1 / np.sqrt(2)
    fidelity = abs(np.vdot(target, sv.data)) ** 2
    assert fidelity == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("depth", [8, 15, 25])
def test_ghz_plus_padding_reaches_depth(depth):
    qc = ghz_plus(3, depth, seed=2)
    assert qc.depth() >= depth


def test_ghz_plus_small_depth_is_plain_prep():
    # GHZ prep on 3 qubits already has depth 3 >= 2: no padding expected.
    qc = ghz_plus(3, 2, seed=0)
    assert _op_list(qc) == [
        ("h", (), (0,)),
        ("cx", (), (0, 1)),
        ("cx", (), (1, 2)),
    ]


def test_ghz_plus_zn_expectation_parity():
    # <Z...Z> on GHZ: +1 for even n, 0 for odd n.
    assert _zn_expectation(ghz_plus(2, 10, seed=4)) == pytest.approx(1.0, abs=1e-9)
    assert _zn_expectation(ghz_plus(4, 10, seed=4)) == pytest.approx(1.0, abs=1e-9)
    assert _zn_expectation(ghz_plus(3, 10, seed=4)) == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# hw_efficient_ansatz
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_qubits,depth", [(2, 1), (3, 3), (5, 4)])
def test_hw_efficient_ansatz_structure(n_qubits, depth):
    qc = hw_efficient_ansatz(n_qubits, depth, seed=8)
    counts = dict(qc.count_ops())
    assert counts.get("ry", 0) == n_qubits * (depth + 1)
    assert counts.get("rz", 0) == n_qubits * (depth + 1)
    assert counts.get("cx", 0) == (n_qubits - 1) * depth
    assert qc.num_parameters == 0, "all angles must be bound numerically"


def test_hw_efficient_ansatz_angles_in_range():
    qc = hw_efficient_ansatz(3, 2, seed=8)
    for inst in qc.data:
        for p in inst.operation.params:
            assert 0.0 <= float(p) < 2 * np.pi


# ---------------------------------------------------------------------------
# mirror_circuit
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_qubits", [2, 3, 4, 5])
@pytest.mark.parametrize("depth", [1, 2, 6, 9])
def test_mirror_ideal_zz_is_plus_one(n_qubits, depth):
    qc = mirror_circuit(n_qubits, depth, seed=13)
    assert _zn_expectation(qc) == pytest.approx(1.0, abs=1e-9)


def test_mirror_is_u_then_u_inverse():
    n_qubits, depth, seed = 3, 8, 5
    u = layered_random(n_qubits, max(1, depth // 2), seed)
    qc = mirror_circuit(n_qubits, depth, seed)
    assert len(qc.data) == 2 * len(u.data)
    assert _op_list(qc)[: len(u.data)] == _op_list(u), "first half must be U"


def test_mirror_depth_one_still_valid():
    qc = mirror_circuit(2, 1, seed=0)  # U has max(1, 0) = 1 layer
    assert len(qc.data) > 0
    assert _zn_expectation(qc) == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# FAMILIES registry
# ---------------------------------------------------------------------------


def test_families_registry_contents():
    assert set(FAMILIES) == {
        "layered_random",
        "near_clifford",
        "ghz_plus",
        "hw_efficient_ansatz",
        "mirror_circuit",
    }
    assert all(callable(f) for f in FAMILIES.values())
    assert FAMILIES["mirror_circuit"] is mirror_circuit


# ---------------------------------------------------------------------------
# generate_suite
# ---------------------------------------------------------------------------


def _small_config() -> dict:
    return {
        "families": ["layered_random", "mirror_circuit"],
        "n_qubits": [2, 3],
        "depths": [2, 4],
        "seeds": [0, 1],
    }


def test_generate_suite_counts_and_order():
    suite = generate_suite(_small_config())
    assert len(suite) == 2 * 2 * 2 * 2
    # Nesting order: family outermost, then n_qubits, depths, seeds innermost.
    expected_ids = [
        f"{fam}_q{n}_d{d}_s{s}"
        for fam in ["layered_random", "mirror_circuit"]
        for n in [2, 3]
        for d in [2, 4]
        for s in [0, 1]
    ]
    assert [spec.circuit_id for _, spec in suite] == expected_ids
    assert len(set(expected_ids)) == len(expected_ids), "ids must be unique"


def test_generate_suite_pairs_match_direct_generation():
    suite = generate_suite(_small_config())
    for circuit, spec in suite:
        assert circuit.num_qubits == spec.n_qubits
        regen = FAMILIES[spec.family](
            spec.n_qubits, spec.depth, spec.seed, **spec.params
        )
        assert _op_list(circuit) == _op_list(regen)
        names = {inst.operation.name for inst in circuit.data}
        assert "measure" not in names
        assert circuit.num_clbits == 0


def test_generate_suite_deterministic_across_calls():
    a = generate_suite(_small_config())
    b = generate_suite(_small_config())
    assert [spec for _, spec in a] == [spec for _, spec in b]
    assert all(_op_list(ca) == _op_list(cb) for (ca, _), (cb, _) in zip(a, b))


def test_generate_suite_params_passthrough():
    config = {
        "families": ["near_clifford"],
        "n_qubits": [3],
        "depths": [6],
        "seeds": [0],
        "params": {"near_clifford": {"non_clifford_fraction": 0.0}},
    }
    suite = generate_suite(config)
    assert len(suite) == 1
    circuit, spec = suite[0]
    assert spec.params == {"non_clifford_fraction": 0.0}
    names = {inst.operation.name for inst in circuit.data}
    assert "t" not in names and "rz" not in names, "params must reach generator"


def test_generate_suite_params_for_other_family_ignored():
    config = _small_config()
    config["params"] = {"near_clifford": {"non_clifford_fraction": 0.5}}
    suite = generate_suite(config)
    assert all(spec.params == {} for _, spec in suite)


# ---------------------------------------------------------------------------
# generate_suite: min_abs_ideal source-level rejection sampling
# ---------------------------------------------------------------------------


def _fake_seed_keyed_ideal(values_by_attempt: dict[int, float]):
    """Fake ideal_expectation keyed on the generator's rejection attempt.

    Every family generator embeds its seed in ``circuit.name``
    (``..._s{seed}``); the attempt index is ``seed // SUB_SEED_STRIDE``.
    Lets rejection-sampling tests be fully deterministic with no
    statevector work.
    """

    def _ideal(circuit: QuantumCircuit, pauli: str) -> float:
        seed = int(circuit.name.rsplit("_s", 1)[1])
        attempt = seed // SUB_SEED_STRIDE
        return values_by_attempt.get(attempt, 0.0)

    return _ideal


def _screened_config(**overrides) -> dict:
    config = {
        "families": ["layered_random"],
        "n_qubits": [2],
        "depths": [4],
        "seeds": [0],
        "min_abs_ideal": 0.5,
    }
    config.update(overrides)
    return config


def test_min_abs_ideal_accepts_first_passing_subseed(monkeypatch):
    # Attempt 0 fails (0.1 < 0.5), attempt 1 passes (0.9) -> the emitted
    # circuit must be the attempt-1 circuit with its BUMPED seed recorded.
    monkeypatch.setattr(
        "qemsel.ideal.ideal_expectation",
        _fake_seed_keyed_ideal({0: 0.1, 1: 0.9}),
    )
    suite = generate_suite(_screened_config())
    assert len(suite) == 1
    circuit, spec = suite[0]
    assert spec.seed == 0 + 1 * SUB_SEED_STRIDE
    assert spec.circuit_id == f"layered_random_q2_d4_s{SUB_SEED_STRIDE}"
    # Reproducibility contract: spec.seed recreates the emitted circuit.
    regen = FAMILIES[spec.family](spec.n_qubits, spec.depth, spec.seed)
    assert _op_list(circuit) == _op_list(regen)


def test_min_abs_ideal_negative_ideal_magnitude_counts(monkeypatch):
    # |ideal| is what matters: -0.8 passes a 0.5 threshold on attempt 0.
    monkeypatch.setattr(
        "qemsel.ideal.ideal_expectation",
        _fake_seed_keyed_ideal({0: -0.8}),
    )
    suite = generate_suite(_screened_config())
    assert suite[0][1].seed == 0


def test_min_abs_ideal_cap_keeps_best_so_far_with_warning(monkeypatch):
    # No attempt ever reaches 0.5; attempt 7 has the largest |ideal| (0.3)
    # -> after the attempt cap the attempt-7 circuit is kept and a
    # RuntimeWarning is emitted. Values are all DISTINCT so the
    # constant-ideal early exit does not trigger.
    values = {
        k: 0.1 + k * 1e-6 for k in range(MIN_ABS_IDEAL_MAX_ATTEMPTS)
    }
    values[7] = 0.3
    monkeypatch.setattr(
        "qemsel.ideal.ideal_expectation", _fake_seed_keyed_ideal(values)
    )
    with pytest.warns(RuntimeWarning, match="best-so-far"):
        suite = generate_suite(_screened_config())
    assert suite[0][1].seed == 0 + 7 * SUB_SEED_STRIDE


@pytest.mark.parametrize("family", sorted(MIN_ABS_IDEAL_EXEMPT_FAMILIES))
def test_min_abs_ideal_exempt_families_untouched(monkeypatch, family):
    # Exempt families (seed-INDEPENDENT ideals: mirror_circuit exactly +1,
    # ghz_plus exactly GHZ at every padding seed): even an impossible
    # threshold must leave their seeds untouched and never call
    # ideal_expectation.
    assert MIN_ABS_IDEAL_EXEMPT_FAMILIES == {"mirror_circuit", "ghz_plus"}

    def _explode(circuit, pauli):  # pragma: no cover - must not be called
        raise AssertionError("ideal_expectation called for exempt family")

    monkeypatch.setattr("qemsel.ideal.ideal_expectation", _explode)
    suite = generate_suite(
        _screened_config(families=[family], seeds=[0, 1], min_abs_ideal=0.99)
    )
    assert [spec.seed for _, spec in suite] == [0, 1]


def test_min_abs_ideal_zero_disables_screening(monkeypatch):
    def _explode(circuit, pauli):  # pragma: no cover - must not be called
        raise AssertionError("ideal_expectation called with screening off")

    monkeypatch.setattr("qemsel.ideal.ideal_expectation", _explode)
    suite = generate_suite(_screened_config(min_abs_ideal=0.0))
    assert suite[0][1].seed == 0


def test_min_abs_ideal_real_statevector_guarantees_threshold():
    # End-to-end with the REAL ideal: every emitted non-exempt circuit must
    # either satisfy the threshold or be an explicitly warned best-so-far
    # straggler (deterministic: fixed seeds, exact statevector). The
    # continuous-rotation families must ALWAYS pass; quantized-ideal
    # near_clifford may honestly exhaust its attempts.
    import warnings as _warnings

    from qemsel.circuits import SUB_SEED_STRIDE as _STRIDE

    config = {
        "families": ["layered_random", "hw_efficient_ansatz", "near_clifford"],
        "n_qubits": [2, 3],
        "depths": [4],
        "seeds": [0, 1, 2],
        "min_abs_ideal": 0.25,
    }
    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        suite = generate_suite(config)
    warned = " || ".join(str(w.message) for w in caught)
    assert len(suite) == 3 * 2 * 1 * 3
    for circuit, spec in suite:
        if abs(_zn_expectation(circuit)) >= 0.25:
            continue
        base_seed = spec.seed % _STRIDE
        marker = (
            f"{spec.family} n_qubits={spec.n_qubits} depth={spec.depth} "
            f"seed={base_seed}"
        )
        assert marker in warned, f"unwarned below-threshold {spec.circuit_id}"
        assert spec.family == "near_clifford", (
            "continuous-rotation families must always reach the threshold: "
            f"{spec.circuit_id}"
        )
    # Deterministic across calls (byte-identical suites).
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore")
        again = generate_suite(config)
    assert [s.circuit_id for _, s in again] == [s.circuit_id for _, s in suite]
    assert all(
        _op_list(ca) == _op_list(cb) for (ca, _), (cb, _) in zip(suite, again)
    )
    # circuit_ids stay unique even with bumped seeds.
    ids = [s.circuit_id for _, s in suite]
    assert len(set(ids)) == len(ids)


def test_min_abs_ideal_layered_random_always_passes_at_research_threshold():
    # The research configs rely on the continuous-rotation families passing
    # the 0.25 source threshold without stragglers at every configured
    # size; lock that in for the exact (n, depth) grid research.yaml uses.
    config = {
        "families": ["layered_random", "hw_efficient_ansatz"],
        "n_qubits": [2, 3, 4, 5],
        "depths": [4, 8, 16],
        "seeds": [0, 1, 2],
        "min_abs_ideal": 0.25,
    }
    suite = generate_suite(config)
    assert len(suite) == 2 * 4 * 3 * 3
    for circuit, spec in suite:
        assert abs(_zn_expectation(circuit)) >= 0.25, spec.circuit_id


@pytest.mark.parametrize("bad", [-0.1, 1.0, 1.5, True, "0.3"])
def test_min_abs_ideal_invalid_values_raise(bad):
    with pytest.raises(ValueError, match="min_abs_ideal"):
        generate_suite(_screened_config(min_abs_ideal=bad))


def test_generate_suite_unknown_family_raises():
    config = _small_config()
    config["families"] = ["layered_random", "nope_circuit"]
    with pytest.raises(ValueError, match="nope_circuit"):
        generate_suite(config)


@pytest.mark.parametrize("key", ["families", "n_qubits", "depths", "seeds"])
def test_generate_suite_empty_list_raises(key):
    config = _small_config()
    config[key] = []
    with pytest.raises(ValueError):
        generate_suite(config)


@pytest.mark.parametrize("key", ["families", "n_qubits", "depths", "seeds"])
def test_generate_suite_missing_key_raises(key):
    config = _small_config()
    del config[key]
    with pytest.raises(ValueError):
        generate_suite(config)
