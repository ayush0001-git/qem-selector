"""Unit tests for the V2 additions to qemsel.mitigation (builder B1).

Covers the three new techniques ('zne_fr', 'cdr_ridge', 'cdr_rf'), the shared
``richardson_coefficients`` source of truth, the V2 constants/cost model, and
a CAPTURE-FIRST byte-identical regression pinning the pre-change values of the
frozen V1 techniques (raw/raw_plus/zne/cdr/rem) on a fixed circuit+seed.

Standalone by design: unit tests use the conftest fakes (``tiny_circuit``,
``tiny_identity_circuit``, ``fake_executor``) plus a monkeypatched exact
statevector ``qemsel.ideal.ideal_expectation`` and a monkeypatched
``qemsel.backends.make_executor`` (both cross-builder modules), so no noisy Aer
simulation is needed except in the explicitly ``@pytest.mark.slow`` regression.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli, Statevector

import qemsel.ideal
from qemsel import backends, mitigation
from qemsel.mitigation import (
    MitigationError,
    SHOT_MULTIPLIER,
    SHOT_MULTIPLIER_V2,
    TECHNIQUES,
    TECHNIQUES_V2,
    apply_technique,
    richardson_coefficients,
    shots_consumed,
)

BACKEND = "FakeManilaV2"  # metadata only except in the slow regression
BASE_SHOTS = 256
SEED = 3
PAULI = "ZZ"

CDR_FIT_WARNING = "ignore:Covariance of the parameters could not be estimated"

# ---------------------------------------------------------------------------
# Byte-identical reference values for the FROZEN V1 techniques, CAPTURED FIRST
# by running the pre-change code (scratchpad/capture_v1_ref.py) on
# capture_circuit() with FakeManilaV2 @ 256 shots, seed 3, pauli "ZZ". Any V2
# edit that perturbs a V1 dispatch path breaks these exact-equality asserts.
# ---------------------------------------------------------------------------
V1_REFERENCE_VALUES = {
    "raw": 0.203125,
    "raw_plus": 0.19176136363636365,
    "zne": 0.2187499999999998,
    "cdr": 0.16611249369922473,
    "rem": 0.2311111111111111,
}

# New-technique deterministic values captured AFTER implementation on the same
# (circuit, backend, shots, seed) — pins their reproducibility, not correctness.
V2_REFERENCE_VALUES = {
    "zne_fr": 0.296875,
    # Re-captured 2026-07-23 after the findings-applier switched cdr_ridge from
    # the over-regularized Ridge(alpha=1.0) (old value 0.7762758893745265 — the
    # review-Finding-3 bug, 3.6x worse than plain cdr) to RidgeCV. The new value
    # equals plain cdr's 0.166 on this circuit — exactly correct: RidgeCV's LOO
    # picks near-zero alpha and reproduces linear CDR (the Korolev anchor).
    "cdr_ridge": 0.16611846587957096,
    "cdr_rf": 0.6309239494731316,
}


def _sv_exp(circuit: QuantumCircuit, pauli: str) -> float:
    """Exact <pauli> (qemsel convention: pauli[i] acts on qubit i)."""
    circ = circuit.remove_final_measurements(inplace=False)
    return float(np.real(Statevector(circ).expectation_value(Pauli(pauli[::-1]))))


def capture_circuit() -> QuantumCircuit:
    """The circuit the V1/V2 reference values were captured on.

    Identical to the ``cdr_circuit`` fixture: 2q with non-Clifford rz/ry/rx
    content (so CDR + its sklearn variants pass the fully-Clifford and
    training-spread guards).
    """
    qc = QuantumCircuit(2)
    for _ in range(3):
        qc.h(0)
        qc.h(1)
        qc.rz(0.15, 0)
        qc.rz(0.2, 1)
        qc.h(0)
        qc.h(1)
        qc.cx(0, 1)
        qc.rz(0.25, 0)
        qc.rz(0.1, 1)
        qc.cx(0, 1)
    qc.ry(0.3, 0)
    qc.rx(0.7, 1)
    return qc


@pytest.fixture()
def cdr_circuit() -> QuantumCircuit:
    """Same non-Clifford 2q circuit as ``capture_circuit`` (fixture form)."""
    return capture_circuit()


def _counting(
    executor: Callable[[QuantumCircuit, str], float],
) -> tuple[Callable[[QuantumCircuit, str], float], list[str]]:
    """Wrap an executor, recording the pauli of every invocation."""
    calls: list[str] = []

    def _exec(circuit: QuantumCircuit, pauli: str) -> float:
        calls.append(pauli)
        return executor(circuit, pauli)

    return _exec, calls


@pytest.fixture()
def patched_ideal(monkeypatch) -> dict:
    """Replace qemsel.ideal.ideal_expectation with an exact statevector version
    and count calls (mirrors the V1 test-suite fixture)."""
    calls = {"n": 0}

    def _ideal(circuit: QuantumCircuit, pauli: str) -> float:
        calls["n"] += 1
        return _sv_exp(circuit, pauli)

    monkeypatch.setattr(qemsel.ideal, "ideal_expectation", _ideal)
    return calls


@pytest.fixture()
def recording_make_executor(monkeypatch, fake_executor) -> dict:
    """Patch backends.make_executor with a recording noiseless statevector fake.

    zne_fr rebuilds a per-level executor through backends.make_executor; unit
    tests must not spin up Aer noise models, so this returns a noiseless
    executor of the same contract and records:
        'make_calls': list of (backend_name, shots, seed) passed to make,
        'exec_paulis': paulis of every built-executor invocation,
        'closed': how many times close() was called on built executors.
    """
    record = {"make_calls": [], "exec_paulis": [], "closed": 0}

    def _fake_make(backend_name, shots, seed):
        record["make_calls"].append((backend_name, shots, seed))

        def _exec(circuit: QuantumCircuit, pauli: str) -> float:
            record["exec_paulis"].append(pauli)
            return fake_executor(circuit, pauli)

        def _close() -> None:
            record["closed"] += 1

        _exec.close = _close
        return _exec

    monkeypatch.setattr("qemsel.backends.make_executor", _fake_make)
    return record


# ===========================================================================
# V2 constants and cost model
# ===========================================================================


def test_techniques_v2_is_v1_plus_three_additive() -> None:
    assert TECHNIQUES_V2 == TECHNIQUES + ["zne_fr", "cdr_ridge", "cdr_rf"]
    # V1 surface is frozen and unchanged.
    assert TECHNIQUES == ["raw", "raw_plus", "zne", "cdr", "rem"]


def test_shot_multiplier_v2_is_truthful_superset() -> None:
    # Superset with identical values on the frozen V1 keys.
    for name, mult in SHOT_MULTIPLIER.items():
        assert SHOT_MULTIPLIER_V2[name] == mult
    # zne_fr is cost-neutral vs raw under equal_split (spends ONE base budget);
    # the sklearn-CDR variants cost 1 + N training executions like plain cdr.
    assert SHOT_MULTIPLIER_V2["zne_fr"] == (
        1
        if mitigation.ZNE_FR_SHOT_ALLOCATION == "equal_split"
        else len(mitigation.ZNE_FR_SCALE_FACTORS)
    )
    assert (
        SHOT_MULTIPLIER_V2["cdr_ridge"]
        == 1 + mitigation.CDR_SKLEARN_NUM_TRAINING_CIRCUITS
    )
    assert (
        SHOT_MULTIPLIER_V2["cdr_rf"]
        == 1 + mitigation.CDR_SKLEARN_NUM_TRAINING_CIRCUITS
    )
    for value in SHOT_MULTIPLIER_V2.values():
        assert isinstance(value, int) and value >= 1


@pytest.mark.parametrize("name", ["zne_fr", "cdr_ridge", "cdr_rf"])
def test_shots_consumed_new_techniques(name: str) -> None:
    total = shots_consumed(name, 4096)
    assert isinstance(total, int)
    assert total == 4096 * SHOT_MULTIPLIER_V2[name]


def test_zne_fr_scale_factors_are_the_two_point_rule() -> None:
    """Spike-retuned to the Scavino (1, 3) k=1 nodes so the zne_fr estimate and
    the qemsel.boundary theory share exactly these nodes (else the overlay is
    apples-to-oranges)."""
    assert mitigation.ZNE_FR_SCALE_FACTORS == (1.0, 3.0)
    assert mitigation.ZNE_FR_SHOT_ALLOCATION == "equal_split"
    assert mitigation.ZNE_FR_FOLD_METHOD == "global"


def test_cdr_sklearn_shares_cdr_training_settings() -> None:
    # The three cdr variants must differ ONLY in the regressor (Angle 2 control).
    assert (
        mitigation.CDR_SKLEARN_NUM_TRAINING_CIRCUITS
        == mitigation.CDR_NUM_TRAINING_CIRCUITS
    )


# ===========================================================================
# richardson_coefficients — shared source of truth (imported by boundary.py)
# ===========================================================================


def test_richardson_two_point_rule_is_three_halves_minus_half() -> None:
    assert richardson_coefficients((1.0, 3.0)) == pytest.approx((1.5, -0.5))


def test_richardson_three_point_rule_matches_hand_calc() -> None:
    # (1, 3, 5) -> (15/8, -5/4, 3/8) (spike-boundary.md §4).
    assert richardson_coefficients((1.0, 3.0, 5.0)) == pytest.approx(
        (1.875, -1.25, 0.375)
    )


def test_zne_fr_nodes_give_expected_coefficients() -> None:
    assert richardson_coefficients(mitigation.ZNE_FR_SCALE_FACTORS) == pytest.approx(
        (1.5, -0.5)
    )


@pytest.mark.parametrize(
    "nodes",
    [(1.0, 3.0), (1.0, 2.0, 3.0), (1.0, 3.0, 5.0), (1.0, 2.0, 4.0, 8.0)],
)
def test_richardson_satisfies_lagrange_constraints(nodes: tuple) -> None:
    """sum c_k == 1 and sum c_k * s_k^m == 0 for m = 1..len-1 (Lagrange at 0)."""
    c = np.asarray(richardson_coefficients(nodes))
    s = np.asarray(nodes)
    assert c.sum() == pytest.approx(1.0)
    for m in range(1, len(nodes)):
        assert float(np.dot(c, s**m)) == pytest.approx(0.0, abs=1e-9)


def test_richardson_matches_independent_lagrange_formula() -> None:
    """Cross-check against the spike's independent numpy implementation."""

    def spike_rc(lambdas):
        lam = np.asarray(lambdas, float)
        coeffs = np.empty_like(lam)
        for j in range(lam.size):
            others = np.delete(lam, j)
            coeffs[j] = np.prod(others / (others - lam[j]))
        return tuple(float(x) for x in coeffs)

    for nodes in [(1.0, 3.0), (1.0, 3.0, 5.0), (1.0, 2.5, 4.0)]:
        assert richardson_coefficients(nodes) == pytest.approx(spike_rc(nodes))


@pytest.mark.parametrize(
    "bad",
    [
        (1.0,),  # fewer than 2
        (),  # empty
        (1.0, 1.0),  # not distinct
        (0.5, 2.0),  # a node < 1.0
        (1.0, float("inf")),  # non-finite
    ],
)
def test_richardson_rejects_bad_nodes(bad: tuple) -> None:
    with pytest.raises(ValueError):
        richardson_coefficients(bad)


def test_richardson_returns_plain_float_tuple() -> None:
    c = richardson_coefficients((1.0, 3.0))
    assert isinstance(c, tuple)
    assert all(isinstance(x, float) for x in c)


# ===========================================================================
# zne_fr: fixed-Richardson, equal-split rebuilt executors, global folding
# ===========================================================================


@pytest.mark.parametrize("circuit_fn,pauli", [(None, "ZZ")])
def test_zne_fr_noiseless_returns_ideal(
    circuit_fn, pauli, cdr_circuit, fake_executor, recording_make_executor
) -> None:
    """1.5*E(scale1) - 0.5*E(scale3) == ideal on a noiseless executor (global
    folding preserves the logical action, so both levels read the ideal)."""
    passed, passed_calls = _counting(fake_executor)
    ideal = _sv_exp(cdr_circuit, pauli)
    value = apply_technique(
        "zne_fr", cdr_circuit, pauli, passed, BACKEND, BASE_SHOTS, SEED
    )
    assert value == pytest.approx(ideal, abs=1e-9)
    assert passed_calls == []  # the base-shots-bound passed executor is unused


def test_zne_fr_noiseless_bell_is_one(
    tiny_circuit, fake_executor, recording_make_executor
) -> None:
    value = apply_technique(
        "zne_fr", tiny_circuit, "ZZ", fake_executor, BACKEND, BASE_SHOTS, SEED
    )
    assert value == pytest.approx(1.0, abs=1e-9)


def test_zne_fr_rebuilds_split_budget_executors_and_closes_them(
    cdr_circuit, fake_executor, recording_make_executor
) -> None:
    """Cost-model truthfulness for zne_fr: one rebuilt executor PER level, each
    at base_shots // n_levels, same seed; total shots == base_shots (multiplier
    1); each rebuilt executor closed."""
    passed, passed_calls = _counting(fake_executor)
    apply_technique(
        "zne_fr", cdr_circuit, "ZZ", passed, BACKEND, BASE_SHOTS, SEED
    )
    n = len(mitigation.ZNE_FR_SCALE_FACTORS)
    level_shots = BASE_SHOTS // n
    assert recording_make_executor["make_calls"] == [
        (BACKEND, level_shots, SEED)
    ] * n
    assert recording_make_executor["exec_paulis"] == ["ZZ"] * n  # one per level
    assert recording_make_executor["closed"] == n
    assert passed_calls == []
    total_shots = sum(s for _b, s, _sd in recording_make_executor["make_calls"])
    assert total_shots == BASE_SHOTS
    assert total_shots == shots_consumed("zne_fr", BASE_SHOTS)


def test_zne_fr_applies_fixed_coefficients_in_level_order(
    monkeypatch, cdr_circuit, fake_executor
) -> None:
    """The estimate is sum_k c_k * E_k with the FIXED coefficients applied to
    the levels in ZNE_FR_SCALE_FACTORS order (level 0 executed first)."""
    preset = iter([10.0, 20.0])

    def _make(backend_name, shots, seed):
        def _exec(circuit, pauli):
            return next(preset)

        return _exec

    monkeypatch.setattr("qemsel.backends.make_executor", _make)
    coeffs = richardson_coefficients(mitigation.ZNE_FR_SCALE_FACTORS)
    value = apply_technique(
        "zne_fr", cdr_circuit, "ZZ", fake_executor, BACKEND, BASE_SHOTS, SEED
    )
    assert value == pytest.approx(coeffs[0] * 10.0 + coeffs[1] * 20.0)


def test_zne_fr_deterministic_for_fixed_seed(
    cdr_circuit, fake_executor, recording_make_executor
) -> None:
    first = apply_technique(
        "zne_fr", cdr_circuit, "ZZ", fake_executor, BACKEND, BASE_SHOTS, SEED
    )
    second = apply_technique(
        "zne_fr", cdr_circuit, "ZZ", fake_executor, BACKEND, BASE_SHOTS, SEED
    )
    assert first == second


def test_zne_fr_make_executor_failure_wrapped(
    cdr_circuit, fake_executor, monkeypatch
) -> None:
    def _boom(backend_name, shots, seed):
        raise RuntimeError("no such backend")

    monkeypatch.setattr("qemsel.backends.make_executor", _boom)
    with pytest.raises(MitigationError) as excinfo:
        apply_technique(
            "zne_fr", cdr_circuit, "ZZ", fake_executor, BACKEND, BASE_SHOTS, SEED
        )
    assert excinfo.value.technique == "zne_fr"


def test_zne_fr_closes_built_executor_even_on_failure(
    cdr_circuit, fake_executor, monkeypatch
) -> None:
    closed = {"n": 0}

    def _make(backend_name, shots, seed):
        def _exec(circuit, pauli):
            raise RuntimeError("level executor exploded")

        def _close():
            closed["n"] += 1

        _exec.close = _close
        return _exec

    monkeypatch.setattr("qemsel.backends.make_executor", _make)
    with pytest.raises(MitigationError) as excinfo:
        apply_technique(
            "zne_fr", cdr_circuit, "ZZ", fake_executor, BACKEND, BASE_SHOTS, SEED
        )
    assert excinfo.value.technique == "zne_fr"
    assert closed["n"] == 1  # close() runs in a finally even when exec raises


def test_zne_fr_too_few_shots_to_split_fails_loudly(
    cdr_circuit, fake_executor, recording_make_executor
) -> None:
    # base_shots=1 over 2 levels rounds the per-level budget to 0.
    with pytest.raises(MitigationError) as excinfo:
        apply_technique("zne_fr", cdr_circuit, "ZZ", fake_executor, BACKEND, 1, SEED)
    assert excinfo.value.technique == "zne_fr"


# ===========================================================================
# cdr_ridge / cdr_rf: Route B (generate_training_circuits + sklearn fit)
# ===========================================================================


@pytest.mark.parametrize("name", ["cdr_ridge", "cdr_rf"])
def test_cdr_sklearn_executor_call_count(
    name: str, cdr_circuit, fake_executor, patched_ideal
) -> None:
    """1 target + N training executions == SHOT_MULTIPLIER_V2[name]."""
    counting, calls = _counting(fake_executor)
    apply_technique(name, cdr_circuit, "ZZ", counting, BACKEND, BASE_SHOTS, SEED)
    assert len(calls) == SHOT_MULTIPLIER_V2[name]
    assert len(calls) == 1 + mitigation.CDR_SKLEARN_NUM_TRAINING_CIRCUITS


@pytest.mark.parametrize("name", ["cdr_ridge", "cdr_rf"])
def test_cdr_sklearn_uses_qemsel_ideal_for_training_labels(
    name: str, cdr_circuit, fake_executor, patched_ideal
) -> None:
    apply_technique(name, cdr_circuit, "ZZ", fake_executor, BACKEND, BASE_SHOTS, SEED)
    assert patched_ideal["n"] >= mitigation.CDR_SKLEARN_NUM_TRAINING_CIRCUITS


@pytest.mark.parametrize("name", ["cdr_ridge", "cdr_rf"])
def test_cdr_sklearn_clifford_circuit_fails_loudly(
    name: str, tiny_circuit, fake_executor, patched_ideal
) -> None:
    """Same fully-Clifford guard as _apply_cdr — no noisy execution spent."""
    counting, calls = _counting(fake_executor)
    with pytest.raises(MitigationError) as excinfo:
        apply_technique(name, tiny_circuit, "ZZ", counting, BACKEND, BASE_SHOTS, SEED)
    assert excinfo.value.technique == name
    assert "Clifford" in str(excinfo.value)
    assert calls == []


@pytest.mark.parametrize("name", ["cdr_ridge", "cdr_rf"])
def test_cdr_sklearn_degenerate_spread_fails_loudly(
    name: str, fake_executor, patched_ideal
) -> None:
    """GHZ + one t: every training circuit's ideal <ZZ> is +1 -> degenerate
    regression -> same fail-loud guard as _apply_cdr, before any noisy call."""
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.t(0)
    counting, calls = _counting(fake_executor)
    with pytest.raises(MitigationError) as excinfo:
        apply_technique(name, qc, "ZZ", counting, BACKEND, BASE_SHOTS, SEED)
    assert excinfo.value.technique == name
    assert "same ideal value" in str(excinfo.value)
    assert calls == []


@pytest.mark.parametrize("name", ["cdr_ridge", "cdr_rf"])
def test_cdr_sklearn_deterministic_for_fixed_seed(
    name: str, cdr_circuit, fake_executor, patched_ideal
) -> None:
    first = apply_technique(
        name, cdr_circuit, "ZZ", fake_executor, BACKEND, BASE_SHOTS, SEED
    )
    second = apply_technique(
        name, cdr_circuit, "ZZ", fake_executor, BACKEND, BASE_SHOTS, SEED
    )
    assert first == second


def test_cdr_sklearn_rejects_unknown_regressor(
    cdr_circuit, fake_executor, patched_ideal
) -> None:
    with pytest.raises(ValueError):
        mitigation._apply_cdr_sklearn(
            cdr_circuit, "ZZ", fake_executor, SEED, regressor="cdr_bogus"
        )


@pytest.mark.parametrize("name", ["cdr_ridge", "cdr_rf"])
def test_cdr_sklearn_internal_failure_wrapped(
    name: str, cdr_circuit, patched_ideal
) -> None:
    class Boom(RuntimeError):
        pass

    def broken(circuit, pauli):
        raise Boom("executor exploded")

    with pytest.raises(MitigationError) as excinfo:
        apply_technique(name, cdr_circuit, "ZZ", broken, BACKEND, BASE_SHOTS, SEED)
    assert excinfo.value.technique == name


# ===========================================================================
# Non-mutation across all V2 techniques
# ===========================================================================


@pytest.mark.parametrize("name", ["zne_fr", "cdr_ridge", "cdr_rf"])
def test_v2_techniques_never_mutate_caller_circuit(
    name: str, cdr_circuit, fake_executor, patched_ideal, recording_make_executor
) -> None:
    reference = cdr_circuit.copy()
    apply_technique(name, cdr_circuit, "ZZ", fake_executor, BACKEND, BASE_SHOTS, SEED)
    assert cdr_circuit == reference


# ===========================================================================
# CAPTURE-FIRST byte-identical regression (real Aer noise) — the frozen V1
# techniques must reproduce their pre-change values exactly, and the three new
# techniques must reproduce their captured deterministic values.
# ===========================================================================


@pytest.mark.slow
@pytest.mark.filterwarnings(CDR_FIT_WARNING)
def test_v1_techniques_byte_identical_on_real_backend() -> None:
    qc = capture_circuit()
    executor = backends.make_executor(BACKEND, BASE_SHOTS, SEED)
    for name, ref in V1_REFERENCE_VALUES.items():
        value = apply_technique(name, qc, PAULI, executor, BACKEND, BASE_SHOTS, SEED)
        assert value == ref, f"{name}: {value!r} != captured {ref!r}"


@pytest.mark.slow
@pytest.mark.filterwarnings(CDR_FIT_WARNING)
def test_v2_techniques_reproduce_captured_values_on_real_backend() -> None:
    qc = capture_circuit()
    executor = backends.make_executor(BACKEND, BASE_SHOTS, SEED)
    for name, ref in V2_REFERENCE_VALUES.items():
        value = apply_technique(name, qc, PAULI, executor, BACKEND, BASE_SHOTS, SEED)
        assert value == ref, f"{name}: {value!r} != captured {ref!r}"


@pytest.mark.slow
def test_zne_fr_is_cost_neutral_on_real_backend() -> None:
    """zne_fr spends the SAME total budget as one raw execution (multiplier 1):
    2 levels x (base//2) shots == base_shots."""
    assert shots_consumed("zne_fr", BASE_SHOTS) == shots_consumed("raw", BASE_SHOTS)
