"""Unit tests for qemsel.mitigation.

Standalone by design: uses the conftest fakes (``tiny_circuit``,
``tiny_identity_circuit``, ``fake_executor``) plus a monkeypatched exact
statevector ``qemsel.ideal.ideal_expectation`` (that module belongs to
another builder and may still be a stub when these tests run).

Core invariant: with a PERFECT (noiseless statevector) executor, every
technique must return approximately the ideal value — mitigating nothing is
approximately the identity.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli, Statevector

import qemsel.ideal
from qemsel import mitigation
from qemsel.mitigation import (
    MitigationError,
    SHOT_MULTIPLIER,
    TECHNIQUES,
    apply_technique,
    shots_consumed,
)

BACKEND = "FakeManilaV2"  # metadata only — no noisy simulation in these tests
SHOTS = 256
SEED = 3

# CDR's perfect fit makes scipy warn that it cannot estimate the covariance.
CDR_FIT_WARNING = (
    "ignore:Covariance of the parameters could not be estimated"
)


def _statevector_expectation(circuit: QuantumCircuit, pauli: str) -> float:
    """Exact <pauli> (qemsel convention: pauli[i] acts on qubit i)."""
    circ = circuit.remove_final_measurements(inplace=False)
    return float(np.real(Statevector(circ).expectation_value(Pauli(pauli[::-1]))))


@pytest.fixture()
def patched_ideal(monkeypatch) -> dict:
    """Replace qemsel.ideal.ideal_expectation (possibly an unimplemented
    stub) with an exact statevector version. Returns a call counter."""
    calls = {"n": 0}

    def _ideal(circuit: QuantumCircuit, pauli: str) -> float:
        calls["n"] += 1
        return _statevector_expectation(circuit, pauli)

    monkeypatch.setattr(qemsel.ideal, "ideal_expectation", _ideal)
    return calls


@pytest.fixture()
def cdr_circuit() -> QuantumCircuit:
    """2-qubit circuit with plenty of non-Clifford rz/ry/rx content.

    CDR needs non-Clifford gates to build training circuits from; the ry/rx
    gates additionally exercise the CDR-internal basis translation.
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


def _counting(
    executor: Callable[[QuantumCircuit, str], float],
) -> tuple[Callable[[QuantumCircuit, str], float], list[str]]:
    """Wrap an executor, recording the pauli of every invocation."""
    calls: list[str] = []

    def _exec(circuit: QuantumCircuit, pauli: str) -> float:
        calls.append(pauli)
        return executor(circuit, pauli)

    return _exec, calls


#: The techniques that consume ONLY the passed-in executor ('raw_plus'
#: instead rebuilds its own via backends.make_executor — covered by the
#: dedicated raw_plus tests below).
PASSED_EXECUTOR_TECHNIQUES = ["raw", "zne", "cdr", "rem"]


@pytest.fixture()
def patched_make_executor(monkeypatch, fake_executor) -> dict:
    """Patch qemsel.backends.make_executor with a recording noiseless fake.

    raw_plus rebuilds its executor through backends.make_executor; unit
    tests must not spin up real Aer noise models, so this fixture returns a
    noiseless statevector executor of the SAME contract and records:
        'make_calls': list of (backend_name, shots, seed) make_executor got,
        'exec_paulis': paulis of every built-executor invocation,
        'closed': how many times close() was called on built executors.
    """
    record = {"make_calls": [], "exec_paulis": [], "closed": 0}

    def _fake_make_executor(backend_name, shots, seed):
        record["make_calls"].append((backend_name, shots, seed))

        def _exec(circuit: QuantumCircuit, pauli: str) -> float:
            record["exec_paulis"].append(pauli)
            return fake_executor(circuit, pauli)

        def _close() -> None:
            record["closed"] += 1

        _exec.close = _close
        return _exec

    monkeypatch.setattr("qemsel.backends.make_executor", _fake_make_executor)
    return record


# ---------------------------------------------------------------------------
# Constants and cost model
# ---------------------------------------------------------------------------


def test_techniques_canonical_order() -> None:
    assert TECHNIQUES == ["raw", "raw_plus", "zne", "cdr", "rem"]


def test_shot_multiplier_keys_and_derivation() -> None:
    assert set(SHOT_MULTIPLIER) == set(TECHNIQUES)
    assert SHOT_MULTIPLIER["raw"] == 1
    assert SHOT_MULTIPLIER["raw_plus"] == mitigation.RAW_PLUS_MULTIPLIER
    assert SHOT_MULTIPLIER["zne"] == len(mitigation.ZNE_SCALE_FACTORS)
    assert SHOT_MULTIPLIER["cdr"] == 1 + mitigation.CDR_NUM_TRAINING_CIRCUITS
    assert (
        SHOT_MULTIPLIER["rem"]
        == 1 + mitigation.REM_NUM_CALIBRATION_CIRCUITS
    )
    for value in SHOT_MULTIPLIER.values():
        assert isinstance(value, int) and value >= 1


def test_raw_plus_multiplier_equals_most_expensive_technique() -> None:
    """raw_plus is the EQUAL-budget baseline: its budget must equal the
    largest multiplier of any real technique (CDR's 11), so 'just take more
    shots' is compared at the most expensive technique's cost."""
    others = {t: m for t, m in SHOT_MULTIPLIER.items() if t != "raw_plus"}
    assert mitigation.RAW_PLUS_MULTIPLIER == max(others.values())
    assert mitigation.RAW_PLUS_MULTIPLIER == SHOT_MULTIPLIER["cdr"] == 11
    assert shots_consumed("raw_plus", 4096) == shots_consumed("cdr", 4096)


def test_rem_min_damping_floor_is_strict() -> None:
    """Reviewer item: 1e-6 let REM invert near-singular readout (e.g. 46%
    error on FakeLagosV2 q2), amplifying shot noise ~29x into the dataset.
    The floor must be at least 0.02 so those refuse loudly instead."""
    assert mitigation.REM_MIN_DAMPING >= 0.02


@pytest.mark.parametrize("name", TECHNIQUES)
def test_shots_consumed_sane_positive_ints(name: str) -> None:
    total = shots_consumed(name, 1024)
    assert isinstance(total, int)
    assert total == 1024 * SHOT_MULTIPLIER[name]
    assert total >= 1024


def test_shots_consumed_unknown_name_raises() -> None:
    with pytest.raises(ValueError):
        shots_consumed("nope", 1024)


def test_apply_technique_unknown_name_raises(
    tiny_circuit: QuantumCircuit, fake_executor
) -> None:
    with pytest.raises(ValueError):
        apply_technique(
            "nope", tiny_circuit, "ZZ", fake_executor, BACKEND, SHOTS, SEED
        )


# ---------------------------------------------------------------------------
# Core invariant: perfect executor => every technique returns ~ideal
# ---------------------------------------------------------------------------


@pytest.mark.filterwarnings(CDR_FIT_WARNING)
@pytest.mark.parametrize("name", TECHNIQUES)
def test_noiseless_executor_returns_ideal(
    name: str,
    cdr_circuit: QuantumCircuit,
    fake_executor,
    patched_ideal,
    patched_make_executor,
) -> None:
    ideal = _statevector_expectation(cdr_circuit, "ZZ")
    value = apply_technique(
        name, cdr_circuit, "ZZ", fake_executor, BACKEND, SHOTS, SEED
    )
    assert value == pytest.approx(ideal, abs=1e-6)


@pytest.mark.parametrize("name", ["raw", "raw_plus", "zne", "rem"])
@pytest.mark.parametrize("pauli,expected", [("ZZ", 1.0), ("XX", 1.0), ("ZI", 0.0)])
def test_noiseless_bell_expectations(
    name: str,
    pauli: str,
    expected: float,
    tiny_circuit: QuantumCircuit,
    fake_executor,
    patched_make_executor,
) -> None:
    value = apply_technique(
        name, tiny_circuit, pauli, fake_executor, BACKEND, SHOTS, SEED
    )
    assert value == pytest.approx(expected, abs=1e-9)


def test_noiseless_identity_circuit_zzz(
    tiny_identity_circuit: QuantumCircuit, fake_executor, patched_ideal
) -> None:
    # tiny_identity_circuit is FULLY CLIFFORD, so cdr must fail loudly
    # (classical-simulation guard) rather than return a fake perfect value.
    for name in ("raw", "zne", "rem"):
        value = apply_technique(
            name,
            tiny_identity_circuit,
            "ZZZ",
            fake_executor,
            BACKEND,
            SHOTS,
            SEED,
        )
        assert value == pytest.approx(1.0, abs=1e-9), name
    with pytest.raises(MitigationError):
        apply_technique(
            "cdr", tiny_identity_circuit, "ZZZ", fake_executor, BACKEND, SHOTS, SEED
        )


# ---------------------------------------------------------------------------
# Cost-model truthfulness: executor call counts match SHOT_MULTIPLIER
# ---------------------------------------------------------------------------


@pytest.mark.filterwarnings(CDR_FIT_WARNING)
@pytest.mark.parametrize("name", PASSED_EXECUTOR_TECHNIQUES)
def test_executor_call_count_matches_shot_multiplier(
    name: str, cdr_circuit: QuantumCircuit, fake_executor, patched_ideal
) -> None:
    counting_executor, calls = _counting(fake_executor)
    apply_technique(
        name, cdr_circuit, "ZZ", counting_executor, BACKEND, SHOTS, SEED
    )
    assert len(calls) == SHOT_MULTIPLIER[name]


# ---------------------------------------------------------------------------
# raw_plus: equal-budget baseline semantics
# ---------------------------------------------------------------------------


def test_raw_plus_builds_boosted_executor_and_skips_passed_one(
    cdr_circuit: QuantumCircuit, fake_executor, patched_make_executor
) -> None:
    """Cost-model truthfulness for raw_plus: exactly ONE execution at
    RAW_PLUS_MULTIPLIER * base shots on a freshly built executor; the
    passed (base-shots-bound) executor is never invoked — calling a seeded
    executor 11 times would return 11 identical values, i.e. a fake 11x."""
    counting_executor, passed_calls = _counting(fake_executor)
    value = apply_technique(
        "raw_plus", cdr_circuit, "ZZ", counting_executor, BACKEND, SHOTS, SEED
    )
    assert passed_calls == []  # passed executor unused
    assert patched_make_executor["make_calls"] == [
        (BACKEND, SHOTS * mitigation.RAW_PLUS_MULTIPLIER, SEED)
    ]
    assert patched_make_executor["exec_paulis"] == ["ZZ"]  # exactly 1 call
    ideal = _statevector_expectation(cdr_circuit, "ZZ")
    assert value == pytest.approx(ideal, abs=1e-9)


def test_raw_plus_closes_built_executor(
    cdr_circuit: QuantumCircuit, fake_executor, patched_make_executor
) -> None:
    apply_technique(
        "raw_plus", cdr_circuit, "ZZ", fake_executor, BACKEND, SHOTS, SEED
    )
    assert patched_make_executor["closed"] == 1


def test_raw_plus_make_executor_failure_wrapped(
    cdr_circuit: QuantumCircuit, fake_executor, monkeypatch
) -> None:
    def _boom(backend_name, shots, seed):
        raise RuntimeError("no such backend")

    monkeypatch.setattr("qemsel.backends.make_executor", _boom)
    with pytest.raises(MitigationError) as excinfo:
        apply_technique(
            "raw_plus", cdr_circuit, "ZZ", fake_executor, BACKEND, SHOTS, SEED
        )
    assert excinfo.value.technique == "raw_plus"


def test_raw_plus_execution_failure_wrapped_and_still_closed(
    cdr_circuit: QuantumCircuit, fake_executor, monkeypatch
) -> None:
    closed = {"n": 0}

    def _make(backend_name, shots, seed):
        def _exec(circuit, pauli):
            raise RuntimeError("boosted executor exploded")

        def _close():
            closed["n"] += 1

        _exec.close = _close
        return _exec

    monkeypatch.setattr("qemsel.backends.make_executor", _make)
    with pytest.raises(MitigationError) as excinfo:
        apply_technique(
            "raw_plus", cdr_circuit, "ZZ", fake_executor, BACKEND, SHOTS, SEED
        )
    assert excinfo.value.technique == "raw_plus"
    assert closed["n"] == 1  # close() must run even on failure (finally)


# ---------------------------------------------------------------------------
# Technique-specific behavior
# ---------------------------------------------------------------------------


@pytest.mark.filterwarnings(CDR_FIT_WARNING)
def test_cdr_uses_qemsel_ideal_simulator(
    cdr_circuit: QuantumCircuit, fake_executor, patched_ideal
) -> None:
    apply_technique(
        "cdr", cdr_circuit, "ZZ", fake_executor, BACKEND, SHOTS, SEED
    )
    # The near-Clifford training labels must come from qemsel.ideal.
    assert patched_ideal["n"] >= 1


def test_cdr_clifford_circuit_fails_loudly(
    tiny_circuit: QuantumCircuit, fake_executor, patched_ideal
) -> None:
    """Fail-loud guard (science review): mitiq would short-circuit a
    fully-Clifford circuit and return the classical simulator value — zero
    error by construction, no quantum execution. Recording that as a CDR
    result labeled 40% of the pre-fix dataset 'cdr wins' for classical
    simulability, so _apply_cdr must raise MitigationError instead."""
    counting_executor, calls = _counting(fake_executor)
    with pytest.raises(MitigationError) as excinfo:
        apply_technique(
            "cdr", tiny_circuit, "ZZ", counting_executor, BACKEND, SHOTS, SEED
        )
    assert excinfo.value.technique == "cdr"
    assert "Clifford" in str(excinfo.value)
    assert calls == []  # noisy executor never invoked


def test_cdr_degenerate_training_ideals_fail_loudly(
    fake_executor, patched_ideal
) -> None:
    """Fail-loud guard: a non-Clifford circuit whose near-Clifford training
    circuits all share the same ideal value gives a degenerate (constant)
    regression — classical simulation in disguise. GHZ + one t gate is such
    a circuit: the single non-Clifford t is replaced by a Clifford in every
    training circuit, so every training ideal of <ZZ> is exactly +1."""
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.t(0)
    counting_executor, calls = _counting(fake_executor)
    with pytest.raises(MitigationError) as excinfo:
        apply_technique(
            "cdr", qc, "ZZ", counting_executor, BACKEND, SHOTS, SEED
        )
    assert excinfo.value.technique == "cdr"
    assert "same ideal value" in str(excinfo.value)
    assert calls == []  # refused BEFORE spending any noisy executions


@pytest.mark.filterwarnings(CDR_FIT_WARNING)
def test_zne_and_cdr_deterministic_for_fixed_seed(
    cdr_circuit: QuantumCircuit, fake_executor, patched_ideal
) -> None:
    for name in ("zne", "cdr"):
        first = apply_technique(
            name, cdr_circuit, "ZZ", fake_executor, BACKEND, SHOTS, SEED
        )
        second = apply_technique(
            name, cdr_circuit, "ZZ", fake_executor, BACKEND, SHOTS, SEED
        )
        assert first == second, name


def test_rem_inverts_symmetric_readout_damping(
    tiny_circuit: QuantumCircuit,
    tiny_identity_circuit: QuantumCircuit,
    cdr_circuit: QuantumCircuit,
) -> None:
    """A per-qubit symmetric readout channel damps <Z_S> by d^|S|; REM's
    calibration measures exactly that factor, so the inversion is exact."""
    d = 0.82

    def damped_executor(circuit: QuantumCircuit, pauli: str) -> float:
        k = sum(c != "I" for c in pauli)
        return _statevector_expectation(circuit, pauli) * d**k

    cases = [
        (tiny_circuit, "ZZ"),  # even support, ideal +1
        (tiny_identity_circuit, "ZZZ"),  # odd support, ideal +1
        (cdr_circuit, "ZZ"),  # non-trivial ideal value
        (tiny_circuit, "ZI"),  # single-qubit support, ideal 0
    ]
    for circuit, pauli in cases:
        ideal = _statevector_expectation(circuit, pauli)
        raw = damped_executor(circuit, pauli)
        value = apply_technique(
            "rem", circuit, pauli, damped_executor, BACKEND, SHOTS, SEED
        )
        assert value == pytest.approx(ideal, abs=1e-9), (pauli, ideal)
        # And it actually changed something when the raw value was biased.
        if abs(ideal) > 1e-12:
            assert abs(value - ideal) < abs(raw - ideal)


def test_rem_identity_observable_returns_raw(
    tiny_circuit: QuantumCircuit, fake_executor
) -> None:
    counting_executor, calls = _counting(fake_executor)
    value = apply_technique(
        "rem", tiny_circuit, "II", counting_executor, BACKEND, SHOTS, SEED
    )
    assert value == pytest.approx(1.0, abs=1e-12)
    assert len(calls) == 1  # no calibration for an empty support


def test_rem_near_singular_damping_raises() -> None:
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)

    def dead_executor(circuit: QuantumCircuit, pauli: str) -> float:
        return 0.0  # readout so broken that calibration reads nothing

    with pytest.raises(MitigationError) as excinfo:
        apply_technique("rem", qc, "ZZ", dead_executor, BACKEND, SHOTS, SEED)
    assert excinfo.value.technique == "rem"


def _damped_executor(d: float) -> Callable[[QuantumCircuit, str], float]:
    """Executor with a symmetric per-qubit readout damping factor d."""

    def _exec(circuit: QuantumCircuit, pauli: str) -> float:
        k = sum(c != "I" for c in pauli)
        return _statevector_expectation(circuit, pauli) * d**k

    return _exec


def test_rem_refuses_damping_below_floor(tiny_circuit: QuantumCircuit) -> None:
    """Reviewer item (REM_MIN_DAMPING 1e-6 -> 0.02): a two-qubit support
    with per-qubit damping 0.1 gives total damping 0.01 < 0.02 — inverting
    it would amplify shot noise 100x, so REM must refuse loudly instead of
    recording amplified noise as a mitigated value."""
    with pytest.raises(MitigationError) as excinfo:
        apply_technique(
            "rem", tiny_circuit, "ZZ", _damped_executor(0.1), BACKEND, SHOTS, SEED
        )
    assert excinfo.value.technique == "rem"
    assert "too close to zero" in str(excinfo.value)


def test_rem_inverts_damping_just_above_floor(
    tiny_circuit: QuantumCircuit,
) -> None:
    # d = 0.3 on a 2-qubit support -> damping 0.09 >= 0.02: must still work.
    value = apply_technique(
        "rem", tiny_circuit, "ZZ", _damped_executor(0.3), BACKEND, SHOTS, SEED
    )
    assert value == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Error wrapping and non-mutation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", PASSED_EXECUTOR_TECHNIQUES)
def test_internal_failures_wrapped_in_mitigation_error(
    name: str, cdr_circuit: QuantumCircuit, patched_ideal
) -> None:
    # raw_plus never invokes the passed executor; its failure wrapping is
    # covered by the dedicated raw_plus tests above.
    class Boom(RuntimeError):
        pass

    def broken_executor(circuit: QuantumCircuit, pauli: str) -> float:
        raise Boom("executor exploded")

    with pytest.raises(MitigationError) as excinfo:
        apply_technique(
            name, cdr_circuit, "ZZ", broken_executor, BACKEND, SHOTS, SEED
        )
    assert excinfo.value.technique == name
    assert name in str(excinfo.value)
    assert isinstance(excinfo.value, RuntimeError)


@pytest.mark.filterwarnings(CDR_FIT_WARNING)
@pytest.mark.parametrize("name", TECHNIQUES)
def test_caller_circuit_never_mutated(
    name: str,
    cdr_circuit: QuantumCircuit,
    fake_executor,
    patched_ideal,
    patched_make_executor,
) -> None:
    reference = cdr_circuit.copy()
    apply_technique(
        name, cdr_circuit, "ZZ", fake_executor, BACKEND, SHOTS, SEED
    )
    assert cdr_circuit == reference
