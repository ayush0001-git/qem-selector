"""Benchmark circuit generators.

Contract for EVERY generator in this module
-------------------------------------------
* Returned circuits contain NO final measurements and NO classical registers.
  Executors (``qemsel.backends.make_executor``) append measurements themselves.
* Use only gates that transpile cleanly to IBM fake backends:
  {h, x, y, z, s, sdg, t, tdg, sx, rx, ry, rz, cx, cz, barrier}.
* Deterministic given ``seed``: use ``numpy.random.default_rng(seed)`` locally,
  NEVER the global numpy random state.
* ``depth`` is the family's structural layer count (meaning defined per
  docstring), not necessarily the exact transpiled depth.
* Generators must be pure: no I/O, no global state mutation.

Implementation notes (builder-circuits):
* No barriers are emitted anywhere: mitiq's qiskit->cirq conversion and
  feature gate-counting are both simpler without them.
* All rng draws happen in a fixed order, so identical seeds reproduce
  byte-identical circuits.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from qiskit import QuantumCircuit

# Cross-module import through the public interface only; ideal.py imports
# nothing from qemsel, so no cycle. Accessed as a module attribute
# (``_ideal.ideal_expectation``) so tests can monkeypatch it.
from qemsel import ideal as _ideal


@dataclass
class CircuitSpec:
    """Metadata describing one generated benchmark circuit.

    Attributes:
        family: family name, one of ``FAMILIES`` keys (e.g. 'layered_random').
        n_qubits: number of qubits in the circuit.
        depth: structural depth parameter passed to the generator.
        seed: RNG seed passed to the generator (full reproducibility:
            ``FAMILIES[family](n_qubits, depth, seed, **params)`` recreates
            the identical circuit).
        params: extra keyword args passed to the generator ({} if none).
    """

    family: str
    n_qubits: int
    depth: int
    seed: int
    params: dict = field(default_factory=dict)

    @property
    def circuit_id(self) -> str:
        """Canonical unique id: '{family}_q{n_qubits}_d{depth}_s{seed}'."""
        return f"{self.family}_q{self.n_qubits}_d{self.depth}_s{self.seed}"


# --------------------------------------------------------------------------
# Private helpers
# --------------------------------------------------------------------------

_ROTATION_GATES: tuple[str, ...] = ("rx", "ry", "rz")
_CLIFFORD_1Q_GATES: tuple[str, ...] = ("h", "s", "sdg", "x", "z")
_HALF_PI: float = np.pi / 2.0


def _check_args(n_qubits: int, depth: int) -> None:
    """Validate the shared (n_qubits, depth) arguments of every generator."""
    if n_qubits < 1:
        raise ValueError(f"n_qubits must be >= 1, got {n_qubits}")
    if depth < 1:
        raise ValueError(f"depth must be >= 1, got {depth}")


def _brick_pairs(n_qubits: int, layer_index: int) -> list[tuple[int, int]]:
    """Non-overlapping adjacent (q, q+1) pairs for a brick-work CX layer.

    Even layers pair starting at qubit 0, odd layers starting at qubit 1,
    giving the alternating brick pattern. May be empty (e.g. odd layers when
    n_qubits == 2).
    """
    start = layer_index % 2
    return [(q, q + 1) for q in range(start, n_qubits - 1, 2)]


def _random_angle(rng: np.random.Generator) -> float:
    """Uniform angle in [0, 2*pi)."""
    return float(rng.uniform(0.0, 2.0 * np.pi))


def _non_clifford_angle(rng: np.random.Generator) -> float:
    """Random angle guaranteed NOT to be a multiple of pi/2 (tol 1e-6).

    Rejection sampling; terminates almost surely and is deterministic given
    the rng state.
    """
    while True:
        angle = _random_angle(rng)
        residue = angle % _HALF_PI
        if min(residue, _HALF_PI - residue) > 1e-6:
            return angle


# --------------------------------------------------------------------------
# Circuit families
# --------------------------------------------------------------------------


def layered_random(n_qubits: int, depth: int, seed: int) -> QuantumCircuit:
    """Brick-work random circuit.

    ``depth`` layers; each layer applies a random 1-qubit rotation
    (rx/ry/rz with angle ~ U[0, 2*pi)) to every qubit, then CX gates on a
    random non-overlapping pairing of adjacent qubits (alternate even/odd
    brick pattern across layers).

    Returns:
        QuantumCircuit with no measurements.
    """
    _check_args(n_qubits, depth)
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(
        n_qubits, name=f"layered_random_q{n_qubits}_d{depth}_s{seed}"
    )
    for layer in range(depth):
        for q in range(n_qubits):
            gate = _ROTATION_GATES[int(rng.integers(0, len(_ROTATION_GATES)))]
            getattr(qc, gate)(_random_angle(rng), q)
        for a, b in _brick_pairs(n_qubits, layer):
            qc.cx(a, b)
    return qc


def near_clifford(
    n_qubits: int, depth: int, seed: int, non_clifford_fraction: float = 0.15
) -> QuantumCircuit:
    """Random circuit that is mostly Clifford gates.

    ``depth`` layers of gates drawn from the Clifford set
    {h, s, sdg, x, z, cx}; each 1-qubit gate slot is replaced, with
    probability ``non_clifford_fraction``, by a T gate or an rz with a
    random non-multiple-of-pi/2 angle. CDR is expected to shine here.

    Returns:
        QuantumCircuit with no measurements.
    """
    _check_args(n_qubits, depth)
    if not 0.0 <= non_clifford_fraction <= 1.0:
        raise ValueError(
            "non_clifford_fraction must be in [0, 1], "
            f"got {non_clifford_fraction}"
        )
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(
        n_qubits, name=f"near_clifford_q{n_qubits}_d{depth}_s{seed}"
    )
    for layer in range(depth):
        for q in range(n_qubits):
            if rng.random() < non_clifford_fraction:
                # Non-Clifford slot: T gate or an rz at a generic angle.
                if rng.random() < 0.5:
                    qc.t(q)
                else:
                    qc.rz(_non_clifford_angle(rng), q)
            else:
                gate = _CLIFFORD_1Q_GATES[
                    int(rng.integers(0, len(_CLIFFORD_1Q_GATES)))
                ]
                getattr(qc, gate)(q)
        for a, b in _brick_pairs(n_qubits, layer):
            qc.cx(a, b)
    return qc


def _pad_cx_pair(qc: QuantumCircuit, rng: np.random.Generator) -> None:
    """Append CX;CX on a random adjacent pair (identity as an operator)."""
    q = int(rng.integers(0, qc.num_qubits - 1))
    qc.cx(q, q + 1)
    qc.cx(q, q + 1)


def _pad_rz_pair(qc: QuantumCircuit, rng: np.random.Generator) -> None:
    """Append rz(a);rz(-a) on a random qubit (identity as an operator)."""
    q = int(rng.integers(0, qc.num_qubits))
    angle = _random_angle(rng)
    qc.rz(angle, q)
    qc.rz(-angle, q)


def _pad_x_pair(qc: QuantumCircuit, rng: np.random.Generator) -> None:
    """Append X;X on a random qubit (identity as an operator)."""
    q = int(rng.integers(0, qc.num_qubits))
    qc.x(q)
    qc.x(q)


def _pad_h_pair(qc: QuantumCircuit, rng: np.random.Generator) -> None:
    """Append H;H on a random qubit (identity as an operator)."""
    q = int(rng.integers(0, qc.num_qubits))
    qc.h(q)
    qc.h(q)


def ghz_plus(n_qubits: int, depth: int, seed: int) -> QuantumCircuit:
    """GHZ preparation plus identity-equivalent padding to reach target depth.

    Standard GHZ prep (h on q0, CX chain q0->q1->...->q(n-1)); then, while the
    circuit depth is below ``depth``, append pairs of self-inverting gate
    layers (e.g. CX immediately followed by the same CX, or rz(a)/rz(-a))
    chosen with the rng, so the final STATE is still exactly GHZ but the
    circuit is deeper/noisier. ``seed`` controls the padding choices.

    Ideal expectations on the GHZ state: <Z...Z> = 1 for even n_qubits,
    0 for odd; use ``qemsel.ideal.ideal_expectation`` rather than assuming.

    Returns:
        QuantumCircuit with no measurements.
    """
    _check_args(n_qubits, depth)
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(n_qubits, name=f"ghz_plus_q{n_qubits}_d{depth}_s{seed}")
    qc.h(0)
    for q in range(n_qubits - 1):
        qc.cx(q, q + 1)
    padders: list[Callable[[QuantumCircuit, np.random.Generator], None]] = [
        _pad_rz_pair,
        _pad_x_pair,
        _pad_h_pair,
    ]
    if n_qubits >= 2:
        padders.append(_pad_cx_pair)
    while qc.depth() < depth:
        padders[int(rng.integers(0, len(padders)))](qc, rng)
    return qc


def hw_efficient_ansatz(n_qubits: int, depth: int, seed: int) -> QuantumCircuit:
    """Hardware-efficient variational ansatz with random parameters.

    ``depth`` blocks; each block = ry(theta) then rz(phi) on every qubit
    (angles ~ U[0, 2*pi) from the rng, BOUND numerically — no free
    qiskit Parameters in the returned circuit), followed by a linear CX
    entangling chain q0->q1->...->q(n-1). One final ry+rz layer after the
    last block.

    Returns:
        QuantumCircuit with no measurements and no unbound parameters.
    """
    _check_args(n_qubits, depth)
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(
        n_qubits, name=f"hw_efficient_ansatz_q{n_qubits}_d{depth}_s{seed}"
    )

    def _rotation_layer() -> None:
        for q in range(n_qubits):
            qc.ry(_random_angle(rng), q)
        for q in range(n_qubits):
            qc.rz(_random_angle(rng), q)

    for _ in range(depth):
        _rotation_layer()
        for q in range(n_qubits - 1):
            qc.cx(q, q + 1)
    _rotation_layer()
    return qc


def mirror_circuit(n_qubits: int, depth: int, seed: int) -> QuantumCircuit:
    """Mirror (Loschmidt echo) circuit: U followed by U-dagger.

    U is a layered_random-style circuit with ``max(1, depth // 2)`` layers;
    the returned circuit is U.compose(U.inverse()) so it equals the identity.
    Therefore the ideal expectation of 'Z' * n_qubits on input |0...0> is
    EXACTLY +1.0 — this family gives a known-answer benchmark at any size.

    Returns:
        QuantumCircuit with no measurements, logically equal to identity.
    """
    _check_args(n_qubits, depth)
    u = layered_random(n_qubits, max(1, depth // 2), seed)
    qc = u.compose(u.inverse())
    qc.name = f"mirror_circuit_q{n_qubits}_d{depth}_s{seed}"
    return qc


#: Registry mapping family name -> generator. All generators share the base
#: signature (n_qubits, depth, seed) plus optional keyword-only extras that
#: generate_suite passes through from config['params'][family].
FAMILIES: dict[str, Callable[..., QuantumCircuit]] = {
    "layered_random": layered_random,
    "near_clifford": near_clifford,
    "ghz_plus": ghz_plus,
    "hw_efficient_ansatz": hw_efficient_ansatz,
    "mirror_circuit": mirror_circuit,
}


#: Max deterministic rejection-sampling attempts per suite slot when the
#: config sets ``min_abs_ideal`` (see generate_suite). After this many
#: sub-seed bumps the best-so-far circuit is kept with a RuntimeWarning.
MIN_ABS_IDEAL_MAX_ATTEMPTS: int = 50

#: Sub-seed stride for rejection sampling: attempt k uses seed
#: ``base_seed + k * SUB_SEED_STRIDE``. A large prime so bumped seeds can
#: never collide with another config seed (config seeds are small ints like
#: 0..2) nor with another base seed's bumped sequence — colliding effective
#: seeds would emit byte-identical circuits under different circuit_ids
#: (duplicate rows with potentially conflicting labels).
SUB_SEED_STRIDE: int = 1_000_003

#: Families exempt from min_abs_ideal rejection sampling — families whose
#: ideal is provably seed-INDEPENDENT, so bumping the seed can never change
#: it: mirror_circuit's <Z..Z> is EXACTLY +1.0 by construction at every
#: seed, and ghz_plus's state is exactly GHZ regardless of the padding seed
#: (<Z^n> is +1 for even n and 0 for odd n, forever). For ghz_plus odd n
#: the experiment layer's per-family pauli override (ghz_plus: X, <X^n> =
#: +1 at every n) plus its min_abs_ideal screen is the real fix; sampling
#: here would only burn statevector work and emit noise warnings.
MIN_ABS_IDEAL_EXEMPT_FAMILIES: frozenset[str] = frozenset(
    {"mirror_circuit", "ghz_plus"}
)


def _sample_above_threshold(
    generator: Callable[..., QuantumCircuit],
    family: str,
    n: int,
    d: int,
    base_seed: int,
    family_params: dict,
    threshold: float,
) -> tuple[QuantumCircuit, int]:
    """Deterministically find a seed whose circuit has |<Z..Z>| >= threshold.

    Attempt k (k = 0 .. MIN_ABS_IDEAL_MAX_ATTEMPTS-1) generates the circuit
    at ``base_seed + k * SUB_SEED_STRIDE`` and accepts the first one whose
    exact ideal <Z^n> magnitude reaches ``threshold``. If none does, the
    best-so-far (largest |ideal|; earliest attempt on ties) is kept and a
    RuntimeWarning is emitted — the experiment layer's min_abs_ideal screen
    remains the final honest filter for such stragglers.

    NOTE: no early exit on repeated identical ideals — families with
    QUANTIZED ideals (near_clifford: Clifford-dominated circuits give
    <Z^n> in {0, ±2^-k, ±1}) can produce the same value several attempts
    in a row by chance without being seed-independent; the provably
    seed-independent families are handled by
    ``MIN_ABS_IDEAL_EXEMPT_FAMILIES`` instead.

    Returns:
        (circuit, effective_seed) — the CircuitSpec must record
        ``effective_seed`` so ``FAMILIES[family](n, d, spec.seed, **params)``
        still recreates the emitted circuit exactly.
    """
    pauli = "Z" * n
    best_circuit: QuantumCircuit | None = None
    best_seed = base_seed
    best_abs = -1.0
    for attempt in range(MIN_ABS_IDEAL_MAX_ATTEMPTS):
        eff_seed = base_seed + attempt * SUB_SEED_STRIDE
        circuit = generator(n, d, eff_seed, **family_params)
        abs_ideal = abs(float(_ideal.ideal_expectation(circuit, pauli)))
        if abs_ideal >= threshold:
            return circuit, eff_seed
        if abs_ideal > best_abs:
            best_circuit, best_seed, best_abs = circuit, eff_seed, abs_ideal
    warnings.warn(
        f"generate_suite: {family} n_qubits={n} depth={d} seed={base_seed}: "
        f"no sub-seed reached |<{pauli}>| >= {threshold} in "
        f"{MIN_ABS_IDEAL_MAX_ATTEMPTS} attempts; keeping best-so-far "
        f"(seed={best_seed}, |ideal|={best_abs:.3f}) — the experiment-level "
        "min_abs_ideal screen will drop it if it is still low-signal there",
        RuntimeWarning,
        stacklevel=3,
    )
    if best_circuit is None:
        raise RuntimeError("generate_suite: no valid circuit could be found.")
    return best_circuit, best_seed


def generate_suite(config: dict) -> list[tuple[QuantumCircuit, CircuitSpec]]:
    """Generate the full benchmark suite from a config dict.

    Config schema (all keys required unless noted)::

        {
          "families": ["layered_random", "mirror_circuit", ...],  # subset of FAMILIES
          "n_qubits": [2, 3, 4],
          "depths":   [4, 8, 16],
          "seeds":    [0, 1, 2],
          "params":   {"near_clifford": {"non_clifford_fraction": 0.2}},  # optional
          "min_abs_ideal": 0.25,  # optional (default 0.0 = off), see below
        }

    Behaviour:
    * Cartesian product families x n_qubits x depths x seeds, in that nesting
      order (family outermost, seed innermost) so output order is deterministic.
    * For each combo calls ``FAMILIES[family](n, d, s, **params.get(family, {}))``
      and pairs the circuit with a ``CircuitSpec`` recording exactly those args.
    * Raises ``ValueError`` on unknown family names, empty lists, or a
      ``min_abs_ideal`` outside [0, 1).

    min_abs_ideal (source-level signal guarantee, 2026-07-21):
    * When > 0, every emitted circuit of a NON-exempt family (see
      ``MIN_ABS_IDEAL_EXEMPT_FAMILIES``; mirror_circuit is always exactly
      +1) is rejection-sampled DETERMINISTICALLY so its exact ideal
      ``<Z * n_qubits>`` satisfies ``|ideal| >= min_abs_ideal``: attempt k
      regenerates the circuit at ``seed + k * SUB_SEED_STRIDE`` and the
      first passing attempt wins, capped at ``MIN_ABS_IDEAL_MAX_ATTEMPTS``
      attempts, after which the best-so-far circuit is kept with a
      RuntimeWarning.
    * This fixes the family-mix skew of downstream |ideal| screening AT THE
      SOURCE: previously ~38% of random-family units were screened out
      AFTER generation, leaving a Clifford-heavy suite.
    * The check is on the Z^n observable. Families measured with a
      different per-family pauli at the experiment layer (e.g. ghz_plus
      with X^n) should rely on that layer's screen; ghz_plus's ideal is
      seed-independent anyway, so sampling never changes it.
    * The accepted (possibly bumped) seed is recorded in ``CircuitSpec.seed``
      — full reproducibility (``FAMILIES[family](n, d, spec.seed, **params)``
      recreates the circuit) and unique circuit_ids are preserved.
    * Deterministic: identical configs emit byte-identical suites.

    Returns:
        List of (circuit, spec) tuples.
    """
    required_keys = ("families", "n_qubits", "depths", "seeds")
    for key in required_keys:
        if key not in config:
            raise ValueError(f"config missing required key: {key!r}")
        if not config[key]:
            raise ValueError(f"config[{key!r}] must be a non-empty list")
    unknown = [f for f in config["families"] if f not in FAMILIES]
    if unknown:
        raise ValueError(
            f"unknown circuit families: {unknown}; "
            f"valid families: {sorted(FAMILIES)}"
        )
    min_abs_ideal = config.get("min_abs_ideal", 0.0)
    if isinstance(min_abs_ideal, bool) or not isinstance(
        min_abs_ideal, (int, float)
    ):
        raise ValueError(
            f"config['min_abs_ideal'] must be a number in [0, 1), got "
            f"{min_abs_ideal!r}"
        )
    min_abs_ideal = float(min_abs_ideal)
    if not 0.0 <= min_abs_ideal < 1.0:
        raise ValueError(
            f"config['min_abs_ideal'] must be in [0, 1), got {min_abs_ideal!r}"
        )
    all_params: dict = config.get("params") or {}

    suite: list[tuple[QuantumCircuit, CircuitSpec]] = []
    for family in config["families"]:
        family_params = dict(all_params.get(family, {}))
        generator = FAMILIES[family]
        screened = (
            min_abs_ideal > 0.0
            and family not in MIN_ABS_IDEAL_EXEMPT_FAMILIES
        )
        for n in config["n_qubits"]:
            for d in config["depths"]:
                for s in config["seeds"]:
                    if screened:
                        circuit, eff_seed = _sample_above_threshold(
                            generator, family, n, d, s, family_params,
                            min_abs_ideal,
                        )
                    else:
                        circuit, eff_seed = generator(n, d, s, **family_params), s
                    spec = CircuitSpec(
                        family=family,
                        n_qubits=n,
                        depth=d,
                        seed=eff_seed,
                        params=dict(family_params),
                    )
                    suite.append((circuit, spec))
    return suite
