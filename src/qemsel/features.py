"""Per-circuit feature extraction for the ML recommender.

Features are computed on the circuit AS GENERATED (pre-transpilation), plus
two backend-noise summary numbers pulled from
``qemsel.backends.get_backend_info`` (the only cross-module call allowed
here).

Clifford accounting
-------------------
A gate counts as Clifford when its name is in the fixed Clifford set
{id, h, x, y, z, s, sdg, sx, sxdg, cx, cz, swap}, or when it is an
rx/ry/rz/p rotation whose (bound, numeric) angle is an integer multiple of
pi/2 within an absolute tolerance of 1e-9. Everything else (t, tdg, u,
unbound-parameter rotations, 3+-qubit gates, ...) counts as non-Clifford.
"""

from __future__ import annotations

import math

from qiskit import QuantumCircuit

#: EXACT feature names, in canonical order. This order is the ML feature
#: vector order everywhere (model.py, recommend.py). Never reorder/rename
#: without updating both consumers.
FEATURE_NAMES: list[str] = [
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

#: V2 feature set: the FROZEN V1 list plus five ADDITIVE features, in this
#: exact order. Selected via ``extract_features(..., version=2)``; the V1
#: list and the version=1 path stay byte-identical.
#:
#: * log2_shots          — log2(base_shots). THE shots axis: without it the
#:                         selector is shots-blind and cannot learn the
#:                         Angle 3 help-harm boundary in (noise x shots).
#: * n_2q_layers         — number of critical-path layers containing a
#:                         2+-qubit gate: ``circuit.depth(filter_function=
#:                         lambda instr: len(instr.qubits) >= 2)``. Proxy
#:                         for the effective noise-amplification depth ZNE
#:                         folds (F4: ZNE wins concentrate at depth 8/16).
#: * entangling_density  — n_2q_gates / (n_qubits * depth); 0.0 when the
#:                         denominator is 0. Fraction of circuit area
#:                         occupied by entangling ops (2q errors dominate
#:                         eps).
#: * mean_rz_angle_dist  — mean over all angle-parameterized rotation gates
#:                         (rx/ry/rz/p, bound numeric params) of the
#:                         distance from the angle to the nearest integer
#:                         multiple of pi/2, normalized to [0, 1] by
#:                         dividing by pi/4 (i.e.
#:                         ``abs(math.remainder(angle, pi/2)) / (pi/4)``);
#:                         unbound-parameter rotations count as 1.0
#:                         (conservatively maximally non-Clifford, matching
#:                         _is_clifford); 0.0 when no such gates exist.
#:                         Continuous non-Cliffordness magnitude — the CDR
#:                         regressor-choice axis Angle 2 needs; also the
#:                         first feature that varies across seeds.
#: * backend_avg_1q_error — get_backend_info(...)['avg_1q_error']:
#:                         completes the backend noise triple; relevant in
#:                         the low-noise (Heron-like) dial-down regime where
#:                         1q-vs-2q error balance shifts.
FEATURE_NAMES_V2: list[str] = FEATURE_NAMES + [
    "log2_shots",
    "n_2q_layers",
    "entangling_density",
    "mean_rz_angle_dist",
    "backend_avg_1q_error",
]

#: Version -> feature-name list. The only valid versions. model.py bundles
#: record 'feature_version'; experiment configs select it via the
#: ``feature_version`` key (default 1).
FEATURE_NAMES_BY_VERSION: dict[int, list[str]] = {
    1: FEATURE_NAMES,
    2: FEATURE_NAMES_V2,
}

#: Gates that are Clifford by name alone (no angle inspection needed).
_CLIFFORD_GATES: frozenset[str] = frozenset(
    {"id", "h", "x", "y", "z", "s", "sdg", "sx", "sxdg", "cx", "cz", "swap"}
)

#: Parameterized rotations that are Clifford iff their angle is an integer
#: multiple of pi/2 (p(pi/2) == S, rz(pi) == Z up to phase, etc.).
_ANGLE_CHECKED_GATES: frozenset[str] = frozenset({"rx", "ry", "rz", "p"})

#: Instructions that are not gates and are excluded from all gate counts.
_NON_GATE_OPS: frozenset[str] = frozenset({"barrier", "delay"})

#: Absolute tolerance for the "angle is a multiple of pi/2" Clifford test.
_CLIFFORD_ANGLE_TOL: float = 1e-9


def _is_pi_half_multiple(angle: float) -> bool:
    """Return True iff ``angle`` is an integer multiple of pi/2 within 1e-9.

    Uses ``math.remainder`` (IEEE remainder, result in [-pi/4, pi/4]) so the
    check is exact-in-spirit for arbitrarily large angles.
    """
    return abs(math.remainder(angle, math.pi / 2.0)) <= _CLIFFORD_ANGLE_TOL


def _is_clifford(name: str, params: tuple) -> bool:
    """Classify a single gate (by name + numeric params) as Clifford or not.

    Unbound ParameterExpressions cannot be evaluated and are conservatively
    classified as non-Clifford.
    """
    if name in _CLIFFORD_GATES:
        return True
    if name in _ANGLE_CHECKED_GATES:
        try:
            angles = [float(p) for p in params]
        except (TypeError, ValueError):  # unbound ParameterExpression
            return False
        return all(_is_pi_half_multiple(a) for a in angles)
    return False


def _angle_clifford_distance(params: tuple) -> float:
    """Normalized distance of a rotation gate's angle(s) to the nearest pi/2.

    Used only by the V2 ``mean_rz_angle_dist`` feature (callers restrict this
    to gates in ``_ANGLE_CHECKED_GATES``). For bound numeric angles, returns
    the mean over the gate's params of
    ``abs(math.remainder(angle, pi/2)) / (pi/4)`` — 0.0 when the angle is an
    exact integer multiple of pi/2 (Clifford), rising to 1.0 at the pi/4
    midpoint (maximally non-Clifford). Unbound ParameterExpressions cannot be
    evaluated and are conservatively treated as 1.0 (mirrors ``_is_clifford``
    classifying them as non-Clifford).
    """
    try:
        angles = [float(p) for p in params]
    except (TypeError, ValueError):  # unbound ParameterExpression
        return 1.0
    if not angles:  # defensive; rx/ry/rz/p always carry one param
        return 0.0
    quarter = math.pi / 4.0
    dists = [abs(math.remainder(a, math.pi / 2.0)) / quarter for a in angles]
    return sum(dists) / len(dists)


def extract_features(
    circuit: QuantumCircuit,
    backend_name: str,
    *,
    version: int = 1,
    base_shots: int | float | None = None,
) -> dict[str, float]:
    """Extract the ML feature vector for one circuit on one backend.

    Args:
        circuit: QuantumCircuit WITHOUT final measurements, as produced by
            ``qemsel.circuits`` generators (features are computed on the
            circuit AS GENERATED, pre-transpilation). Never mutated.
        backend_name: one of ``qemsel.backends.BACKENDS``; backend features
            come from ``qemsel.backends.get_backend_info`` (the ONLY
            allowed cross-module call here).
        version: feature-set version (keyword-only; V2 addition). 1
            (default) = the frozen V1 behavior below, byte-identical —
            ``base_shots`` is IGNORED on this path. 2 = keys are EXACTLY
            ``FEATURE_NAMES_V2`` in order: the V1 values computed
            identically, plus the five V2 features documented at
            ``FEATURE_NAMES_V2`` (builder-features / B5 implements).
        base_shots: the unit's base shot budget (keyword-only; V2
            addition). REQUIRED (> 0) when version == 2 — it feeds
            ``log2_shots`` — and ignored when version == 1.

    Returns:
        dict whose keys are EXACTLY ``FEATURE_NAMES_BY_VERSION[version]``
        (in that order) and whose values are all plain Python floats.
        For version 1:
            n_qubits: circuit.num_qubits
            depth: circuit.depth()
            n_1q_gates: count of 1-qubit gates (barriers excluded)
            n_2q_gates: count of 2-qubit gates
            n_cnot: count of cx gates (subset of n_2q_gates)
            n_non_clifford: gates NOT in the Clifford set
                {h, x, y, z, s, sdg, sx, sxdg, cx, cz, swap} — an
                rx/ry/rz counts as Clifford only when its angle is an
                integer multiple of pi/2 within 1e-9; t/tdg are non-Clifford
            clifford_fraction: (total_gates - n_non_clifford) / total_gates,
                where total_gates = n_1q_gates + n_2q_gates; define as 1.0
                when total_gates == 0
            depth_per_qubit: depth / n_qubits
            backend_avg_2q_error: get_backend_info(...)['avg_2q_error']
            backend_avg_readout_error: get_backend_info(...)['avg_readout_error']
        For version 2 the same ten values, then the five features documented
        at ``FEATURE_NAMES_V2``:
            log2_shots: math.log2(base_shots)
            n_2q_layers: circuit.depth over 2+-qubit gates only
            entangling_density: n_2q_gates / (n_qubits * depth), 0.0 if denom 0
            mean_rz_angle_dist: mean per-rotation distance to nearest pi/2
                (normalized by pi/4; unbound params -> 1.0), 0.0 if no rotations
            backend_avg_1q_error: get_backend_info(...)['avg_1q_error']

    Raises:
        ValueError: if the circuit contains measurement instructions (the
            contract is measurement-free circuits), backend_name is
            unknown, ``version`` is not in ``FEATURE_NAMES_BY_VERSION``,
            or version == 2 with a missing/non-positive ``base_shots``.
    """
    if version not in FEATURE_NAMES_BY_VERSION:
        raise ValueError(
            f"unknown feature version {version!r}; known: "
            f"{sorted(FEATURE_NAMES_BY_VERSION)}"
        )
    if version == 2 and (base_shots is None or base_shots <= 0):
        raise ValueError(
            "feature version 2 requires a positive base_shots (it feeds "
            f"log2_shots); got base_shots={base_shots!r}"
        )
    # ---- shared V1 computation --------------------------------------------
    # version=1 stays byte-identical to the frozen surface; version=2 reuses
    # these exact V1 values and appends five additive features below.
    # Deferred import so features.py stays importable/testable even while
    # backends.py is a stub; tests monkeypatch qemsel.backends.get_backend_info.
    try:
        from qemsel import backends as _backends
    except ImportError as exc:  # pragma: no cover - defensive
        raise ImportError(
            "qemsel.backends is required for backend noise features"
        ) from exc

    n_1q = 0
    n_2q = 0
    n_multi = 0  # gates on 3+ qubits; none in our families, counted defensively
    n_cnot = 0
    n_non_clifford = 0
    angle_dist_sum = 0.0  # V2: sum of per-gate mean_rz distances
    angle_dist_count = 0  # V2: number of rx/ry/rz/p gates seen

    for instruction in circuit.data:
        op = instruction.operation
        name = op.name
        if name == "measure":
            raise ValueError(
                "extract_features requires a measurement-free circuit "
                "(qemsel convention: executors measure a copy), but found a "
                "'measure' instruction"
            )
        if name in _NON_GATE_OPS:
            continue
        nq = len(instruction.qubits)
        if nq == 1:
            n_1q += 1
        elif nq == 2:
            n_2q += 1
            if name == "cx":
                n_cnot += 1
        else:
            n_multi += 1
        params = tuple(op.params)
        if not _is_clifford(name, params):
            n_non_clifford += 1
        if name in _ANGLE_CHECKED_GATES:
            # V2 accumulation only; does not affect the V1 dict returned below.
            angle_dist_sum += _angle_clifford_distance(params)
            angle_dist_count += 1

    # total_gates = n_1q + n_2q per the contract; n_multi is added defensively
    # so clifford_fraction stays in [0, 1] even if a 3+-qubit gate sneaks in
    # (our circuit families never emit one).
    total_gates = n_1q + n_2q + n_multi
    if total_gates == 0:
        clifford_fraction = 1.0
    else:
        clifford_fraction = (total_gates - n_non_clifford) / total_gates

    n_qubits = circuit.num_qubits
    depth = circuit.depth()
    depth_per_qubit = (depth / n_qubits) if n_qubits > 0 else 0.0

    info = _backends.get_backend_info(backend_name)

    # Build in FEATURE_NAMES order — dicts preserve insertion order and this
    # order IS the ML feature-matrix column order.
    features: dict[str, float] = {
        "n_qubits": float(n_qubits),
        "depth": float(depth),
        "n_1q_gates": float(n_1q),
        "n_2q_gates": float(n_2q),
        "n_cnot": float(n_cnot),
        "n_non_clifford": float(n_non_clifford),
        "clifford_fraction": float(clifford_fraction),
        "depth_per_qubit": float(depth_per_qubit),
        "backend_avg_2q_error": float(info["avg_2q_error"]),
        "backend_avg_readout_error": float(info["avg_readout_error"]),
    }
    if version == 1:
        if list(features) != FEATURE_NAMES:
            raise RuntimeError(
                f"Feature order broken: expected {FEATURE_NAMES}, got {list(features)}"
            )
        return features

    # ---- version 2: append five additive features (order = FEATURE_NAMES_V2)
    # n_2q_layers: critical-path layers containing a 2+-qubit gate.
    n_2q_layers = circuit.depth(
        filter_function=lambda instr: len(instr.qubits) >= 2
    )
    # entangling_density: 2q-gate area fraction; 0.0 when denominator is 0.
    denom = n_qubits * depth
    entangling_density = (n_2q / denom) if denom > 0 else 0.0
    # mean_rz_angle_dist: mean per-gate non-Cliffordness; 0.0 when no rotations.
    mean_rz_angle_dist = (
        (angle_dist_sum / angle_dist_count) if angle_dist_count > 0 else 0.0
    )
    features.update(
        {
            "log2_shots": float(math.log2(float(base_shots))),
            "n_2q_layers": float(n_2q_layers),
            "entangling_density": float(entangling_density),
            "mean_rz_angle_dist": float(mean_rz_angle_dist),
            "backend_avg_1q_error": float(info["avg_1q_error"]),
        }
    )
    if list(features) != FEATURE_NAMES_V2:
        raise RuntimeError(
            f"Feature V2 order broken: expected {FEATURE_NAMES_V2}, got {list(features)}"
        )
    return features


def convert_circuit_to_graph(circuit: QuantumCircuit) -> dict:
    """Represent the quantum circuit as a graph structure for GNN training.

    Converts the circuit into a Directed Acyclic Graph (DAG) using Qiskit's
    built-in converter, then extracts node features (gate types) and the
    adjacency list (edge indices) in a format directly consumable by PyTorch
    Geometric or other GNN libraries.

    Returns:
        A dictionary with keys:
            - 'nodes': List of dicts, each detailing node ID, gate name, and qubits.
            - 'edge_index': List of tuples (source_node_id, target_node_id) representing connections.
    """
    from qiskit.converters import circuit_to_dag

    dag = circuit_to_dag(circuit)

    nodes = []
    node_to_id = {}

    # 1. Extract nodes and assign IDs
    for idx, node in enumerate(dag.op_nodes()):
        node_to_id[node] = idx
        nodes.append({
            "id": idx,
            "op": node.op.name,
            "qargs": [circuit.find_bit(q).index for q in node.qargs]
        })

    # 2. Extract edges representing qubit data flow between gates
    edge_index = []
    for u, v, edge_data in dag.edges():
        # Only care about flow between operation nodes (gates)
        if u in node_to_id and v in node_to_id:
            edge_index.append((node_to_id[u], node_to_id[v]))

    return {
        "nodes": nodes,
        "edge_index": edge_index
    }

