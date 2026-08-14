"""Noisy backend access: noise summaries and expectation-value executors.

Pauli-string convention used EVERYWHERE in qemsel
------------------------------------------------
``pauli[i]`` acts on qubit ``i`` — index 0 of the string is qubit 0
(left-to-right = q0, q1, ...). This is the REVERSE of qiskit's
``quantum_info.Pauli`` label convention (where the RIGHTMOST character is
qubit 0). If you build a qiskit Pauli/SparsePauliOp from one of our strings,
reverse it first: ``Pauli(pauli[::-1])``. Likewise qiskit counts bitstrings
are little-endian (rightmost bit = q0) — be careful computing parities.

Noise-scaled backend variants: ``"<FakeName>@x<scale>"``
--------------------------------------------------------
``make_executor`` and ``get_backend_info`` also accept scaled variants of
the fake-backend names, e.g. ``"FakeManilaV2@x1.5"`` or
``"FakeLagosV2@x2.0"`` (grammar: exactly one ``@``, suffix ``x<float>``,
scale finite and > 0; anything else is a ValueError; ``ibm_*`` hardware
names accept NO suffix — noise scaling is simulation-only).

* **Plain names, and any name with scale exactly 1.0, use the verbatim
  ``AerSimulator.from_backend`` path** (byte-identical to the pre-scaling
  code, including its thermal-relaxation modeling) — this is the
  regression-critical baseline every dataset row so far was produced with.
* **For scale != 1.0** the executor keeps the SAME ``from_backend``
  simulator (identical coupling map, basis gates, gate directions, and
  therefore identical transpilation) but swaps in a synthetic noise model
  built from the device calibration (``backend.target``):

  - per-gate depolarizing error with
    ``p = min(scale * calibrated_gate_error, 0.9)`` attached to the exact
    ``(gate, qargs)`` entries the calibration stores (so routing/direction
    coverage matches the plain path), and
  - symmetric readout confusion per qubit with
    ``p01 = p10 = min(scale * calibrated_readout_error, 0.45)``. (The
    Target stores a single symmetric readout error number per qubit — the
    same number the plain ``from_backend`` model uses.)

  **This is a controlled noise-strength DIAL, not a claim of physical
  fidelity at scale != 1.** Thermal relaxation (T1/T2) is deliberately NOT
  added on the scaled path: the calibrated gate error already includes the
  relaxation contribution, so depolarizing at ``scale * total_error`` PLUS
  a separately scaled relaxation channel would double-count and make the
  dial super-linear in ``scale``. Coherent/non-Markovian errors are not
  modeled either (depolarizing is an incoherent proxy). Consequently the
  scaled path at a hypothetical scale of 1 is close to, but not identical
  to, the plain path — which is why scale == 1.0 always routes to the
  plain path instead.
* The caps (0.9 gate, 0.45 readout) apply ONLY on the scaled path — they
  are stability guards, not physics. Quirk: a calibration value already
  above a cap (FakeLagosV2 q2 stores 46.4% readout error) reads slightly
  LOWER at e.g. x1.5 (capped to 45%) than at x1.0 (uncapped 46.4%).
  CONSEQUENCE (fixer pass 2026-07-21): on such cap-saturated devices the
  dial COMPRESSES — FakeLagosV2's realized avg readout error is only
  ~1.28x plain at nominal x1.5 and ~1.44x at x2.0 (q2 flat at 0.45 for
  both; max_readout_error non-monotone in scale), while Manila/Jakarta
  scale exactly. Any "noise increased by <scale>" claim must quote the
  realized ``get_backend_info`` numbers, not the nominal suffix (report
  section 5 prints them per backend).
* ``get_backend_info`` on a scaled name reports the SCALED (and capped)
  per-entry averages, so the model features
  ``backend_avg_2q_error`` / ``backend_avg_readout_error`` reflect what
  the noise model actually applies.
* Determinism: the scaled model is built from static calibration data, so
  the same ``(name, shots, seed)`` triple gives identical results, across
  scales, exactly like the plain path.
"""

from __future__ import annotations

import math
from typing import Callable

from qiskit import QuantumCircuit, transpile

#: Fake backends verified to work with NoiseModel.from_backend + AerSimulator.
#: FakeLagosV2 stores extreme readout error (q0 16.9%, q1 13.6%, q2 46.4%!) —
#: the "REM should win" case. Noise-scaled variants of these names
#: ("<FakeName>@x<scale>", see module docstring) are accepted by
#: make_executor / get_backend_info but are NOT listed here.
BACKENDS: list[str] = [
    "FakeManilaV2",
    "FakeJakartaV2",
    "FakeLagosV2",
    "FakeSherbrooke",
]

#: V2 (builder-backends / B2 — INTERFACES.md section V2): the noise dial is
#: extended DOWN to cover the Heron-like clean regime the research grid
#: lacked (END_RESULT.md finding F7: the selector's one hardware miss sits
#: exactly there). ``parse_backend_name`` has ALWAYS accepted any finite
#: scale > 0, so "Fake...@x0.25" / "Fake...@x0.5" already parse; B2's job is
#: to VERIFY the whole scaled path at these scales (executor runs, scaled
#: get_backend_info numbers, feature flow-through) and to pin it with
#: regression tests: (a) raw |error| monotone over scales {0.25, 0.5, 1.0}
#: on at least one device (mirror of the existing upward-monotonicity test),
#: (b) get_backend_info averages scale linearly below 1.0 (caps never bind
#: there), (c) the plain-name path stays byte-identical. Same grammar, same
#: caps, same determinism guarantees as the upward dial — the module
#: docstring's "controlled dial, not physics" caveat applies unchanged (at
#: scale < 1 the synthetic depolarizing+readout model REPLACES the richer
#: composite x1.0 channels, so the noise-character caveat is if anything
#: stronger; the report's realized-rates table already covers it).
LOW_NOISE_SCALES: tuple[float, ...] = (0.25, 0.5)

#: Target operation names that are neither gates nor measurements — excluded
#: from the 1-qubit gate-error average in get_backend_info.
_NON_GATE_OPS = {"measure", "reset", "delay", "barrier", "id"}

#: Caps applied on the SCALED noise path only (module docstring): a scaled
#: per-gate depolarizing probability never exceeds 0.9 and a scaled per-qubit
#: readout flip probability never exceeds 0.45. Stability guards, not physics.
_SCALED_GATE_ERROR_CAP: float = 0.9
_SCALED_READOUT_ERROR_CAP: float = 0.45

#: Cache for get_backend_info results, keyed by backend name. Populated
#: lazily; get_backend_info returns a fresh copy so callers cannot corrupt it.
_INFO_CACHE: dict[str, dict] = {}


def _validate_pauli(pauli: str, n_qubits: int) -> None:
    """Raise ValueError unless ``pauli`` is a valid string for ``n_qubits``.

    Valid = length equals ``n_qubits`` and characters drawn from {I, X, Y, Z}
    (qemsel convention: ``pauli[i]`` acts on qubit i).
    """
    if len(pauli) != n_qubits:
        raise ValueError(
            f"pauli length {len(pauli)} != circuit num_qubits {n_qubits}"
        )
    invalid = set(pauli) - set("IXYZ")
    if invalid:
        raise ValueError(
            f"invalid pauli characters {sorted(invalid)!r} in {pauli!r}; "
            "allowed: I, X, Y, Z"
        )


def _make_fake_backend(name: str):
    """Instantiate a fake backend class from qiskit_ibm_runtime.fake_provider.

    Raises:
        ValueError: if ``name`` is not in ``BACKENDS``.
    """
    if name not in BACKENDS:
        raise ValueError(f"unknown backend {name!r}; choose one of {BACKENDS}")
    from qiskit_ibm_runtime import fake_provider

    return getattr(fake_provider, name)()


def parse_backend_name(name: str) -> tuple[str, float]:
    """Split an optional noise-scale suffix off a backend name.

    Grammar (module docstring): ``"<base>@x<scale>"`` with exactly one
    ``@`` and ``<scale>`` a finite float > 0. A name without ``@`` is a
    plain name at scale 1.0. The base name is NOT validated against
    ``BACKENDS`` here — that stays with the consumers (so ``ibm_*``
    hardware names keep flowing through the dispatch seam unchanged).

    Examples:
        ``"FakeManilaV2"``       -> ``("FakeManilaV2", 1.0)``
        ``"FakeManilaV2@x1.5"``  -> ``("FakeManilaV2", 1.5)``
        ``"FakeLagosV2@x2.0"``   -> ``("FakeLagosV2", 2.0)``

    Raises:
        ValueError: malformed suffix (missing ``x``, non-numeric, zero,
            negative, non-finite, more than one ``@``, empty base) or a
            scale suffix on an ``ibm_*`` hardware name (noise scaling is
            simulation-only — NEVER a real-device operation).
    """
    if "@" not in name:
        return name, 1.0
    base, _sep, suffix = name.partition("@")
    if "@" in suffix:
        raise ValueError(
            f"backend name {name!r} has more than one '@'; expected "
            "'<FakeName>@x<scale>', e.g. 'FakeManilaV2@x1.5'"
        )
    if not base:
        raise ValueError(f"backend name {name!r} has an empty base name")
    if len(suffix) < 2 or not suffix.startswith("x"):
        raise ValueError(
            f"bad noise-scale suffix {'@' + suffix!r} in {name!r}; expected "
            "'@x<scale>', e.g. 'FakeManilaV2@x1.5'"
        )
    try:
        scale = float(suffix[1:])
    except ValueError:
        raise ValueError(
            f"noise-scale suffix {'@' + suffix!r} in {name!r} is not a "
            "number; expected '@x<scale>', e.g. 'FakeManilaV2@x1.5'"
        ) from None
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError(
            f"noise scale must be a finite number > 0, got {scale!r} "
            f"(from {name!r})"
        )
    if base.startswith("ibm_"):
        raise ValueError(
            f"noise scaling is simulation-only; refusing scaled name "
            f"{name!r} on real hardware backend {base!r}"
        )
    return base, scale


def _collect_target_errors(target) -> tuple[list, list]:
    """Walk a qiskit Target once; return ``(gate_entries, readout_entries)``.

    ``gate_entries``: list of ``(op_name, qargs, error)`` for 1- and
    2-qubit gate operations (skips ``_NON_GATE_OPS``, None qargs, and
    None/NaN errors). ``readout_entries``: list of ``(qargs, error)`` for
    the ``measure`` operation. Iteration order is the Target's own —
    shared by get_backend_info (plain AND scaled averages, so the plain
    arithmetic is bit-identical to the pre-refactor loop) and by
    ``_build_scaled_noise_model``.
    """
    gate_entries: list[tuple[str, tuple, float]] = []
    readout_entries: list[tuple[tuple, float]] = []
    for op_name in target.operation_names:
        props_map = target.get(op_name)  # {qargs: InstructionProperties|None}
        if not props_map:
            continue
        for qargs, props in props_map.items():
            if props is None or getattr(props, "error", None) is None:
                continue
            err = float(props.error)
            if math.isnan(err):
                continue
            if op_name == "measure":
                readout_entries.append((qargs, err))
            elif op_name in _NON_GATE_OPS or qargs is None:
                continue
            elif len(qargs) in (1, 2):
                gate_entries.append((op_name, qargs, err))
    return gate_entries, readout_entries


def _build_scaled_noise_model(backend, scale: float):
    """Synthetic noise model for the scaled path (module docstring).

    Per-gate depolarizing with ``p = min(scale * error, 0.9)`` on the exact
    calibrated ``(gate, qargs)`` entries, plus symmetric readout confusion
    ``p = min(scale * error, 0.45)`` per qubit. No thermal relaxation — a
    deliberate, documented approximation (the calibrated gate error already
    contains the relaxation contribution; adding a separately scaled
    relaxation channel would double-count). Deterministic: built from
    static calibration data only.
    """
    from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error

    gate_entries, readout_entries = _collect_target_errors(backend.target)
    noise_model = NoiseModel(
        basis_gates=sorted({op_name for op_name, _q, _e in gate_entries})
    )
    for op_name, qargs, err in gate_entries:
        if err <= 0.0:
            continue  # zero calibrated error (e.g. virtual rz) — nothing to scale
        p = min(scale * err, _SCALED_GATE_ERROR_CAP)
        noise_model.add_quantum_error(
            depolarizing_error(p, len(qargs)), op_name, list(qargs)
        )
    for qargs, err in readout_entries:
        if err <= 0.0:
            continue
        p = min(scale * err, _SCALED_READOUT_ERROR_CAP)
        noise_model.add_readout_error(
            ReadoutError([[1.0 - p, p], [p, 1.0 - p]]), list(qargs)
        )
    return noise_model


def expectation_from_counts(counts: dict[str, int], pauli: str) -> float:
    """Z-basis Pauli expectation value from a qiskit counts dictionary.

    ENDIANNESS — the one place everyone gets bitten: qiskit counts keys are
    LITTLE-ENDIAN bitstrings, i.e. the RIGHTMOST character of the key is
    qubit 0, while the qemsel pauli convention is ``pauli[i]`` acts on
    qubit i (LEFTMOST character = qubit 0). So for key ``'01'``:

    * qubit 0 measured ``1`` (rightmost bit), qubit 1 measured ``0``.
    * pauli ``'ZI'`` (Z on q0) -> eigenvalue (-1)**1 = -1.
    * pauli ``'IZ'`` (Z on q1) -> eigenvalue (-1)**0 = +1.

    We reverse each key once (``key[::-1]``) so that ``bits[i]`` is qubit i,
    then compute the parity of the measured bits on the non-'I' support of
    ``pauli``. Expectation = sum over keys of (+1 for even parity, -1 for
    odd) weighted by frequency.

    LIMITATION: this helper interprets every non-'I' character as a Z-basis
    measurement of that qubit — it is only meaningful on its own for pauli
    strings over {Z, I}. X/Y observables are supported by the executor from
    :func:`make_executor`, which rotates those qubits into the Z basis
    BEFORE measuring and then calls this helper (X ~ h, Y ~ sdg; h).

    Args:
        counts: qiskit-style mapping bitstring -> shot count. Spaces (from
            multiple classical registers) are stripped. Every key must have
            at least ``len(pauli)`` bits after stripping.
        pauli: qemsel-convention Pauli string; 'I' positions are ignored.

    Returns:
        float in [-1, +1]. For an all-'I' pauli (empty support) returns 1.0.

    Raises:
        ValueError: if counts is empty / all-zero, or a key is too short.
    """
    support = [i for i, p in enumerate(pauli) if p != "I"]
    if not support:
        return 1.0
    signed = 0
    total = 0
    for key, n_shots in counts.items():
        bits = key.replace(" ", "")[::-1]  # bits[i] = measured value of qubit i
        if len(bits) < len(pauli):
            raise ValueError(
                f"counts key {key!r} has {len(bits)} bits < pauli length "
                f"{len(pauli)}"
            )
        parity = sum(int(bits[i]) for i in support) % 2
        signed += n_shots if parity == 0 else -n_shots
        total += n_shots
    if total <= 0:
        raise ValueError("counts dictionary is empty or has zero total shots")
    return signed / total


def get_backend_info(name: str) -> dict:
    """Summarize the noise characteristics of a fake backend.

    Args:
        name: one of ``BACKENDS`` (class name in
            ``qiskit_ibm_runtime.fake_provider``; note FakeSherbrooke has no
            V2 suffix but is a V2 backend), optionally with a noise-scale
            suffix ``@x<scale>`` (module docstring), e.g.
            ``"FakeManilaV2@x1.5"``.

    Returns:
        dict with EXACTLY these keys (all derived from ``backend.target``):
            'name'              (str)   echo of ``name`` (suffix included)
            'n_qubits'          (int)
            'avg_1q_error'      (float) mean gate error over 1-qubit gate entries
            'avg_2q_error'      (float) mean gate error over 2-qubit gate entries
            'avg_readout_error' (float) mean measure error over qubits
            'max_readout_error' (float) worst single-qubit measure error

        For a scaled name (scale != 1.0) every error number is the mean/max
        over the PER-ENTRY scaled-and-capped values
        (``min(scale * error, cap)``, caps 0.9 gate / 0.45 readout) — i.e.
        exactly what the scaled noise model applies, so the two
        ``backend_*`` model features flow through features.py scaled.
        Scale 1.0 (plain name or ``@x1.0`` suffix) returns the UNSCALED,
        UNCAPPED calibration averages (pre-scaling behavior, verbatim).

    Raises:
        ValueError: if the base name is not in ``BACKENDS`` or the scale
            suffix is malformed (see :func:`parse_backend_name`).

    Notes:
        Results are cached per full name (features.py calls this for every
        circuit); a copy of the cached dict is returned so callers cannot
        mutate the cache. Target entries whose error is None/NaN are skipped
        when averaging. If a category has no valid entries at all its
        average is NaN (does not happen for the verified BACKENDS).
    """
    base_name, scale = parse_backend_name(name)

    # Dispatch seam: real IBM devices (names starting 'ibm_') are summarized
    # from the LIVE backend target by qemsel.hardware (needs credentials in
    # configs/hardware.yaml). Imported locally to avoid an import cycle and
    # to keep the fake-backend path unchanged. parse_backend_name has
    # already rejected any scale suffix on an ibm_* name.
    if base_name.startswith("ibm_"):
        from qemsel import hardware as _hardware

        return _hardware.get_real_backend_info(base_name)

    if scale == 1.0 and base_name != name:
        # "@x1.0" suffix: plain numbers verbatim, echoing the full name.
        info = get_backend_info(base_name)
        info["name"] = name
        return info

    if name in _INFO_CACHE:
        return dict(_INFO_CACHE[name])

    backend = _make_fake_backend(base_name)
    target = backend.target

    gate_entries, readout_entries = _collect_target_errors(target)
    if scale == 1.0:
        one_q_errors = [e for _op, qargs, e in gate_entries if len(qargs) == 1]
        two_q_errors = [e for _op, qargs, e in gate_entries if len(qargs) == 2]
        readout_errors = [e for _q, e in readout_entries]
    else:
        one_q_errors = [
            min(scale * e, _SCALED_GATE_ERROR_CAP)
            for _op, qargs, e in gate_entries
            if len(qargs) == 1
        ]
        two_q_errors = [
            min(scale * e, _SCALED_GATE_ERROR_CAP)
            for _op, qargs, e in gate_entries
            if len(qargs) == 2
        ]
        readout_errors = [
            min(scale * e, _SCALED_READOUT_ERROR_CAP)
            for _q, e in readout_entries
        ]

    def _mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else float("nan")

    info = {
        "name": name,
        "n_qubits": int(target.num_qubits),
        "avg_1q_error": _mean(one_q_errors),
        "avg_2q_error": _mean(two_q_errors),
        "avg_readout_error": _mean(readout_errors),
        "max_readout_error": max(readout_errors) if readout_errors else float("nan"),
    }
    _INFO_CACHE[name] = info
    return dict(info)


def make_executor(
    backend_name: str, shots: int, seed: int
) -> Callable[[QuantumCircuit, str], float]:
    """Build a noisy expectation-value executor for a fake backend.

    The returned callable has signature ``executor(circuit, pauli) -> float``:

    * ``circuit``: QuantumCircuit WITHOUT final measurements (never mutated —
      the executor works on a copy).
    * ``pauli``: Pauli string with ``len(pauli) == circuit.num_qubits`` in
      the qemsel convention (``pauli[i]`` acts on qubit i; see module
      docstring). Characters from {I, X, Y, Z}; X/Y qubits are rotated into
      the Z basis before measurement (X: h, Y: sdg then h).
    * Returns the shot-estimated expectation value <pauli> in [-1, +1].

    Behaviour:
    1. Copies the circuit, applies the basis changes, then ``measure_all()``.
    2. Uses an AerSimulator built ONCE (at make_executor time) with
       ``AerSimulator.from_backend(<fake backend>)`` — NOT a bare
       ``AerSimulator(noise_model=...)``. The from_backend simulator carries
       the device's COUPLING MAP, basis gates and gate directions, so the
       per-call transpile routes non-adjacent 2-qubit gates onto real device
       edges and direction-fixes ECR. This matters: Aer applies a
       noise-model error only on exact (gate, qargs) matches, so on an
       all-to-all simulator, cx on a non-coupled pair (e.g. (2,3)/(3,4) on
       the H-topology FakeLagosV2/FakeJakartaV2) or a wrong-direction ecr
       (FakeSherbrooke stores one direction per edge) would execute with
       ZERO noise — silently under-noising the dataset (review finding,
       2026-07-21). Routing SWAP overhead is device truth, not a bug.
       Transpile uses ``optimization_level=0`` (MANDATORY so mitiq's folded
       gates survive) and no ``basis_gates=`` kwarg (qiskit 2.5 UserWarning;
       see PROJECT_STATE.md).
    3. Runs with ``shots`` shots, ``seed_simulator=seed``,
       ``seed_transpiler=seed`` — fully reproducible: the same (circuit,
       pauli) pair always returns the same value.
    4. Expectation computed from counts by :func:`expectation_from_counts`
       (little-endian parity on the non-'I' support; 'I' qubits are still
       measured but ignored). ``measure_all()`` happens BEFORE transpile,
       so counts bits always index LOGICAL qubits even after routing /
       padding to the device width.
    * All-'I' pauli returns 1.0 without simulating.
    * Raises ValueError on length mismatch, invalid characters, or a
      circuit WIDER than the backend (extra qubits would simulate with no
      noise at all — silent wrong data).

    Noise-scaled variants (module docstring): ``backend_name`` may carry a
    ``@x<scale>`` suffix, e.g. ``"FakeManilaV2@x1.5"``. Scale 1.0 (plain
    name or ``@x1.0``) runs the verbatim ``from_backend`` path below —
    byte-identical to the pre-scaling code. Scale != 1.0 keeps the SAME
    ``from_backend`` simulator (identical coupling map / basis gates /
    transpilation) but swaps in the synthetic scaled noise model from
    ``_build_scaled_noise_model`` (depolarizing + readout only — a noise
    DIAL, not physical fidelity; see module docstring for the honest
    approximation statement). Determinism holds across scales: same
    ``(name, shots, seed)`` -> identical results.

    Args:
        backend_name: one of ``BACKENDS``, optionally with ``@x<scale>``.
        shots: shots per execution.
        seed: simulator + transpiler seed for reproducibility.

    Returns:
        executor callable as specified above.

    Raises:
        ValueError: if the base name is not in ``BACKENDS`` or the scale
            suffix is malformed (see :func:`parse_backend_name`).
    """
    base_name, scale = parse_backend_name(backend_name)

    # Dispatch seam: names starting 'ibm_' run on REAL hardware via
    # qemsel.hardware (same executor contract; credentials + confirmation
    # gates enforced there / in experiment._validate_config). Local import
    # avoids an import cycle and leaves the fake path unchanged.
    # parse_backend_name has already rejected any scale suffix on ibm_*.
    if base_name.startswith("ibm_"):
        from qemsel import hardware as _hardware

        return _hardware.make_real_executor(base_name, shots, seed)

    from qiskit_aer import AerSimulator

    # Heavy objects built exactly once and closed over by the executor.
    # from_backend keeps noise model + coupling map + gate directions.
    backend = _make_fake_backend(base_name)
    simulator = AerSimulator.from_backend(backend)

    # Enable GPU acceleration if available on the host machine
    try:
        if "GPU" in AerSimulator().available_devices():
            simulator.set_options(device="GPU")
    except Exception:
        pass

    if scale != 1.0:
        # Same simulator (so transpilation is identical across scales);
        # only the noise model option is replaced.
        simulator.set_options(noise_model=_build_scaled_noise_model(backend, scale))
    backend_qubits = int(backend.num_qubits)

    def executor(circuit: QuantumCircuit, pauli: str) -> float:
        """Noisy expectation of <pauli> on ``circuit`` (qemsel convention)."""
        _validate_pauli(pauli, circuit.num_qubits)
        if circuit.num_qubits > backend_qubits:
            raise ValueError(
                f"circuit has {circuit.num_qubits} qubits but backend "
                f"{backend_name!r} has only {backend_qubits}; qubits beyond "
                "the device would simulate with NO noise (silent wrong data)"
            )
        if set(pauli) == {"I"}:
            return 1.0
        measured = circuit.copy()
        for qubit, char in enumerate(pauli):
            if char == "X":
                measured.h(qubit)
            elif char == "Y":
                measured.sdg(qubit)
                measured.h(qubit)
        measured.measure_all()
        transpiled = transpile(
            measured,
            simulator,
            optimization_level=0,
            seed_transpiler=seed,
        )
        result = simulator.run(
            transpiled, shots=shots, seed_simulator=seed
        ).result()
        counts = result.get_counts()
        return expectation_from_counts(counts, pauli)

    return executor


class RealHardwareBackend:
    """LEGACY placeholder — real hardware now lives in ``qemsel.hardware``.

    This class was the pre-implementation stub and is intentionally kept
    non-constructible: the real-hardware entry point is NOT a class but the
    dispatch seam by backend NAME — pass a name starting with ``ibm_``
    (e.g. ``'ibm_brisbane'``) to :func:`make_executor` /
    :func:`get_backend_info` (or use it in an experiment config), with your
    API key + instance CRN in ``configs/hardware.yaml`` and
    ``hardware_confirmed: true`` in the experiment config
    (``qemsel.experiment._validate_config`` enforces the cost-consent gate).
    See ``qemsel.hardware`` and ``scripts/estimate_hardware_cost.py``.
    """

    def __init__(self, name: str = "ibm_brisbane") -> None:
        raise NotImplementedError(
            "RealHardwareBackend is a legacy stub — do not instantiate it. "
            "Real hardware is implemented in qemsel.hardware and reached by "
            "backend NAME: use e.g. 'ibm_brisbane' with make_executor / "
            "get_backend_info / an experiment config, with credentials in "
            "configs/hardware.yaml and hardware_confirmed: true in the "
            "experiment config."
        )
