"""Real IBM Quantum hardware access (quantum.cloud.ibm.com, Open Plan).

This module implements the real-hardware side of the dispatch seam described
in README "Switching to real IBM hardware": backend names starting with
``ibm_`` (e.g. ``ibm_brisbane``) are routed here by
``qemsel.backends.make_executor`` / ``get_backend_info``, while the fake
simulated backends keep their unchanged code path.

Credential handling — SAFETY RULES (binding):
    * Credentials live ONLY in ``configs/hardware.yaml`` (gitignored).
    * The token must NEVER be printed, logged, or embedded in exception
      messages. Functions here never echo the credentials dict; callers must
      not either. Even YAML parse errors are re-raised WITHOUT the parser's
      message (it quotes file content, which could contain the token).
    * No job is ever submitted without (a) the config-level consent flag
      ``hardware_confirmed: true`` (enforced by
      ``qemsel.experiment._validate_config``) and (b) the in-process QPU
      budget guard below.

Cost model (deliberately conservative — used by the budget guard and
``scripts/estimate_hardware_cost.py``):
    * One executor call = one SamplerV2 job with ONE circuit.
    * Estimated QPU seconds per job =
      ``EST_JOB_OVERHEAD_SECONDS + shots * EST_SECONDS_PER_SHOT``.
    * ``EST_SECONDS_PER_SHOT = 0.001`` (1 ms/shot). Real IBM devices default
      to ~250 us repetition delay plus O(us) gate time and O(100 us) readout,
      so ~0.3-0.5 ms/shot is typical; 1 ms builds in a 2-4x safety margin.
    * ``EST_JOB_OVERHEAD_SECONDS = 2.0`` per job for QPU-side load/overhead
      that the Open Plan may bill beyond pure shot time.
    * The suite-level estimate ignores ``min_abs_ideal`` screening (which
      can only REMOVE units via a free local statevector check), so it is an
      upper bound.
    These are estimation heuristics, not IBM's billing formula; treat the
    output as a planning number with margin, not an invoice.

Execution path per executor call (contract identical to
``qemsel.backends.make_executor``: ``executor(circuit_without_measurements,
pauli) -> float`` with the qemsel pauli convention ``pauli[i]`` acts on
qubit i):
    1. copy the circuit, rotate X/Y support into the Z basis (X: h,
       Y: sdg; h), ``measure_all()`` — the caller's circuit is never mutated;
    2. ISA-transpile with ``generate_preset_pass_manager(optimization_level=0,
       backend=..., seed_transpiler=seed)`` — level 0 is MANDATORY so mitiq's
       ZNE-folded G Gdag G sequences survive (routing/basis translation still
       happen, which SamplerV2 requires: it validates ISA circuits against
       ``backend.target``);
    3. charge the estimated QPU seconds against the in-process cap
       (hard-stop BEFORE submission when it would be exceeded);
    4. submit via ``SamplerV2`` inside a shared ``Batch`` (opened lazily on
       the first call, reused by every later call of the same executor);
    5. counts -> expectation via ``qemsel.backends.expectation_from_counts``
       (little-endian parity already solved there; ``measure_all`` happens
       BEFORE transpile, so counts bits index LOGICAL qubits even after
       routing pads the circuit to device width).

Testing: ``QiskitRuntimeService``, ``SamplerV2``, ``Batch`` and
``generate_preset_pass_manager`` are imported at module level ON PURPOSE so
tests can monkeypatch them at this import site (``qemsel.hardware.SamplerV2``
etc.) without touching qiskit internals.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable

import yaml
from qiskit import QuantumCircuit
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_ibm_runtime import Batch, QiskitRuntimeService, SamplerV2

from qemsel.backends import expectation_from_counts

#: Project root (this file is src/qemsel/hardware.py).
_PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

#: Default credentials file (gitignored — verified in .gitignore).
DEFAULT_CREDENTIALS_PATH: Path = _PROJECT_ROOT / "configs" / "hardware.yaml"

#: The new IBM Quantum Platform channel name (quantum.cloud.ibm.com).
#: Valid channels in qiskit-ibm-runtime 0.48.0: 'ibm_quantum_platform'
#: (recommended), 'ibm_cloud' (legacy alias, same platform), 'local'.
DEFAULT_CHANNEL: str = "ibm_quantum_platform"

#: Free Open Plan monthly QPU budget: 10 minutes.
FREE_PLAN_MONTHLY_SECONDS: float = 600.0

#: Default in-process hard cap on cumulative ESTIMATED QPU seconds. One
#: process may never submit past this without an explicit larger cap
#: (make_real_executor(max_qpu_seconds=...) or ``qpu_seconds_cap`` in
#: hardware.yaml). 120 s = 2 min, i.e. 20% of the free monthly budget.
DEFAULT_QPU_SECONDS_CAP: float = 120.0

#: Conservative per-shot QPU time estimate (see module docstring).
EST_SECONDS_PER_SHOT: float = 0.001

#: Conservative per-job QPU-side overhead estimate (see module docstring).
EST_JOB_OVERHEAD_SECONDS: float = 2.0

#: Token values that mean "no token configured" (compared lowercase).
_BLANK_TOKEN_VALUES = {"", "null", "none", "your_api_key", "paste_your_key_here"}

#: Target operation names that are neither gates nor measurements (same set
#: qemsel.backends uses for the fake backends; duplicated on purpose — the
#: fake-backend module stays byte-for-byte independent of this one).
_NON_GATE_OPS = {"measure", "reset", "delay", "barrier", "id"}

#: Cache for get_real_backend_info results, keyed by backend name.
_REAL_INFO_CACHE: dict[str, dict] = {}

#: Cumulative ESTIMATED QPU seconds charged by executors in this process.
_QPU_SECONDS_USED: float = 0.0


class HardwareUnavailableError(RuntimeError):
    """Real-hardware access is not possible (missing/blank credentials)."""


class HardwareBudgetExceededError(RuntimeError):
    """Submitting the next job would exceed the in-process QPU-seconds cap."""


# ---------------------------------------------------------------------------
# In-process usage accounting
# ---------------------------------------------------------------------------


def qpu_seconds_used() -> float:
    """Cumulative estimated QPU seconds charged by this process so far."""
    return _QPU_SECONDS_USED


def reset_qpu_usage() -> None:
    """Reset the in-process usage counter to zero (tests / new budget)."""
    global _QPU_SECONDS_USED
    _QPU_SECONDS_USED = 0.0


def clear_hardware_caches() -> None:
    """Clear the real-backend info cache (tests)."""
    _REAL_INFO_CACHE.clear()


def _charge_qpu_seconds(estimate: float, cap: float) -> None:
    """Charge ``estimate`` seconds against the in-process cap, or hard-stop.

    Raises:
        HardwareBudgetExceededError: BEFORE any submission, when
            ``used + estimate`` would exceed ``cap``. Nothing is charged in
            that case; charging happens only when the job will be submitted.
    """
    global _QPU_SECONDS_USED
    if _QPU_SECONDS_USED + estimate > cap:
        raise HardwareBudgetExceededError(
            f"submitting this job (~{estimate:.1f} estimated QPU-seconds) "
            f"would push in-process usage from {_QPU_SECONDS_USED:.1f}s past "
            f"the cap of {cap:.1f}s — hard stop, nothing was submitted. "
            "Raise the cap only deliberately: make_real_executor("
            "max_qpu_seconds=...) or 'qpu_seconds_cap' in configs/hardware.yaml "
            f"(free Open Plan budget: {FREE_PLAN_MONTHLY_SECONDS:.0f}s/month)."
        )
    _QPU_SECONDS_USED += estimate


# ---------------------------------------------------------------------------
# Credentials + service
# ---------------------------------------------------------------------------


def load_credentials(path: Path | None = None) -> dict | None:
    """Load IBM Quantum credentials from ``configs/hardware.yaml``.

    Returns None (no error) when the file is missing or ``ibm_token`` is
    absent/blank/placeholder — code must keep working without credentials.

    NEVER print, log, or repr() the returned dict: it contains the API token.

    Args:
        path: credentials YAML (default ``configs/hardware.yaml`` at the
            project root). Recognized keys: ``ibm_token`` (required),
            ``instance`` (CRN, recommended), ``default_backend`` (optional),
            ``channel`` (optional, default 'ibm_quantum_platform'),
            ``qpu_seconds_cap`` (optional float, in-process budget cap).

    Returns:
        dict with keys ``token, instance, default_backend, channel,
        qpu_seconds_cap`` (missing optionals are None) — or None when no
        usable token is configured.

    Raises:
        ValueError: when the file exists but is not valid YAML / not a
            mapping. The parser's message is deliberately NOT included (it
            quotes file content, which could contain the token).
    """
    path = Path(path) if path is not None else DEFAULT_CREDENTIALS_PATH
    if not path.exists():
        return None
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(
            f"could not parse {path.name} as YAML "
            f"({type(exc).__name__}; message withheld — it may quote the "
            "token). Fix the file by hand; do not paste it anywhere."
        ) from None
    if data is None:
        return None
    if not isinstance(data, dict):
        raise ValueError(
            f"{path.name} must be a YAML mapping (key: value lines), got "
            f"{type(data).__name__}"
        )
    token = data.get("ibm_token")
    if not isinstance(token, str) or token.strip().lower() in _BLANK_TOKEN_VALUES:
        return None

    def _opt_str(key: str) -> str | None:
        value = data.get(key)
        if not isinstance(value, str) or not value.strip():
            return None
        return value.strip()

    cap = data.get("qpu_seconds_cap")
    if isinstance(cap, bool) or not isinstance(cap, (int, float)) or cap <= 0:
        cap = None
    return {
        "token": token.strip(),
        "instance": _opt_str("instance"),
        "default_backend": _opt_str("default_backend"),
        "channel": _opt_str("channel") or DEFAULT_CHANNEL,
        "qpu_seconds_cap": float(cap) if cap is not None else None,
    }


def get_service(
    path: Path | None = None, credentials: dict | None = None
) -> QiskitRuntimeService:
    """Build a ``QiskitRuntimeService`` for the new IBM Quantum Platform.

    Args:
        path: credentials file for :func:`load_credentials` (ignored when
            ``credentials`` is given).
        credentials: pre-loaded credentials dict (as returned by
            :func:`load_credentials`).

    Returns:
        Connected ``QiskitRuntimeService`` (channel
        'ibm_quantum_platform', token + instance CRN from the credentials).

    Raises:
        HardwareUnavailableError: when no usable credentials exist.
        Exception: whatever ``QiskitRuntimeService`` raises on bad
            token/instance — propagated unchanged (its messages do not echo
            the token).
    """
    creds = credentials if credentials is not None else load_credentials(path)
    if creds is None:
        raise HardwareUnavailableError(
            "no IBM Quantum credentials: configs/hardware.yaml is missing or "
            "its ibm_token is blank. Create a free account at "
            "https://quantum.cloud.ibm.com, then paste your API key "
            "(ibm_token) and instance CRN (instance) into "
            "configs/hardware.yaml (gitignored — never commit it)."
        )
    kwargs: dict[str, Any] = {
        "channel": creds.get("channel") or DEFAULT_CHANNEL,
        "token": creds["token"],
    }
    if creds.get("instance"):
        kwargs["instance"] = creds["instance"]
    return QiskitRuntimeService(**kwargs)


def list_real_backends(
    service: QiskitRuntimeService | None = None, path: Path | None = None
) -> list[dict]:
    """List the real backends visible to the account, with queue depth.

    Submits NOTHING. Purely informational.

    Args:
        service: an existing service (built once by the caller); when None,
            one is created via :func:`get_service`.
        path: credentials file, forwarded to :func:`get_service`.

    Returns:
        list of dicts, one per backend:
        ``{'name', 'n_qubits', 'operational', 'status_msg', 'pending_jobs'}``
        (status fields are None/'unknown' when the status call fails).
    """
    service = service if service is not None else get_service(path=path)
    out: list[dict] = []
    for backend in service.backends():
        entry: dict[str, Any] = {
            "name": str(getattr(backend, "name", "?")),
            "n_qubits": int(getattr(backend, "num_qubits", 0) or 0),
        }
        try:
            status = backend.status()
            entry["operational"] = bool(status.operational)
            entry["status_msg"] = str(status.status_msg)
            entry["pending_jobs"] = int(status.pending_jobs)
        except Exception:  # noqa: BLE001 - status is best-effort metadata
            entry["operational"] = None
            entry["status_msg"] = "unknown"
            entry["pending_jobs"] = None
        out.append(entry)
    return out


# ---------------------------------------------------------------------------
# Backend info (same key contract as qemsel.backends.get_backend_info)
# ---------------------------------------------------------------------------


def _summarize_target(name: str, target: Any) -> dict:
    """Summarize a qiskit ``Target`` into the get_backend_info key contract.

    Same aggregation the fake-backend path uses (duplicated on purpose so the
    fake path stays untouched): mean 1q/2q gate error, mean/max readout
    error; entries with None/NaN errors are skipped; empty categories -> NaN.
    """
    one_q: list[float] = []
    two_q: list[float] = []
    readout: list[float] = []
    for op_name in target.operation_names:
        props_map = target.get(op_name)
        if not props_map:
            continue
        for qargs, props in props_map.items():
            if props is None or getattr(props, "error", None) is None:
                continue
            err = float(props.error)
            if math.isnan(err):
                continue
            if op_name == "measure":
                readout.append(err)
            elif op_name in _NON_GATE_OPS or qargs is None:
                continue
            elif len(qargs) == 1:
                one_q.append(err)
            elif len(qargs) == 2:
                two_q.append(err)

    def _mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else float("nan")

    return {
        "name": name,
        "n_qubits": int(target.num_qubits),
        "avg_1q_error": _mean(one_q),
        "avg_2q_error": _mean(two_q),
        "avg_readout_error": _mean(readout),
        "max_readout_error": max(readout) if readout else float("nan"),
    }


def get_real_backend_info(
    name: str,
    service: QiskitRuntimeService | None = None,
    path: Path | None = None,
) -> dict:
    """get_backend_info for a REAL backend (name starting with ``ibm_``).

    Same key contract as ``qemsel.backends.get_backend_info``: ``name,
    n_qubits, avg_1q_error, avg_2q_error, avg_readout_error,
    max_readout_error`` — derived from the live device's ``backend.target``
    (today's calibration data). Cached per name for the process lifetime;
    a copy is returned so callers cannot corrupt the cache.

    Raises:
        HardwareUnavailableError: when no credentials are configured.
    """
    if name in _REAL_INFO_CACHE:
        return dict(_REAL_INFO_CACHE[name])
    service = service if service is not None else get_service(path=path)
    backend = service.backend(name)
    info = _summarize_target(name, backend.target)
    _REAL_INFO_CACHE[name] = info
    return dict(info)


# ---------------------------------------------------------------------------
# Cost estimation
# ---------------------------------------------------------------------------


def estimate_job_qpu_seconds(n_circuits: int, shots: int) -> float:
    """Conservative estimated QPU seconds for ONE SamplerV2 job.

    ``EST_JOB_OVERHEAD_SECONDS + n_circuits * shots * EST_SECONDS_PER_SHOT``
    — see the module docstring for the assumptions behind the constants.
    """
    return EST_JOB_OVERHEAD_SECONDS + n_circuits * shots * EST_SECONDS_PER_SHOT


def estimate_config_qpu_seconds(config: dict) -> dict:
    """Estimate the real-hardware QPU cost of an experiment config.

    Pure local arithmetic + statevector-free suite counting — no credentials,
    no network, no jobs. Counts only backends whose name starts with
    ``ibm_`` (fake simulated backends are free).

    The per-unit executor-call counts are ``mitigation.SHOT_MULTIPLIER``
    (raw 1, zne 3, cdr 11, rem 3 — the multiplier IS the number of executor
    invocations, each of which becomes one single-circuit hardware job).
    ``min_abs_ideal`` screening is ignored (it can only remove units), so
    the estimate is an upper bound.

    Args:
        config: experiment config dict (run_experiment schema).

    Returns:
        dict with keys: ``n_circuits, n_ibm_backends, ibm_backends, n_units,
        shots, techniques, per_technique_jobs_per_unit, jobs_per_unit,
        total_jobs, est_seconds_per_job, est_total_qpu_seconds,
        free_plan_monthly_seconds, fits_free_plan, assumptions``.

    Raises:
        ValueError: on an unknown technique name or a malformed circuits
            config (via ``circuits.generate_suite``).
    """
    from qemsel import circuits as _circuits
    from qemsel import mitigation as _mitigation

    circuits_cfg = config.get("circuits") or {}
    suite = _circuits.generate_suite(circuits_cfg)
    n_circuits = len(suite)

    ibm_backends = [
        str(b) for b in (config.get("backends") or []) if str(b).startswith("ibm_")
    ]
    n_units = n_circuits * len(ibm_backends)

    techniques = list(config.get("techniques", _mitigation.TECHNIQUES))
    unknown = [t for t in techniques if t not in _mitigation.SHOT_MULTIPLIER]
    if unknown:
        raise ValueError(
            f"unknown technique(s) {unknown!r}; known: "
            f"{sorted(_mitigation.SHOT_MULTIPLIER)!r}"
        )
    per_technique = {t: int(_mitigation.SHOT_MULTIPLIER[t]) for t in techniques}
    jobs_per_unit = sum(per_technique.values())
    total_jobs = n_units * jobs_per_unit

    shots = int(config.get("shots", 1024))
    per_job = estimate_job_qpu_seconds(1, shots)
    total = total_jobs * per_job

    return {
        "n_circuits": n_circuits,
        "n_ibm_backends": len(ibm_backends),
        "ibm_backends": ibm_backends,
        "n_units": n_units,
        "shots": shots,
        "techniques": techniques,
        "per_technique_jobs_per_unit": per_technique,
        "jobs_per_unit": jobs_per_unit,
        "total_jobs": total_jobs,
        "est_seconds_per_job": per_job,
        "est_total_qpu_seconds": total,
        "free_plan_monthly_seconds": FREE_PLAN_MONTHLY_SECONDS,
        "fits_free_plan": total <= FREE_PLAN_MONTHLY_SECONDS,
        "assumptions": (
            f"1 executor call = 1 single-circuit job; "
            f"{EST_JOB_OVERHEAD_SECONDS:.1f}s overhead/job + "
            f"{EST_SECONDS_PER_SHOT * 1000:.1f}ms/shot (conservative: real "
            "devices run ~0.3-0.5 ms/shot); min_abs_ideal screening ignored "
            "(upper bound); planning heuristic, not IBM's billing formula"
        ),
    }


# ---------------------------------------------------------------------------
# Real-hardware executor
# ---------------------------------------------------------------------------


def _validate_pauli(pauli: str, n_qubits: int) -> None:
    """Raise ValueError unless ``pauli`` is a valid string for ``n_qubits``."""
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


def make_real_executor(
    backend_name: str,
    shots: int,
    seed: int,
    *,
    service: QiskitRuntimeService | None = None,
    path: Path | None = None,
    max_qpu_seconds: float | None = None,
) -> Callable[[QuantumCircuit, str], float]:
    """Build an expectation-value executor that runs on REAL IBM hardware.

    Same contract as ``qemsel.backends.make_executor``:
    ``executor(circuit_without_measurements, pauli) -> float`` in the qemsel
    pauli convention. See the module docstring for the exact execution path
    (basis rotation, measure_all on a COPY, ISA transpile at
    optimization_level=0, budget check, SamplerV2 in a shared Batch,
    counts -> expectation via ``expectation_from_counts``).

    Differences vs the simulated executor (unavoidable physics/API facts):
        * ``seed`` seeds ONLY the transpiler — real quantum shot noise cannot
          be seeded; identical calls will differ at the 1/sqrt(shots) level.
        * Every call consumes REAL QPU time from the free 10 min/month
          budget. The in-process budget guard hard-stops (raises
          ``HardwareBudgetExceededError`` BEFORE submitting) once cumulative
          estimated usage would pass the cap.

    The returned callable carries two extra attributes:
        * ``executor.close()`` — closes the shared Batch (call when done).
        * ``executor.estimated_seconds_per_call`` — the per-job estimate.

    Args:
        backend_name: real device name (e.g. ``ibm_brisbane``).
        shots: shots per job.
        seed: transpiler seed (reproducible routing/layout).
        service: pre-built service (skips credential loading — used by
            tests and by callers managing one service for many executors).
        path: credentials file for :func:`load_credentials`.
        max_qpu_seconds: in-process cap override. Default: the
            ``qpu_seconds_cap`` value from hardware.yaml when present, else
            ``DEFAULT_QPU_SECONDS_CAP`` (120 s).

    Raises:
        HardwareUnavailableError: when no credentials are configured and no
            ``service`` was supplied.
    """
    cap = max_qpu_seconds
    if service is None:
        creds = load_credentials(path)
        if creds is None:
            raise HardwareUnavailableError(
                "cannot build a real-hardware executor: no usable credentials "
                "in configs/hardware.yaml (see load_credentials)."
            )
        if cap is None and creds.get("qpu_seconds_cap"):
            cap = float(creds["qpu_seconds_cap"])
        service = get_service(credentials=creds)
    if cap is None:
        cap = DEFAULT_QPU_SECONDS_CAP

    backend = service.backend(backend_name)
    backend_qubits = int(getattr(backend, "num_qubits", 0) or 0)
    # optimization_level=0 is MANDATORY: ZNE-folded G Gdag G sequences must
    # survive. Routing + basis translation still run, producing ISA circuits
    # (SamplerV2 validates them against backend.target).
    pass_manager = generate_preset_pass_manager(
        optimization_level=0, backend=backend, seed_transpiler=seed
    )
    per_call_estimate = estimate_job_qpu_seconds(1, shots)
    state: dict[str, Any] = {"batch": None}

    def executor(circuit: QuantumCircuit, pauli: str) -> float:
        """Hardware expectation of <pauli> on ``circuit`` (qemsel convention)."""
        _validate_pauli(pauli, circuit.num_qubits)
        if backend_qubits and circuit.num_qubits > backend_qubits:
            raise ValueError(
                f"circuit has {circuit.num_qubits} qubits but backend "
                f"{backend_name!r} has only {backend_qubits}"
            )
        if set(pauli) == {"I"}:
            return 1.0  # identity observable: no job needed
        measured = circuit.copy()
        for qubit, char in enumerate(pauli):
            if char == "X":
                measured.h(qubit)
            elif char == "Y":
                measured.sdg(qubit)
                measured.h(qubit)
        measured.measure_all()
        isa_circuit = pass_manager.run(measured)
        # Hard budget stop BEFORE submission (nothing is charged on refusal).
        _charge_qpu_seconds(per_call_estimate, cap)
        if state["batch"] is None:
            state["batch"] = Batch(backend=backend)
        sampler = SamplerV2(mode=state["batch"])
        job = sampler.run([isa_circuit], shots=shots)
        pub_result = job.result()[0]
        counts = pub_result.join_data().get_counts()
        return expectation_from_counts(counts, pauli)

    def _close() -> None:
        batch = state["batch"]
        if batch is not None:
            try:
                batch.close()
            finally:
                state["batch"] = None

    executor.close = _close  # type: ignore[attr-defined]
    executor.estimated_seconds_per_call = per_call_estimate  # type: ignore[attr-defined]
    return executor
