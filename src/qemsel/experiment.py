"""Benchmark experiment runner: circuits x backends x techniques -> labeled dataset.

Crash-safe sweep driver. One work unit = one (circuit, backend) pair; after
every completed unit one row is appended to ``results.csv``, so a crash loses
at most one unit and re-running the same command resumes where it stopped.

Winner columns (both recorded per row)
--------------------------------------
* ``best_technique`` (interface-mandated): the technique with the smallest
  non-NaN ``<tech>_abs_error`` — the pure-accuracy winner. It ignores that
  e.g. CDR consumed 11x more shots than raw.
* ``best_technique_cost_aware`` (EXTRA column, appended after
  ``best_technique``): argmin over techniques of

      abs_error * sqrt(shots_consumed / base_shots)

  Rationale: the statistical (shot-noise) uncertainty of an expectation value
  scales like 1/sqrt(shots). A technique that spends ``k * base_shots`` total
  shots gets a "free" ~sqrt(k) noise reduction purely from extra averaging,
  so we scale its error back up by sqrt(k) to compare all techniques at an
  equal quantum-resource budget. A technique wins the cost-aware column only
  if its error reduction beats the trivial gain of simply taking more shots.
  Like ``best_technique`` it is ``''`` (empty string) when every technique
  failed. NaN entries are excluded from both argmins.

All cross-module calls go through the public interfaces of ``qemsel.circuits``,
``qemsel.backends``, ``qemsel.ideal``, ``qemsel.features`` and
``qemsel.mitigation`` — accessed as module attributes (``_mitigation.apply_technique``)
so tests can monkeypatch them.
"""

from __future__ import annotations

import json
import logging
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

import qemsel
from qemsel import backends as _backends
from qemsel import circuits as _circuits
from qemsel import features as _features
from qemsel import hardware as _hardware
from qemsel import ideal as _ideal
from qemsel import mitigation as _mitigation

_log = logging.getLogger(__name__)

def print(*args, sep=" ", end="\n", file=None, flush=False):
    msg = sep.join(str(arg) for arg in args)
    if "warning" in msg.lower() or "error" in msg.lower() or "abort" in msg.lower():
        _log.warning(msg)
    else:
        _log.info(msg)

_NAN: float = float("nan")

#: Fixed leading columns of every results row (before feat_*/technique columns).
_BASE_COLUMNS: list[str] = [
    "circuit_id",
    "family",
    "n_qubits",
    "depth",
    "seed",
    "backend",
    "pauli",
    "ideal",
]

#: Name of the extra cost-normalized winner column (see module docstring).
COST_AWARE_COLUMN: str = "best_technique_cost_aware"

#: Fixed leading columns of every aggregated.csv row (see _write_aggregated).
_AGG_KEY_COLUMNS: list[str] = ["family", "n_qubits", "depth", "backend"]

#: V2 (INTERFACES.md section V2; builder-experiment / B4 implements): name
#: of the per-unit shot-budget column. Present ONLY when the config's
#: ``shots`` key is a LIST (see ``_normalize_shots``); scalar-shots runs
#: keep the V1 schema byte-identical.
BASE_SHOTS_COLUMN: str = "base_shots"

#: V2 aggregated.csv key columns when the shots axis is active: seeds are
#: averaged WITHIN a shot budget, never across budgets (the boundary needs
#: per-budget labels).
_AGG_KEY_COLUMNS_V2: list[str] = [
    "family",
    "n_qubits",
    "depth",
    "backend",
    "base_shots",
]


def _normalize_shots(shots_cfg: object) -> tuple[list[int], bool]:
    """V2 STUB (builder-experiment / B4 implements) — the shots axis.

    Normalizes the config's ``shots`` key and returns
    ``(budgets, list_mode)``.

    Contract (the FULL V2 experiment behavior, also summarized in
    INTERFACES.md section V2):

    * SCALAR int (V1): returns ``([shots], False)``. In non-list mode every
      observable behavior of run_experiment — results.csv columns, resume
      keys, aggregated.csv, logs, run_meta — stays BYTE-IDENTICAL to V1
      (regression duty: rerun configs/tiny.yaml against the stored
      reference).
    * LIST of ints (V2): returns ``(list(shots_cfg), True)`` after
      validation (non-empty; every entry a positive non-bool int; entries
      distinct; order preserved — it becomes the innermost loop order).
      In list mode:
      - one work unit = (circuit, backend, base_shots); loop order:
        circuits (outer, generate_suite order) x backends (config order) x
        budgets (config order, innermost);
      - results.csv gains the ``BASE_SHOTS_COLUMN`` int column, inserted
        between 'pauli' and 'ideal';
      - the resume key becomes (circuit_id, backend, base_shots); an
        existing results.csv WITHOUT the base_shots column fails the
        column-equality check in _load_existing with the normal
        "use a fresh out_dir" ValueError — never silently mixed;
      - the per-unit executor is built at that unit's budget
        (``make_executor(backend, unit_shots, seed)``); raw_plus/zne_fr
        rebuild THEIR executors from the same unit budget (their shots
        parameter already flows through apply_technique);
      - ``<tech>_shots`` = shots_consumed(tech, unit_shots); both winner
        columns use the unit's own budget as base;
      - the low-signal screen is evaluated once per (circuit, backend) —
        the ideal value is shots-independent — and skips ALL budgets of
        that pair (one skipped_low_signal.log line, V1 format);
      - errors.log lines gain the budget field in list mode ONLY:
        ``'{circuit_id},{backend},s{base_shots},{technique}: {exc!r}'``;
      - aggregated.csv groups by ``_AGG_KEY_COLUMNS_V2`` (base_shots after
        backend), same coverage rule, cost-aware scores from the group's
        own budget.

    Related V2 config keys (validated in _validate_config, B4):

    * ``feature_version`` (optional int, default 1; must be a key of
      ``qemsel.features.FEATURE_NAMES_BY_VERSION``): feat_* columns become
      the selected version's names and extract_features is called as
      ``extract_features(circuit, backend, version=v, base_shots=unit
      budget)``. NOTE for Angle 3 configs: a shots LIST with
      feature_version 1 trains a shots-blind selector — legal, but the
      boundary configs must set ``feature_version: 2``.
    * ``techniques`` may now name any of ``mitigation.TECHNIQUES_V2``
      (validation switches to the V2 list; the DEFAULT stays
      ``mitigation.TECHNIQUES`` — V1 five — so existing configs are
      untouched).

    Raises:
        ValueError: anything else (bool, float, empty list, duplicates,
            non-positive entries).
    """

    def _is_pos_int(x: object) -> bool:
        return isinstance(x, int) and not isinstance(x, bool) and x > 0

    # bool is an int subclass — reject it before the scalar-int branch.
    if isinstance(shots_cfg, bool):
        raise ValueError(
            f"config['shots'] must be a positive int or a list of positive "
            f"ints, got {shots_cfg!r}"
        )
    if isinstance(shots_cfg, int):
        if shots_cfg <= 0:
            raise ValueError(
                f"config['shots'] must be a positive int, got {shots_cfg!r}"
            )
        return [shots_cfg], False
    if isinstance(shots_cfg, (list, tuple)):
        budgets = list(shots_cfg)
        if not budgets:
            raise ValueError("config['shots'] list must be non-empty")
        bad = [s for s in budgets if not _is_pos_int(s)]
        if bad:
            raise ValueError(
                f"config['shots'] list entries must be positive non-bool "
                f"ints, offending: {bad!r}"
            )
        if len(set(budgets)) != len(budgets):
            raise ValueError(
                f"config['shots'] list entries must be distinct, got {budgets!r}"
            )
        return budgets, True
    raise ValueError(
        f"config['shots'] must be a positive int or a list of positive ints, "
        f"got {shots_cfg!r} ({type(shots_cfg).__name__})"
    )


def _result_columns(
    techniques: list[str],
    *,
    feature_names: list[str] | None = None,
    include_base_shots: bool = False,
) -> list[str]:
    """Full ordered column list of results.csv for the given technique set.

    Order: base columns, feat_<name> in feature-name order, then per
    technique (config order) value/abs_error/shots triples, then the two
    winner columns.

    V2 keyword-only params (defaults reproduce V1 byte-identically):

    * ``feature_names`` — the feat_<name> list (``features.FEATURE_NAMES``
      when None, i.e. feature_version 1).
    * ``include_base_shots`` — when True (shots-list mode) the
      ``BASE_SHOTS_COLUMN`` int column is inserted between 'pauli' and
      'ideal'.
    """
    if feature_names is None:
        feature_names = _features.FEATURE_NAMES
    cols = list(_BASE_COLUMNS)
    if include_base_shots:
        cols.insert(cols.index("ideal"), BASE_SHOTS_COLUMN)
    cols += [f"feat_{name}" for name in feature_names]
    for tech in techniques:
        cols += [f"{tech}_value", f"{tech}_abs_error", f"{tech}_shots"]
    cols += ["best_technique", COST_AWARE_COLUMN]
    return cols


def _validate_pauli_spec(spec: object, where: str) -> None:
    """Raise ValueError unless ``spec`` is 'auto' or an I/X/Y/Z string."""
    if not isinstance(spec, str) or spec == "":
        raise ValueError(
            f"{where} must be 'auto' or a Pauli string, got {spec!r}"
        )
    if spec != "auto" and any(c not in "IXYZ" for c in spec):
        raise ValueError(f"{where} contains invalid characters: {spec!r}")


def _resolve_pauli(pauli_cfg: str | dict, family: str, n_qubits: int) -> str:
    """Resolve the config pauli spec for one circuit.

    ``pauli_cfg`` is either a string ('auto' => 'Z'*n, else explicit) or a
    per-family dict {family: spec, 'default': spec}. A single-character spec
    ('X'/'Y'/'Z') is repeated to the circuit width — this is how a family
    gets an O(1)-signal observable at every size (e.g. GHZ <X...X> = +1 for
    every n, while <Z...Z> is 0 for odd n).
    """
    if isinstance(pauli_cfg, dict):
        spec = pauli_cfg.get(family, pauli_cfg.get("default", "auto"))
    else:
        spec = pauli_cfg
    if spec == "auto":
        return "Z" * n_qubits
    if len(spec) == 1:
        return spec * n_qubits
    return spec


def _validate_config(
    config: dict,
) -> tuple[dict, list[str], int, str | dict, list[str], float]:
    """Validate the run config; return
    (circuits_cfg, backends, shots, pauli, techniques, min_abs_ideal).

    Raises:
        ValueError: on missing keys, unknown backend/technique names,
            non-positive shots, empty backend/technique lists, circuits
            wider than a configured backend, or a malformed pauli spec /
            min_abs_ideal.
    """
    if not isinstance(config, dict):
        raise ValueError(f"config must be a dict, got {type(config).__name__}")
    for key in ("circuits", "backends", "shots"):
        if key not in config:
            raise ValueError(f"config missing required key {key!r}")

    circuits_cfg = config["circuits"]
    if not isinstance(circuits_cfg, dict):
        raise ValueError("config['circuits'] must be a dict (generate_suite schema)")

    backend_names = list(config["backends"])
    if not backend_names:
        raise ValueError("config['backends'] must be a non-empty list")
    # Names starting 'ibm_' are REAL hardware (qemsel.hardware dispatch);
    # they are gated separately at the end of this function. Fake-backend
    # names may carry an '@x<scale>' noise-scale suffix (distinct noise
    # environments on one device): validation delegates to
    # backends.parse_backend_name — the SAME grammar make_executor enforces
    # — so a config that validates can never die mid-run on a bad name
    # (malformed suffixes raise ValueError right here).
    ibm_names = [b for b in backend_names if str(b).startswith("ibm_")]
    unknown_b = [
        b
        for b in backend_names
        if not str(b).startswith("ibm_")
        and _backends.parse_backend_name(str(b))[0] not in _backends.BACKENDS
    ]
    if unknown_b:
        raise ValueError(
            f"unknown backend(s) {unknown_b!r}; known: {_backends.BACKENDS!r} "
            "(optionally with an '@x<scale>' noise-scale suffix, e.g. "
            "'FakeManilaV2@x1.5')"
        )

    # Fail fast when the config asks for circuits wider than a backend:
    # qubits beyond the device would simulate with NO noise (silent wrong
    # data; the executor also guards this at call time).
    n_qubits_cfg = circuits_cfg.get("n_qubits")
    if isinstance(n_qubits_cfg, (list, tuple)) and n_qubits_cfg:
        try:
            max_n = max(int(n) for n in n_qubits_cfg)
        except (TypeError, ValueError):
            max_n = None
        if max_n is not None:
            for name in backend_names:
                if str(name).startswith("ibm_"):
                    # Real backends: width is checked at executor call time
                    # (get_backend_info here would hit the network).
                    continue
                # Width is a property of the BASE device (noise scaling
                # never changes the qubit count).
                base_name = _backends.parse_backend_name(str(name))[0]
                backend_n = int(
                    _backends.get_backend_info(base_name)["n_qubits"]
                )
                if max_n > backend_n:
                    raise ValueError(
                        f"config asks for n_qubits up to {max_n} but backend "
                        f"{name!r} has only {backend_n} qubits; qubits beyond "
                        "the device would simulate with no noise"
                    )

    # shots may be a positive int (V1 scalar) OR a list of distinct positive
    # ints (V2 shots axis). _normalize_shots is the single validator; the raw
    # value is returned unchanged (6-tuple arity is FROZEN — test_hardware
    # unpacks it and asserts scalar shots round-trip identically).
    shots = config["shots"]
    _normalize_shots(shots)

    # feature_version (V2; default 1) selects the feat_* column set and how
    # extract_features is called. Validated here so a bad value fails fast,
    # but NOT part of the return tuple (arity frozen); run_experiment re-reads
    # it from the config.
    feature_version = config.get("feature_version", 1)
    if (
        isinstance(feature_version, bool)
        or feature_version not in _features.FEATURE_NAMES_BY_VERSION
    ):
        raise ValueError(
            f"config['feature_version'] must be one of "
            f"{sorted(_features.FEATURE_NAMES_BY_VERSION)}, got "
            f"{feature_version!r}"
        )

    pauli = config.get("pauli", "auto")
    if isinstance(pauli, dict):
        known_keys = set(_circuits.FAMILIES) | {"default"}
        unknown_f = sorted(set(pauli) - known_keys)
        if unknown_f:
            raise ValueError(
                f"config['pauli'] dict has unknown family key(s) {unknown_f!r}; "
                f"known: {sorted(known_keys)!r}"
            )
        for fam, spec in pauli.items():
            _validate_pauli_spec(spec, f"config['pauli'][{fam!r}]")
    else:
        _validate_pauli_spec(pauli, "config['pauli']")

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

    # Default stays the frozen V1 five (TECHNIQUES); configs opt into the V2
    # techniques (zne_fr/cdr_ridge/cdr_rf) by naming them, so validation
    # accepts any TECHNIQUES_V2 name.
    techniques = list(config.get("techniques", _mitigation.TECHNIQUES))
    if not techniques:
        raise ValueError("config['techniques'] must be a non-empty list")
    unknown_t = [t for t in techniques if t not in _mitigation.TECHNIQUES_V2]
    if unknown_t:
        raise ValueError(
            f"unknown technique(s) {unknown_t!r}; known: "
            f"{_mitigation.TECHNIQUES_V2!r}"
        )

    # ---- REAL-HARDWARE GATE (safety-critical) -------------------------
    # 'ibm_*' backends consume real QPU time from the free Open Plan's
    # 10 min/month. They are allowed ONLY when (a) usable credentials load
    # from configs/hardware.yaml AND (b) the config carries the explicit
    # cost-consent flag `hardware_confirmed: true`. The error always states
    # the estimated cost and how to confirm.
    if ibm_names:
        original_mult = _mitigation.SHOT_MULTIPLIER
        _mitigation.SHOT_MULTIPLIER = _mitigation.SHOT_MULTIPLIER_V2
        est_config = dict(config)
        if isinstance(est_config.get("shots"), list):
            est_config["shots"] = max(est_config["shots"])
        try:
            estimate = _hardware.estimate_config_qpu_seconds(est_config)
        finally:
            _mitigation.SHOT_MULTIPLIER = original_mult
        est_s = estimate["est_total_qpu_seconds"]
        cost_txt = (
            f"estimated cost of this config: {estimate['total_jobs']} jobs, "
            f"~{est_s:.0f} QPU-seconds (~{est_s / 60.0:.1f} min) of the free "
            f"plan's {estimate['free_plan_monthly_seconds'] / 60.0:.0f} "
            "min/month (breakdown: scripts/estimate_hardware_cost.py "
            "--config <your config>)"
        )
        if _hardware.load_credentials() is None:
            raise ValueError(
                f"config requests real IBM backend(s) {ibm_names!r} but no "
                "usable credentials were found: configs/hardware.yaml is "
                "missing or its ibm_token is blank. Paste your API key "
                "(ibm_token) and instance CRN (instance) from "
                f"https://quantum.cloud.ibm.com there first. {cost_txt}."
            )
        if config.get("hardware_confirmed") is not True:
            raise ValueError(
                f"config requests real IBM backend(s) {ibm_names!r}; "
                f"{cost_txt}. Real QPU time will be spent -- review the "
                "estimate, then set `hardware_confirmed: true` in the config "
                "to confirm."
            )

    return circuits_cfg, backend_names, shots, pauli, techniques, min_abs_ideal


def _package_versions() -> dict[str, str]:
    """Installed versions of the packages that determine reproducibility."""
    from importlib.metadata import PackageNotFoundError, version

    packages = ["qiskit", "qiskit-aer", "mitiq", "numpy", "scikit-learn", "pandas"]
    out: dict[str, str] = {}
    for pkg in packages:
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:  # pragma: no cover - all pinned in env
            out[pkg] = "not-installed"
    return out


def _write_run_meta(config: dict, out_dir: Path, n_existing: int) -> None:
    """Write (overwrite) the run_meta.json reproducibility sidecar."""
    meta: dict[str, Any] = {
        "config": config,
        "versions": _package_versions(),
        "python_version": sys.version,
        "qemsel_version": qemsel.__version__,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "resumed_existing_rows": n_existing,
        "out_dir": str(out_dir),
    }
    (out_dir / "run_meta.json").write_text(
        json.dumps(meta, indent=2, default=str), encoding="utf-8"
    )


def _repair_torn_tail(csv_path: Path) -> None:
    """Drop a partial final line left by a crash mid-append (self-heal).

    A CSV written by ``_append_row`` always ends with a newline; a file that
    does not was torn by a crash DURING an append. Left in place, the torn
    tail either (a) parses with NaN-filled trailing fields and poisons
    ``done_pairs`` (the unit is skipped forever), or (b) gets the next
    append glued onto it, after which every ``pd.read_csv`` dies with a
    ParserError and the run is unresumable. Truncating the partial line
    loses at most the one unit the crash already lost — exactly the
    documented crash contract.
    """
    raw = csv_path.read_bytes()
    if not raw or raw.endswith(b"\n"):
        return
    keep, _sep, torn = raw.rpartition(b"\n")
    print(
        f"_load_existing: dropping torn partial final line of {csv_path.name} "
        f"(crash mid-append): {torn[:120]!r}",
        flush=True,
    )
    csv_path.write_bytes(keep + b"\n" if keep else b"")


def _load_existing(csv_path: Path, columns: list[str]) -> pd.DataFrame:
    """Load an existing results.csv for resume; empty DataFrame if absent.

    A torn partial final line (crash mid-append) is truncated with a log
    message before parsing — see ``_repair_torn_tail``.

    Raises:
        ValueError: if the existing file's columns do not match ``columns``
            (e.g. the config's technique list changed between runs).
    """
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return pd.DataFrame(columns=columns)
    _repair_torn_tail(csv_path)
    if csv_path.stat().st_size == 0:
        return pd.DataFrame(columns=columns)
    try:
        df = pd.read_csv(csv_path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=columns)
    if list(df.columns) != columns:
        raise ValueError(
            f"existing {csv_path} has columns {list(df.columns)!r} but this config "
            f"produces {columns!r}; use a fresh out_dir or a matching config"
        )
    # Empty-string winners ('' = all techniques failed) round-trip through CSV
    # as NaN — restore them so the schema contract ('' string) holds.
    for col in ("best_technique", COST_AWARE_COLUMN):
        df[col] = df[col].fillna("").astype(str)
    return df


def _append_row(csv_path: Path, row: dict[str, Any], columns: list[str]) -> None:
    """Append one row to results.csv (header written only when file is new).

    Belt-and-braces against a crash mid-append: if the existing file does
    not end with a newline (torn tail), write one first so the new row can
    never be glued onto a partial line (which would make every later
    ``pd.read_csv`` fail with a ParserError).
    """
    exists = csv_path.exists() and csv_path.stat().st_size > 0
    if exists:
        with csv_path.open("rb+") as fh:
            fh.seek(-1, 2)
            if fh.read(1) != b"\n":
                fh.write(b"\n")
    pd.DataFrame([row], columns=columns).to_csv(
        csv_path, mode="a", header=not exists, index=False
    )


def _pick_winners(
    row: dict[str, Any], techniques: list[str], base_shots: int
) -> tuple[str, str]:
    """Compute (best_technique, best_technique_cost_aware) for a finished row.

    best_technique          = argmin_t  abs_error_t            (NaN excluded)
    best_technique_cost_aware = argmin_t  abs_error_t * sqrt(shots_t / base_shots)

    Ties keep the FIRST technique in config order (strict < comparison).
    Both are '' when every technique failed (all abs_error NaN).
    """
    best_tech, best_err = "", math.inf
    best_cost_tech, best_cost = "", math.inf
    for tech in techniques:
        err = row[f"{tech}_abs_error"]
        if not isinstance(err, (int, float)) or math.isnan(err):
            continue
        if err < best_err:
            best_tech, best_err = tech, err
        # sqrt(shots/base) = sqrt(SHOT_MULTIPLIER): penalize shot-hungry
        # techniques by the noise reduction extra averaging alone would buy.
        score = err * math.sqrt(row[f"{tech}_shots"] / base_shots)
        if score < best_cost:
            best_cost_tech, best_cost = tech, score
    return best_tech, best_cost_tech


def _is_budget_exceeded(exc: BaseException | None) -> bool:
    """True if ``exc`` is (or was caused by) HardwareBudgetExceededError.

    ``mitigation.apply_technique`` wraps every internal failure in a
    ``MitigationError`` with the original exception chained as
    ``__cause__``, so the budget signal must be recognized anywhere in the
    cause chain, not just at the top.
    """
    seen: set[int] = set()
    while exc is not None and id(exc) not in seen:
        if isinstance(exc, _hardware.HardwareBudgetExceededError):
            return True
        seen.add(id(exc))
        exc = exc.__cause__ or exc.__context__
    return False


def _close_executor(executor: Any, unit_label: str) -> None:
    """Best-effort ``executor.close()`` when the executor exposes one.

    The real-hardware executor carries a ``close()`` that shuts its shared
    Batch; the simulated executor has none. A close failure must never kill
    (or un-resume) the sweep, so it is logged and swallowed.
    """
    close = getattr(executor, "close", None)
    if not callable(close):
        return
    try:
        close()
    except Exception as exc:  # noqa: BLE001 - cleanup must not kill the run
        print(
            f"run_experiment: executor.close() failed for {unit_label}: "
            f"{exc!r} (ignored)",
            flush=True,
        )


def _errors_log_prefix(
    circuit_id: str,
    backend: str,
    base_shots: int,
    technique: str,
    list_mode: bool,
) -> str:
    """errors.log line prefix (before ``': {exc!r}'``).

    V1 / scalar mode: ``'{circuit_id},{backend},{technique}'`` (byte-identical).
    List mode ONLY: gains the ``s{base_shots}`` budget field so the same
    (circuit, backend, technique) at different budgets stays distinguishable
    in the log.
    """
    if list_mode:
        return f"{circuit_id},{backend},s{base_shots},{technique}"
    return f"{circuit_id},{backend},{technique}"


def _aggregated_columns(
    techniques: list[str],
    *,
    feature_names: list[str] | None = None,
    include_base_shots: bool = False,
) -> list[str]:
    """Ordered column list of aggregated.csv for the given technique set.

    V2 keyword-only params (defaults reproduce V1 byte-identically):

    * ``feature_names`` — feat_<name> list (``features.FEATURE_NAMES`` when
      None).
    * ``include_base_shots`` — when True (shots-list mode) the key columns
      are ``_AGG_KEY_COLUMNS_V2`` (base_shots after backend); seeds are
      averaged WITHIN a shot budget, never across budgets.
    """
    if feature_names is None:
        feature_names = _features.FEATURE_NAMES
    key_columns = _AGG_KEY_COLUMNS_V2 if include_base_shots else _AGG_KEY_COLUMNS
    cols = list(key_columns) + ["n_seeds"]
    cols += [f"feat_{name}" for name in feature_names]
    for tech in techniques:
        cols += [f"{tech}_mean_abs_error", f"{tech}_n_seeds"]
    cols += ["best_technique", COST_AWARE_COLUMN]
    return cols


def _write_aggregated(
    df: pd.DataFrame,
    techniques: list[str],
    base_shots: int | None,
    out_dir: Path,
    *,
    feature_names: list[str] | None = None,
    list_mode: bool = False,
) -> pd.DataFrame:
    """Write (overwrite) aggregated.csv: seed-averaged errors and labels.

    Rows of ``df`` are grouped by (family, n_qubits, depth, backend) — i.e.
    across seeds — and per technique the MEAN of the non-NaN
    ``<tech>_abs_error`` values is recorded as ``<tech>_mean_abs_error``
    (NaN when the technique failed on every seed), together with
    ``<tech>_n_seeds`` = the number of seeds on which the technique
    produced a value. The ``feat_*`` columns are carried over as the MEAN
    over the group's rows (backend features are constant within a group;
    circuit features can differ slightly across seeds), so aggregated.csv
    is a COMPLETE, directly trainable dataset for
    ``qemsel.model.train_and_eval`` — the fixer pass 2026-07-21 wired this
    (previously the seed-averaged labels existed but the model could not
    consume them: no feature columns).

    The winner columns are recomputed FROM THE MEANS, restricted to
    techniques with MAXIMUM seed coverage in the group:

    * ``best_technique``          = argmin of mean abs_error
    * ``best_technique_cost_aware`` = argmin of
      ``mean_abs_error * sqrt(shots_consumed(tech, base_shots) / base_shots)``

    Coverage rule (fixer pass 2026-07-21): a technique is ELIGIBLE to win
    only if its ``<tech>_n_seeds`` equals the maximum over the group's
    techniques (normally == n_seeds, since raw never fails). A mean built
    from 1 of 3 seeds is not comparable to competitors' 3-seed means — the
    seed a technique happened to survive on may simply be the easy one
    (observed live: cdr "winning" ghz_plus aggregates from its single
    non-refused seed). Partial-coverage techniques keep their recorded
    mean and count but cannot be the aggregate winner.

    Rationale (integration next-step 2026-07-21, "seed-averaged labels"):
    per-seed winner flips between close techniques are label noise at
    4000–8000 shots; averaging the error over seeds before choosing the
    winner yields one lower-variance label per configuration. ``n_seeds``
    records how many seeds (rows) each aggregate is built from. Both winner
    columns are ``''`` when every technique's mean is NaN.

    The file is derived data, recomputed from the COMPLETE DataFrame and
    overwritten on every run (including resumed and aborted ones).

    V2 keyword-only params (defaults reproduce V1 byte-identically):

    * ``feature_names`` — feat_<name> list carried through
      (``features.FEATURE_NAMES`` when None).
    * ``list_mode`` — when True (shots-list run) rows are grouped by
      ``_AGG_KEY_COLUMNS_V2`` (base_shots added after backend), so seeds are
      averaged WITHIN a shot budget and each group's cost-aware scores use
      that group's OWN base_shots (the passed ``base_shots`` is ignored and
      may be None). When False the passed scalar ``base_shots`` is used, as
      in V1.

    Returns:
        The aggregated DataFrame (also written to ``out_dir/aggregated.csv``).
    """
    if feature_names is None:
        feature_names = _features.FEATURE_NAMES
    key_columns = _AGG_KEY_COLUMNS_V2 if list_mode else _AGG_KEY_COLUMNS
    columns = _aggregated_columns(
        techniques, feature_names=feature_names, include_base_shots=list_mode
    )
    agg_path = out_dir / "aggregated.csv"
    if df.empty:
        agg_df = pd.DataFrame(columns=columns)
        agg_df.to_csv(agg_path, index=False)
        return agg_df

    feat_cols = [f"feat_{name}" for name in feature_names]
    rows: list[dict[str, Any]] = []
    grouped = df.groupby(key_columns, sort=True)
    for key, group in grouped:
        row: dict[str, Any] = dict(zip(key_columns, key))
        # In list mode base_shots is a group key; each budget scores against
        # its OWN base (the boundary needs per-budget cost-aware labels).
        group_base = int(row[BASE_SHOTS_COLUMN]) if list_mode else base_shots
        row["n_seeds"] = int(group["seed"].nunique())
        for col in feat_cols:
            row[col] = float(group[col].mean())
        mean_err: dict[str, float] = {}
        n_valid: dict[str, int] = {}
        for tech in techniques:
            # pandas mean() skips NaN; all-NaN -> NaN (technique failed on
            # every seed of this configuration).
            errs = group[f"{tech}_abs_error"]
            mean_err[tech] = float(errs.mean())
            n_valid[tech] = int(errs.notna().sum())
            row[f"{tech}_mean_abs_error"] = mean_err[tech]
            row[f"{tech}_n_seeds"] = n_valid[tech]
        max_valid = max(n_valid.values(), default=0)
        best_tech, best_err = "", math.inf
        best_cost_tech, best_cost = "", math.inf
        for tech in techniques:
            # Coverage rule: only techniques with the group's maximum seed
            # coverage may win (see docstring). max_valid == 0 means every
            # technique failed everywhere -> both winners stay ''.
            if (
                max_valid == 0
                or n_valid[tech] < max_valid
                or math.isnan(mean_err[tech])
            ):
                continue
            if mean_err[tech] < best_err:
                best_tech, best_err = tech, mean_err[tech]
            score = mean_err[tech] * math.sqrt(
                _mitigation.shots_consumed(tech, group_base) / group_base
            )
            if score < best_cost:
                best_cost_tech, best_cost = tech, score
        row["best_technique"] = best_tech
        row[COST_AWARE_COLUMN] = best_cost_tech
        rows.append(row)

    agg_df = pd.DataFrame(rows, columns=columns)
    agg_df.to_csv(agg_path, index=False)
    return agg_df


def _run_single_unit(
    spec: Any,
    circuit: Any,
    backend_name: str,
    pauli: str,
    base_shots: int,
    ideal_value: float,
    feature_version: int,
    feature_names: list[str],
    techniques: list[str],
    list_mode: bool,
    errors_path: Path,
) -> dict[str, Any]:
    """Helper to run a single work unit (useful for parallel execution)."""
    row: dict[str, Any] = {
        "circuit_id": spec.circuit_id,
        "family": spec.family,
        "n_qubits": spec.n_qubits,
        "depth": spec.depth,
        "seed": spec.seed,
        "backend": backend_name,
        "pauli": pauli,
    }
    if list_mode:
        row[BASE_SHOTS_COLUMN] = int(base_shots)
    row["ideal"] = ideal_value

    if feature_version == 1:
        feats = _features.extract_features(circuit, backend_name)
    else:
        feats = _features.extract_features(
            circuit,
            backend_name,
            version=feature_version,
            base_shots=base_shots,
        )
    for name in feature_names:
        row[f"feat_{name}"] = float(feats[name])

    executor = _backends.make_executor(
        backend_name, base_shots, seed=spec.seed
    )

    budget_abort_exc = None
    try:
        for tech in techniques:
            try:
                value = float(
                    _mitigation.apply_technique(
                        tech,
                        circuit,
                        pauli,
                        executor,
                        backend_name,
                        base_shots,
                        spec.seed,
                    )
                )
                row[f"{tech}_value"] = value
                row[f"{tech}_abs_error"] = abs(value - row["ideal"])
                row[f"{tech}_shots"] = int(
                    _mitigation.shots_consumed(tech, base_shots)
                )
            except Exception as exc:  # noqa: BLE001 - isolation is the contract
                prefix = _errors_log_prefix(
                    spec.circuit_id,
                    backend_name,
                    base_shots,
                    tech,
                    list_mode,
                )
                if _is_budget_exceeded(exc):
                    budget_abort_exc = exc
                    with errors_path.open("a", encoding="utf-8") as fh:
                        fh.write(
                            f"{prefix}: SWEEP ABORTED - hardware "
                            f"budget exceeded: {exc!r}\n"
                        )
                    break
                row[f"{tech}_value"] = _NAN
                row[f"{tech}_abs_error"] = _NAN
                row[f"{tech}_shots"] = _NAN
                with errors_path.open("a", encoding="utf-8") as fh:
                    fh.write(f"{prefix}: {exc!r}\n")
    finally:
        _close_executor(executor, f"{spec.circuit_id} @ {backend_name}")

    if budget_abort_exc is not None:
        raise budget_abort_exc

    best_tech, best_cost_tech = _pick_winners(row, techniques, base_shots)
    row["best_technique"] = best_tech
    row[COST_AWARE_COLUMN] = best_cost_tech
    return row


def run_experiment(config: dict, out_dir: Path, num_workers: int = 1) -> pd.DataFrame:
    """Run the full benchmark sweep and build the labeled dataset.

    Config schema::

        {
          "circuits": { ...exact generate_suite config, see circuits.py...
                        (may include the source-level "min_abs_ideal"
                        rejection-sampling key) },
          "backends": ["FakeManilaV2", "FakeLagosV2@x1.5"],
                          # base names from backends.BACKENDS, optionally
                          # with an '@x<scale>' noise-scale suffix (grammar
                          # of backends.parse_backend_name) — each scale is
                          # a distinct noise environment; the full name is
                          # stored in the 'backend' column
          "shots": 4096,                                  # base shots per execution
          "pauli": "auto",                                # 'auto' => 'Z' * n_qubits,
                                                          #  an explicit string, OR a
                                                          #  per-family dict, e.g.
                                                          #  {"ghz_plus": "X",
                                                          #   "default": "auto"}
                                                          #  (single char => repeated
                                                          #  to the circuit width)
          "min_abs_ideal": 0.25,                          # optional (default 0.0 =
                                                          #  off): skip circuits with
                                                          #  |ideal| below this — near-
                                                          #  zero ideals make winner
                                                          #  labels shot-noise
                                                          #  lotteries (science review
                                                          #  2026-07-21)
          "techniques": ["raw", "zne", "cdr", "rem"],     # optional; default
                                                          #  mitigation.TECHNIQUES
        }

    Behaviour contract (crash-safety and reproducibility are the point):
    * ``out_dir`` is created if missing (``mkdir(parents=True, exist_ok=True)``).
    * One work unit = one (circuit, backend) pair. Loop order: circuits in
      generate_suite order (outer), backends in config order (inner).
    * ``out_dir / 'results.csv'`` is APPENDED after EVERY completed unit
      (header written once) — a crash loses at most one unit.
    * Resume: on start, load any existing results.csv and SKIP units whose
      (circuit_id, backend) pair is already present.
    * ``out_dir / 'run_meta.json'`` sidecar written at start: the full config,
      package versions (qiskit, qiskit-aer, mitiq, numpy, sklearn, pandas),
      python version, qemsel.__version__, ISO timestamp.
    * Per-technique isolation: each ``apply_technique`` call is wrapped in
      try/except; on failure the technique's value columns get NaN and one
      line ``'{circuit_id},{backend},{technique}: {exception!r}'`` is appended
      to ``out_dir / 'errors.log'``. A technique failing must NOT kill the run.
    * Low-signal screening: when ``min_abs_ideal`` > 0, units whose exact
      ideal expectation satisfies ``|ideal| < min_abs_ideal`` are SKIPPED
      (no results row) and logged to ``out_dir / 'skipped_low_signal.log'``
      — noise biases expectations toward 0, so near-zero ideals make every
      technique's error pure shot noise and the winner label a lottery.
      Skipped units are cheap to re-check on resume (statevector only).
    * Executors are built once per (backend) via
      ``backends.make_executor(backend, shots, seed=spec.seed)`` per unit
      (seeded by the circuit's seed for reproducibility). If the executor
      exposes ``close()`` (the real-hardware executor's shared Batch) it is
      closed in a ``finally`` after the unit's techniques ran — best-effort,
      a close failure is printed and ignored.
    * HARDWARE BUDGET ABORT: if any technique fails with
      ``qemsel.hardware.HardwareBudgetExceededError`` (directly or anywhere
      in the exception's cause chain — ``apply_technique`` wraps failures in
      ``MitigationError``), the WHOLE sweep aborts cleanly instead of
      spamming one refusal per remaining technique/unit: the abort is
      appended to ``errors.log``, the incomplete unit is dropped (NOT
      appended — resume recomputes it once budget exists again), every
      completed row is preserved, and the function returns normally with the
      rows completed so far. Only reachable on ``ibm_*`` backends.
    * ``out_dir / 'aggregated.csv'`` (seed-averaged labels) is rewritten at
      the end of EVERY run from the complete DataFrame — see
      ``_write_aggregated``: rows grouped by (family, n_qubits, depth,
      backend) across seeds, seed-mean ``feat_*`` columns (so the file is
      directly trainable by ``qemsel.model``), per-technique mean abs_error
      (``<tech>_mean_abs_error``) and seed coverage (``<tech>_n_seeds``),
      ``n_seeds``, and both winner columns recomputed from the MEANS
      restricted to maximum-coverage techniques (lower label noise than
      per-seed winners; a 1-of-3-seed mean cannot outrank 3-seed means).

    Row schema of results.csv / returned DataFrame (EXACT column names):
        circuit_id  (str, '{family}_q{n}_d{d}_s{s}' — CircuitSpec.circuit_id)
        family, n_qubits, depth, seed         (from CircuitSpec)
        backend     (str)
        pauli       (str, the resolved observable string)
        ideal       (float, from qemsel.ideal.ideal_expectation)
        feat_<name> for every features.FEATURE_NAMES entry
                    (e.g. feat_n_qubits ... feat_backend_avg_readout_error)
        <tech>_value, <tech>_abs_error, <tech>_shots
                    for every technique in config (abs_error = |value - ideal|;
                    shots = mitigation.shots_consumed(tech, base_shots);
                    all three NaN if the technique failed)
        best_technique (str: technique with the smallest non-NaN abs_error;
                    '' (empty string) if ALL techniques failed)
        best_technique_cost_aware (str, EXTRA column beyond the base
                    interface: argmin of abs_error * sqrt(shots / base_shots)
                    — see module docstring; '' if all failed)

    Args:
        config: dict as above.
        out_dir: output directory (pathlib.Path).
        num_workers: number of parallel threads to use for simulated backends.

    Returns:
        The COMPLETE DataFrame (previously existing + newly computed rows),
        one row per (circuit, backend) unit.

    Raises:
        ValueError: on invalid config (unknown backend/technique, missing keys).
    """
    (
        circuits_cfg,
        backend_names,
        shots_cfg,
        pauli_cfg,
        techniques,
        min_abs_ideal,
    ) = _validate_config(config)

    # V2 shots axis (INTERFACES.md section V2): a scalar ``shots`` collapses
    # to a single budget with ``list_mode`` False — EVERYTHING below stays
    # byte-identical to V1. A LIST becomes the innermost cross-product
    # dimension (base_shots column, per-budget resume key / labels / logs).
    budgets, list_mode = _normalize_shots(shots_cfg)
    # feature_version (validated in _validate_config; default 1). Version 1
    # keeps the EXACT V1 feature call/columns; version 2 selects the extended
    # feature set and threads the unit budget into extract_features.
    feature_version = int(config.get("feature_version", 1))
    feature_names = _features.FEATURE_NAMES_BY_VERSION[feature_version]

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "results.csv"
    errors_path = out_dir / "errors.log"
    skipped_path = out_dir / "skipped_low_signal.log"
    # Skipped units are re-evaluated (cheaply) every run, so the log is
    # rewritten from scratch each time instead of accumulating duplicates.
    skipped_path.unlink(missing_ok=True)

    columns = _result_columns(
        techniques, feature_names=feature_names, include_base_shots=list_mode
    )
    existing_df = _load_existing(csv_path, columns)
    # Resume key = (circuit_id, backend) in scalar mode; (circuit_id, backend,
    # base_shots) in list mode. An existing results.csv WITHOUT the
    # base_shots column can never reach here in list mode — _load_existing's
    # column-equality check fails first with the "use a fresh out_dir" error.
    if list_mode:
        done_pairs: set[tuple] = set(
            zip(
                existing_df["circuit_id"].astype(str),
                existing_df["backend"].astype(str),
                existing_df[BASE_SHOTS_COLUMN].astype(int),
            )
        )
    else:
        done_pairs = set(
            zip(
                existing_df["circuit_id"].astype(str),
                existing_df["backend"].astype(str),
            )
        )
    _write_run_meta(config, out_dir, n_existing=len(existing_df))

    suite = _circuits.generate_suite(circuits_cfg)

    # Fail fast if a resolved pauli cannot fit its circuit (only explicit
    # multi-character specs can mismatch; 'auto' and single chars always fit).
    bad = [
        (spec.circuit_id, _resolve_pauli(pauli_cfg, spec.family, circ.num_qubits))
        for circ, spec in suite
        if len(_resolve_pauli(pauli_cfg, spec.family, circ.num_qubits))
        != circ.num_qubits
    ]
    if bad:
        raise ValueError(
            f"pauli spec resolves to the wrong length for circuits "
            f"{bad[:5]!r}{'...' if len(bad) > 5 else ''}; "
            "use pauli: 'auto' (or single-character specs) for mixed-size suites"
        )

    n_units = len(suite) * len(backend_names) * len(budgets)
    budget_txt = (
        f" x {len(budgets)} shot-budgets" if list_mode else ""
    )

    t_start = time.monotonic()
    new_rows: list[dict[str, Any]] = []
    budget_abort_exc: BaseException | None = None

    # Pre-scan and generate tasks, filtering out skipped low-signal pairs
    tasks = []
    total_skipped = 0
    for circuit, spec in suite:
        for backend_name in backend_names:
            pauli = _resolve_pauli(pauli_cfg, spec.family, circuit.num_qubits)
            ideal_value = float(_ideal.ideal_expectation(circuit, pauli))

            if min_abs_ideal > 0.0 and abs(ideal_value) < min_abs_ideal:
                with skipped_path.open("a", encoding="utf-8") as fh:
                    fh.write(
                        f"{spec.circuit_id},{backend_name},{pauli},"
                        f"ideal={ideal_value:+.6f},"
                        f"min_abs_ideal={min_abs_ideal}\n"
                    )
                total_skipped += len(budgets)
                continue

            for base_shots in budgets:
                key = (spec.circuit_id, backend_name, base_shots) if list_mode else (spec.circuit_id, backend_name)
                if key in done_pairs:
                    continue
                tasks.append((spec, circuit, backend_name, pauli, base_shots, ideal_value, key))

    n_pending = len(tasks)
    print(
        f"run_experiment: {n_units} units ({len(suite)} circuits x "
        f"{len(backend_names)} backends{budget_txt}), {len(done_pairs)} "
        f"already in {csv_path.name}, {total_skipped} skipped low-signal, "
        f"{n_pending} pending execution. out_dir={out_dir}, num_workers={num_workers}",
        flush=True,
    )

    if n_pending > 0:
        if num_workers > 1 and not any(b.startswith("ibm_") for b in backend_names):
            import concurrent.futures
            # Run in parallel using ThreadPoolExecutor (thread-safe, does not require pickling)
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures_map = {
                    executor.submit(
                        _run_single_unit,
                        spec,
                        circuit,
                        backend_name,
                        pauli,
                        base_shots,
                        ideal_value,
                        feature_version,
                        feature_names,
                        techniques,
                        list_mode,
                        errors_path,
                    ): (key, time.monotonic())
                    for spec, circuit, backend_name, pauli, base_shots, ideal_value, key in tasks
                }

                completed_count = 0
                for future in concurrent.futures.as_completed(futures_map):
                    completed_count += 1
                    key, t_unit = futures_map[future]
                    unit_label = f"{key[0]} @ {key[1]}" + (f" @ {key[2]} shots" if list_mode else "")
                    try:
                        row = future.result()
                        _append_row(csv_path, row, columns)
                        new_rows.append(row)
                        done_pairs.add(key)

                        elapsed = time.monotonic() - t_start
                        print(
                            f"[{completed_count}/{n_pending}] {unit_label}: "
                            f"best={row['best_technique'] or '<all failed>'} "
                            f"cost-aware={row[COST_AWARE_COLUMN] or '<all failed>'} "
                            f"(unit {time.monotonic() - t_unit:.1f}s, "
                            f"elapsed {elapsed:.1f}s)",
                            flush=True,
                        )
                    except Exception as exc:
                        if _is_budget_exceeded(exc):
                            print(f"ABORTING SWEEP — hardware QPU budget exceeded: {exc!r}", flush=True)
                            budget_abort_exc = exc
                            break
                        else:
                            print(f"Warning: unit {unit_label} failed with: {exc!r}", flush=True)
        else:
            # Run sequentially
            completed_count = 0
            for spec, circuit, backend_name, pauli, base_shots, ideal_value, key in tasks:
                completed_count += 1
                t_unit = time.monotonic()
                unit_label = f"{key[0]} @ {key[1]}" + (f" @ {key[2]} shots" if list_mode else "")
                try:
                    row = _run_single_unit(
                        spec,
                        circuit,
                        backend_name,
                        pauli,
                        base_shots,
                        ideal_value,
                        feature_version,
                        feature_names,
                        techniques,
                        list_mode,
                        errors_path,
                    )
                    _append_row(csv_path, row, columns)
                    new_rows.append(row)
                    done_pairs.add(key)

                    elapsed = time.monotonic() - t_start
                    print(
                        f"[{completed_count}/{n_pending}] {unit_label}: "
                        f"best={row['best_technique'] or '<all failed>'} "
                        f"cost-aware={row[COST_AWARE_COLUMN] or '<all failed>'} "
                        f"(unit {time.monotonic() - t_unit:.1f}s, "
                        f"elapsed {elapsed:.1f}s)",
                        flush=True,
                    )
                except Exception as exc:
                    if _is_budget_exceeded(exc):
                        print(f"ABORTING SWEEP — hardware QPU budget exceeded: {exc!r}", flush=True)
                        budget_abort_exc = exc
                        break
                    else:
                        print(f"Warning: unit {unit_label} failed with: {exc!r}", flush=True)

    total_elapsed = time.monotonic() - t_start
    status = "ABORTED (hardware budget)" if budget_abort_exc is not None else "finished"
    print(
        f"run_experiment: {status} — {len(new_rows)} new rows, "
        f"{len(existing_df)} resumed rows, {total_elapsed:.1f}s total",
        flush=True,
    )

    if new_rows:
        new_df = pd.DataFrame(new_rows, columns=columns)
        if existing_df.empty:
            full_df = new_df
        else:
            full_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        full_df = existing_df

    # Seed-averaged aggregate (derived data, overwritten every run — also
    # after aborts/resumes so it always reflects the on-disk results.csv). In
    # list mode the per-budget base is taken from each group's own key.
    _write_aggregated(
        full_df,
        techniques,
        None if list_mode else budgets[0],
        out_dir,
        feature_names=feature_names,
        list_mode=list_mode,
    )

    return full_df
