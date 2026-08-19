"""Quantum error mitigation techniques, behind one uniform interface.

All techniques take the SAME executor produced by
``qemsel.backends.make_executor`` (signature ``executor(circuit, pauli) -> float``)
so raw vs mitigated comparisons see identical noise. The one deliberate
exception is ``raw_plus`` (below), which rebuilds the executor at a larger
shot budget — that is its entire point.

Implementation notes (mitiq 1.0.0):

* ``raw_plus``: the EMPIRICAL equal-budget baseline (stats review 2026-07-21,
  previously deferred): a single unmitigated execution at
  ``RAW_PLUS_MULTIPLIER * base_shots`` shots, where ``RAW_PLUS_MULTIPLIER``
  equals the LARGEST multiplier of any real technique (CDR's 11), so
  "just take more shots" is compared to every technique at the most
  expensive technique's budget. It rebuilds a fresh executor via
  ``qemsel.backends.make_executor(backend_name, base_shots *
  RAW_PLUS_MULTIPLIER, seed)`` because the passed executor is already bound
  to ``base_shots`` (calling a seeded executor N times would return N
  identical values — averaging them is statistically a 1x measurement
  dressed up as 11x). If the rebuilt executor exposes ``close()`` (the real
  hardware executor does) it is closed in a ``finally``.

* ``zne``: ``mitiq.zne.execute_with_zne`` with a ``RichardsonFactory`` over
  ``ZNE_SCALE_FACTORS`` and *seeded* ``fold_gates_at_random`` (unseeded folding
  re-randomizes per call and makes Richardson extrapolation flaky under shot
  noise).
* ``cdr``: ``mitiq.cdr.execute_with_cdr`` with
  ``CDR_NUM_TRAINING_CIRCUITS`` near-Clifford training circuits and
  ``qemsel.ideal.ideal_expectation`` as the noiseless simulator. The circuit
  is first transpiled to ``CDR_BASIS_GATES`` because CDR requires every
  non-Clifford gate to be an ``rz`` rotation. FAIL-LOUD GUARDS (science
  review 2026-07-21): CDR raises :class:`MitigationError` instead of
  returning a classically-simulated value when (a) the compiled circuit is
  fully Clifford (mitiq would short-circuit and return the ideal simulator
  value — zero error by construction, no quantum execution at all), or
  (b) every near-Clifford training circuit has the same ideal expectation
  (the regression collapses to a constant equal to the classical value and
  ignores the noisy measurement entirely). Without these guards 2 of the 5
  circuit families (ghz_plus, near_clifford) recorded fake ~1e-16 CDR
  "wins" that measured classical simulability, not mitigation quality.
* ``rem``: expectation-level tensored readout-error inversion, calibrated
  through the SAME executor with two basis-state circuits (|0...0> and
  |1...1> on the observable's support). NOTE: ``mitiq.rem.execute_with_rem``
  requires an executor returning ``mitiq.MeasurementResult`` bitstrings and is
  therefore incompatible with the uniform float-returning executor contract;
  this is the calibration-circuit variant of REM proven in
  ``spikes/spike_rem.py``, specialized to the single measured Pauli (see
  ``_apply_rem`` for the math and its accuracy limits).

Cost model: the executor-invocation counts of each technique are fixed and
derived from the module-level settings below; ``SHOT_MULTIPLIER`` is computed
from them so it can never drift out of sync with the code.

Plugging in your own CDR regressor (roadmap item 5 — the thesis ablation)
-------------------------------------------------------------------------
Two routes, depending on the regressor:

1. **Parametric fits (polynomial, custom curve shapes):** mitiq's
   ``execute_with_cdr`` accepts ``fit_function=`` and
   ``num_fit_parameters=`` (default: ``linear_fit_function``, 2 params).
   ``fit_function`` is a scipy ``curve_fit``-style callable
   ``f(x, *params) -> y``. Set the module constants ``CDR_FIT_FUNCTION``
   and ``CDR_NUM_FIT_PARAMETERS`` below — ``_apply_cdr`` passes them
   through, so a polynomial ablation is a one-line change.
2. **sklearn regressors (ridge, random forest, ...):** these are NOT
   curve_fit-parametric, so they cannot go through ``fit_function``.
   Instead bypass mitiq's fit inside ``_apply_cdr``: call
   ``mitiq.cdr.generate_training_circuits(compiled, n, fraction,
   random_state=seed)``, run each training circuit through both the noisy
   ``executor`` and ``qemsel.ideal.ideal_expectation``, fit any sklearn
   regressor noisy -> ideal, then apply it to the target's noisy value.
   (``_apply_cdr`` already pre-generates the training set for its
   degeneracy guard — extend from there.)
"""

from __future__ import annotations

import functools
import math
from typing import Callable

import numpy as np
from mitiq import zne
from mitiq.cdr import execute_with_cdr
from mitiq.zne import RichardsonFactory
from mitiq.zne.scaling import fold_gates_at_random
from qiskit import QuantumCircuit, transpile

# NOTE: backends does NOT import mitigation (hardware.py imports both, but
# backends only imports hardware lazily inside functions), so this
# module-level import cannot create a cycle. Accessed as a module attribute
# (``_backends.make_executor``) so tests can monkeypatch it.
from qemsel import backends as _backends
from qemsel import ideal as _ideal

#: The techniques compared in this project, in canonical order.
#: 'raw' = unmitigated baseline at base shots; 'raw_plus' = unmitigated
#: baseline at RAW_PLUS_MULTIPLIER * base shots (empirical equal-budget
#: control). These strings are the class labels for the ML model — never
#: rename without updating INTERFACES.md.
TECHNIQUES: list[str] = ["raw", "raw_plus", "zne", "cdr", "rem"]

# ---------------------------------------------------------------------------
# Technique settings (module-level constants so reviewers can audit the cost
# model; SHOT_MULTIPLIER below is derived from these).
# ---------------------------------------------------------------------------

#: ZNE noise-scale factors for Richardson extrapolation (one noisy execution
#: of the folded circuit per factor).
ZNE_SCALE_FACTORS: tuple[float, ...] = (1.0, 2.0, 3.0)

#: Number of near-Clifford training circuits CDR executes on the noisy
#: backend (in addition to one execution of the target circuit).
CDR_NUM_TRAINING_CIRCUITS: int = 10

#: Fraction of the target circuit's non-Clifford gates KEPT non-Clifford in
#: each CDR training circuit (mitiq convention: kept, not replaced).
CDR_FRACTION_NON_CLIFFORD: float = 0.2

#: Basis the circuit is compiled to before CDR. CDR requires all non-Clifford
#: content in rz gates; this set also matches the Aer noise-model basis so the
#: executor's internal transpile is a near-no-op. Verified to survive mitiq's
#: qiskit -> cirq qasm round-trip.
CDR_BASIS_GATES: tuple[str, ...] = ("rz", "sx", "x", "cx")

#: Optional custom CDR fit function (scipy curve_fit style: f(x, *params)).
#: None = mitiq's default linear_fit_function. See the module docstring
#: ("Plugging in your own CDR regressor") — parametric fits go here;
#: sklearn regressors need the generate_training_circuits route instead.
CDR_FIT_FUNCTION = None

#: Number of free parameters of CDR_FIT_FUNCTION (required by mitiq when a
#: custom fit_function is supplied). Ignored while CDR_FIT_FUNCTION is None.
CDR_NUM_FIT_PARAMETERS: int | None = None

#: Training-circuit ideal values must span at least this range; below it the
#: CDR regression is degenerate (constant fit = classical simulation) and
#: _apply_cdr raises MitigationError instead of returning a fake result.
CDR_MIN_TRAINING_IDEAL_SPREAD: float = 1e-9

#: REM runs two calibration circuits (|0...0> and |1...1> on the support)
#: through the executor, in addition to one execution of the target circuit.
REM_NUM_CALIBRATION_CIRCUITS: int = 2

#: If the calibrated readout damping factor is smaller than this, the
#: inversion would blow up numerically; REM raises MitigationError instead.
#: 0.02 (was 1e-6 — code review 2026-07-21): a damping of 0.02 already
#: amplifies shot noise 50x on inversion; anything smaller (e.g. multi-qubit
#: supports crossing FakeLagosV2's 46%-readout-error q2) turns REM values
#: into amplified coin flips that poison the winner labels. Near-singular
#: readout must fail loudly as MitigationError, not record noise.
REM_MIN_DAMPING: float = 0.02

#: Shot multiplier of the 'raw_plus' equal-budget baseline: a single
#: unmitigated execution at this many times the base shots. Set to the
#: LARGEST multiplier of any real technique (CDR's 1 + 10 = 11) so the
#: "just take more shots instead of mitigating" control is compared at the
#: most expensive technique's budget. Derived, so it can never drift.
RAW_PLUS_MULTIPLIER: int = max(
    1,  # raw
    len(ZNE_SCALE_FACTORS),
    1 + CDR_NUM_TRAINING_CIRCUITS,
    1 + REM_NUM_CALIBRATION_CIRCUITS,
)

#: Multiplier on base_shots representing total quantum-resource cost.
#: Derived from the settings above; MUST stay consistent with what
#: apply_technique actually executes (see the per-technique helpers):
#:   raw: 1 execution.
#:   raw_plus: 1 execution at RAW_PLUS_MULTIPLIER * base shots.
#:   zne: len(ZNE_SCALE_FACTORS) executions (scale factors 1, 2, 3).
#:   cdr: 1 target execution + CDR_NUM_TRAINING_CIRCUITS training executions.
#:   rem: 1 execution + REM_NUM_CALIBRATION_CIRCUITS calibration executions.
SHOT_MULTIPLIER: dict[str, int] = {
    "raw": 1,
    "raw_plus": RAW_PLUS_MULTIPLIER,
    "zne": len(ZNE_SCALE_FACTORS),
    "cdr": 1 + CDR_NUM_TRAINING_CIRCUITS,
    "rem": 1 + REM_NUM_CALIBRATION_CIRCUITS,
}

# ===========================================================================
# V2 ADDITIONS (INTERFACES.md section "V2"; builder-mitigation / B1 implements
# every body below that raises NotImplementedError). The V1 constants above
# are FROZEN: ``TECHNIQUES`` and ``SHOT_MULTIPLIER`` keep exactly their five
# entries — regression tests pin both — so all new capability lives behind
# the new ``*_V2`` names. Existing techniques must stay byte-identical.
# ===========================================================================

#: V2 technique menu: the frozen V1 five plus three ADDITIVE techniques.
#: These strings are ML class labels — never rename. This is NOT the default
#: technique list: ``experiment`` keeps defaulting to ``TECHNIQUES``; configs
#: opt in to the new techniques by listing them explicitly.
#:
#: * ``zne_fr``    — fixed-Richardson ZNE aligned to the variant analyzed in
#:                   Scavino arXiv:2605.08251 (the Angle 3 boundary needs
#:                   this alignment, else the overlay is apples-to-oranges).
#: * ``cdr_ridge`` — CDR with a RidgeCV regressor (LOO-selected alpha;
#:                   Korolev arXiv:2606.02697 anchor: regularized-linear
#:                   usually wins).
#: * ``cdr_rf``    — CDR with a RandomForest regressor (the nonlinear
#:                   contender of the Angle 2 overfitting map).
TECHNIQUES_V2: list[str] = TECHNIQUES + ["zne_fr", "cdr_ridge", "cdr_rf"]

#: Noise-scale nodes of the fixed-Richardson variant. The Richardson
#: coefficients are FIXED a priori from these nodes (Lagrange-at-zero, see
#: :func:`richardson_coefficients`) — NOT refit from the measured values.
#: SPIKE MAY ADJUST the node values; qemsel.boundary reads this constant, so
#: the theory curve and the implementation can never disagree on the nodes.
#:
#: SPIKE-RETUNED to the TWO-POINT rule (1.0, 3.0) analyzed in Scavino
#: arXiv:2605.08251 (notes/spike-boundary.md §2 + the BOUNDARY SPIKE): the
#: paper's headline theory and the K_q variance penalty (4·nu at q=0, 5·nu at
#: q=1) are derived for the k=1 nodes (1, 3) with fixed coefficients
#: (3/2, -1/2) and a uniform B/2+B/2 split — NOT the three-point (1, 2, 3)
#: our mitiq `zne` uses. The overlay in qemsel.boundary is only
#: apples-to-apples if `zne_fr` and the theory share these exact nodes; since
#: boundary.py imports both this constant and :func:`richardson_coefficients`,
#: they can never drift.
ZNE_FR_SCALE_FACTORS: tuple[float, ...] = (1.0, 3.0)

#: Per-level shot allocation of zne_fr. 'equal_split' (Scavino's analyzed
#: allocation): the base budget B is SPLIT over the levels — each level runs
#: at ``base_shots // len(ZNE_FR_SCALE_FACTORS)`` shots, so zne_fr consumes
#: ONE base budget total and the zne_fr-vs-raw comparison is exactly the
#: equal-budget ΔMSE(ε, B) the boundary formula describes. The only other
#: allowed value is 'full' (every level at base_shots — multiplier becomes
#: len(nodes)). SPIKE DECIDES; SHOT_MULTIPLIER_V2 derives from this constant
#: so the cost model can never drift.
ZNE_FR_SHOT_ALLOCATION: str = "equal_split"

#: Noise-amplification method for zne_fr: 'global' = deterministic
#: ``mitiq.zne.scaling.fold_global`` (no per-call randomness — the analytic
#: variance side assumes deterministic amplification; seeded random folding
#: would add folding variance the formula does not model). SPIKE MAY ADJUST.
ZNE_FR_FOLD_METHOD: str = "global"

#: DEPRECATED (2026-07-23 findings-applier): the fixed Ridge(alpha=1.0) on a
#: single UNSTANDARDIZED feature (noisy expectation in [-1, 1], Sxx ~ 0.01-1
#: over ~10 training points) shrank the CDR slope 43-99% and produced a
#: "Ridge-CDR" 3.6-6.7x WORSE than raw — a penalty-scale artifact that would
#: have inverted the Korolev regularized-linear-usually-wins anchor Angle 2
#: relies on (verified: layered_random n2 d4 @ FakeManilaV2, cdr_ridge error
#: 0.288 vs cdr 0.0069 vs raw 0.079). Kept only so old notes/configs that
#: mention the name still resolve; the implementation now uses
#: ``CDR_RIDGE_ALPHAS`` below.
CDR_RIDGE_ALPHA: float = 1.0

#: Ridge regularization grid for cdr_ridge: ``sklearn.linear_model.RidgeCV``
#: over alphas = logspace(-6, 3, 19), selected by RidgeCV's DETERMINISTIC
#: efficient leave-one-out CV (cv=None; no randomness, no extra noisy
#: executions). This is the cdr-nl spike's verified recommendation
#: (notes/spike-cdr-nl.md section "Regressors"): on well-spread 1-D training
#: data LOO-CV picks a near-zero alpha and Ridge-CDR reproduces linear CDR,
#: which is exactly Korolev's regularized-linear baseline behavior.
CDR_RIDGE_ALPHAS: tuple[float, ...] = tuple(
    float(a) for a in np.logspace(-6.0, 3.0, 19)
)

#: RandomForestRegressor settings for cdr_rf (random_state comes from the
#: per-call seed, never a global).
CDR_RF_N_ESTIMATORS: int = 100
CDR_RF_MAX_DEPTH: int | None = None

#: Training-circuit count for the sklearn-CDR variants. Defaults to CDR's
#: own count so the three cdr variants differ ONLY in the regressor (the
#: Angle 2 control). The overfitting-map sweep varies THIS constant (via a
#: driver script), never CDR_NUM_TRAINING_CIRCUITS.
CDR_SKLEARN_NUM_TRAINING_CIRCUITS: int = CDR_NUM_TRAINING_CIRCUITS

#: V2 cost model: superset of the FROZEN ``SHOT_MULTIPLIER`` (V1 keys keep
#: identical values). ``shots_consumed``/``apply_technique`` validate against
#: the V2 names; values MUST stay truthful to what the helpers execute.
SHOT_MULTIPLIER_V2: dict[str, int] = {
    **SHOT_MULTIPLIER,
    # equal_split: all levels share ONE base budget; 'full': one budget/level.
    "zne_fr": (
        1
        if ZNE_FR_SHOT_ALLOCATION == "equal_split"
        else len(ZNE_FR_SCALE_FACTORS)
    ),
    "cdr_ridge": 1 + CDR_SKLEARN_NUM_TRAINING_CIRCUITS,
    "cdr_rf": 1 + CDR_SKLEARN_NUM_TRAINING_CIRCUITS,
}


def richardson_coefficients(scale_factors: tuple[float, ...]) -> tuple[float, ...]:
    """Fixed Richardson extrapolation coefficients for the given noise nodes.

    The Lagrange-interpolation-at-zero coefficients c_k for nodes s_k:
    ``c_k = prod_{j != k} s_j / (s_j - s_k)``, satisfying ``sum c_k == 1``
    and ``sum c_k * s_k^m == 0`` for m = 1..len(nodes)-1. Deterministic and
    computable a priori — these are the "fixed coefficients" of Scavino's
    analyzed variant (cf. Mohammadipour & Li arXiv:2502.20673), used both by
    ``_apply_zne_fr`` (the estimate is ``sum_k c_k * E_k``) and by
    ``qemsel.boundary.variance_k_q`` (the a-priori variance side K_q). This
    is the SINGLE public source of truth for the coefficients — boundary.py
    imports it from here (mitigation never imports boundary).

    Args:
        scale_factors: strictly increasing, distinct, all >= 1.0, first
            element 1.0 (validated; ValueError otherwise).

    Returns:
        tuple of floats, same length/order as ``scale_factors``.

    Raises:
        ValueError: nodes not distinct / not >= 1.0 / fewer than 2.
    """
    nodes = tuple(float(s) for s in scale_factors)
    if len(nodes) < 2:
        raise ValueError(
            f"need at least 2 noise-scale nodes for Richardson extrapolation, "
            f"got {len(nodes)}: {scale_factors!r}"
        )
    if any((not math.isfinite(s)) or s < 1.0 for s in nodes):
        raise ValueError(
            f"all noise-scale nodes must be finite and >= 1.0, got "
            f"{scale_factors!r}"
        )
    if len(set(nodes)) != len(nodes):
        raise ValueError(
            f"noise-scale nodes must be distinct, got {scale_factors!r}"
        )
    # Lagrange basis at 0: c_k = prod_{j != k} s_j / (s_j - s_k). Satisfies
    # sum c_k == 1 and sum c_k s_k^m == 0 for m = 1..len(nodes)-1.
    lam = np.asarray(nodes, dtype=float)
    coeffs = np.empty_like(lam)
    for k in range(lam.size):
        others = np.delete(lam, k)
        coeffs[k] = np.prod(others / (others - lam[k]))
    return tuple(float(c) for c in coeffs)


def estimator_sigma(name: str, value: float, base_shots: float) -> float:
    """Shot-noise standard deviation of one TECHNIQUE'S estimate (V2, additive).

    ``stats.sigma_shot(value, shots)`` is only valid for a single direct
    measurement whose ``shots`` all pool into one average. The mitigated
    techniques do NOT pool: extrapolation/correction estimators AMPLIFY
    variance (findings-applier 2026-07-23; the amplification factor
    ``||gamma||_1**2`` is the one cited in docs/LITERATURE.md via
    Mohammadipour & Li / Krebsbach). Feeding the consumed-budget ledger
    (``<tech>_shots`` = multiplier x base) into ``sigma_shot`` understated
    sigma ~2-8x for zne/zne_fr/cdr/rem and corrupted the significance labels
    and tie flags. This function is the corrected, per-technique model:

    * ``raw``:       one execution at B                -> Var = v/B
    * ``raw_plus``:  one execution at 11B              -> Var = v/(11B)
    * ``zne``:       Richardson over ``ZNE_SCALE_FACTORS`` (one execution of
                     B shots PER level), fixed refit coefficients c_k ->
                     Var = sum(c_k^2) * v/B  (19x for nodes (1,2,3))
    * ``zne_fr``:    fixed coefficients over ``ZNE_FR_SCALE_FACTORS`` with
                     the ``ZNE_FR_SHOT_ALLOCATION`` split ->
                     Var = sum(c_k^2 / pi_k) * v/B (5x for (1,3) equal-split)
    * ``cdr``/``cdr_ridge``/``cdr_rf``: the target is measured ONCE at B
                     shots; training-circuit shots do NOT average the target
                     down -> Var >= v/B (regression/fit noise ignored — a
                     documented conservative LOWER bound on sigma).
    * ``rem``:       raw measured at B then divided by the calibrated
                     damping (>= 1/damping amplification, damping unknown
                     here) -> Var >= v/B (same conservative lower bound).

    with ``v = 1 - min(value**2, 1)`` and B = ``base_shots``. The zne/zne_fr
    coefficients come from :func:`richardson_coefficients` on the LIVE
    module constants, so this can never drift from what the techniques
    execute (same single-source-of-truth rule as ``qemsel.boundary``).

    Overshoot rule: a mitigated estimate with ``|value| > 1`` is exactly the
    variance-blow-up case, so this returns ``math.inf`` (forcing any
    significance test toward 'tie') — NEVER the sigma=0 that clamping the
    variance term would give (sigma=0 on the blow-up rows made ANY margin
    'significant', the worst possible direction).

    Args:
        name: one of ``TECHNIQUES_V2``.
        value: the technique's estimate of the Pauli expectation.
        base_shots: the run's BASE shot budget B (NOT the consumed ledger;
            divide ``<tech>_shots`` by ``SHOT_MULTIPLIER_V2[name]`` first).

    Returns:
        float sigma >= 0, or ``math.inf`` for |value| > 1.

    Raises:
        ValueError: unknown technique, NaN value, or base_shots <= 0.
    """
    if name not in TECHNIQUES_V2:
        raise ValueError(
            f"unknown technique {name!r}; expected one of {TECHNIQUES_V2}"
        )
    v = float(value)
    b = float(base_shots)
    if math.isnan(v):
        raise ValueError("estimator_sigma: value is NaN")
    if math.isnan(b) or b <= 0:
        raise ValueError(f"estimator_sigma: base_shots must be > 0 (got {base_shots!r})")
    if abs(v) > 1.0:
        return math.inf
    var_term = 1.0 - v * v
    if name == "raw_plus":
        factor = 1.0 / RAW_PLUS_MULTIPLIER
    elif name == "zne":
        coeffs = richardson_coefficients(ZNE_SCALE_FACTORS)
        factor = float(sum(c * c for c in coeffs))  # each level at B shots
    elif name == "zne_fr":
        coeffs = richardson_coefficients(ZNE_FR_SCALE_FACTORS)
        n_levels = len(ZNE_FR_SCALE_FACTORS)
        if ZNE_FR_SHOT_ALLOCATION == "equal_split":
            factor = float(sum(c * c for c in coeffs)) * n_levels  # B/n per level
        else:  # 'full': one base budget per level
            factor = float(sum(c * c for c in coeffs))
    else:
        # raw, cdr, cdr_ridge, cdr_rf, rem: one B-shot estimate of the target
        # (conservative lower bound for the correction-based techniques).
        factor = 1.0
    return float(math.sqrt(factor * var_term / b))


class MitigationError(RuntimeError):
    """A mitigation technique failed internally.

    Carries the technique name so ``run_experiment`` can log which technique
    produced the failure (it catches this, records NaN, and moves on).

    Attributes:
        technique: name of the technique that failed (one of ``TECHNIQUES``).
    """

    def __init__(self, technique: str, message: str) -> None:
        super().__init__(f"[{technique}] {message}")
        self.technique = technique


def apply_technique(
    name: str,
    circuit: QuantumCircuit,
    pauli: str,
    executor: Callable[[QuantumCircuit, str], float],
    backend_name: str,
    shots: int,
    seed: int,
) -> float:
    """Estimate <pauli> for ``circuit`` using one (possibly mitigated) technique.

    Args:
        name: one of ``TECHNIQUES_V2`` (the V1 ``TECHNIQUES`` five plus
            'zne_fr' / 'cdr_ridge' / 'cdr_rf'; V1 names behave exactly as
            before — byte-identical dispatch path).
        circuit: QuantumCircuit WITHOUT final measurements. Never mutated.
        pauli: Pauli string, qemsel convention (pauli[i] acts on qubit i).
        executor: noisy executor from ``qemsel.backends.make_executor`` —
            already bound to a backend/shots/seed. mitiq needs a 1-arg
            callable; wrapped internally as ``lambda circ: executor(circ, pauli)``.
            NOT used by 'raw_plus' (see below).
        backend_name: same backend the executor was built for. Metadata only
            for 'rem' (it self-calibrates through the executor), but
            LOAD-BEARING for 'raw_plus', which rebuilds a fresh executor for
            this backend at a bigger shot budget.
        shots: base shots per single execution (informational for most
            techniques — the executor is already bound to this and is NOT
            rebuilt; 'raw_plus' uses it to size its rebuilt executor at
            ``RAW_PLUS_MULTIPLIER * shots``).
        seed: seed for any technique-internal randomness (ZNE gate-folding,
            CDR training-circuit sampling, CDR pre-transpile). Same inputs =>
            same output modulo simulator shot noise, which is itself seeded
            via the executor.

    Technique implementations:
        'raw': returns ``executor(circuit, pauli)`` directly.
        'raw_plus': one unmitigated execution at ``RAW_PLUS_MULTIPLIER *
            shots`` through a freshly built executor for ``backend_name``
            (same seed) — the empirical equal-budget baseline. The passed
            ``executor`` is not invoked.
        'zne': ``mitiq.zne.execute_with_zne`` with RichardsonFactory over
            ``ZNE_SCALE_FACTORS``, seeded ``fold_gates_at_random``.
        'cdr': ``mitiq.cdr.execute_with_cdr`` with
            ``CDR_NUM_TRAINING_CIRCUITS`` near-Clifford training circuits;
            the noiseless simulator reuses ``qemsel.ideal.ideal_expectation``
            (wrapped to 1-arg).
        'rem': tensored readout-error inversion calibrated through the
            executor with two basis-state circuits (see ``_apply_rem``).
        'zne_fr' (V2): fixed-Richardson ZNE, Scavino-aligned — see
            ``_apply_zne_fr``.
        'cdr_ridge' / 'cdr_rf' (V2): CDR with a sklearn Ridge / RandomForest
            regressor — see ``_apply_cdr_sklearn``.

    Returns:
        float estimate of <pauli> (mitigated values may fall outside
        [-1, 1]; NOT clipped — the experiment records the value as-is).

    Raises:
        ValueError: if ``name`` not in ``TECHNIQUES_V2``.
        MitigationError: any technique-internal failure is wrapped in a
            ``MitigationError`` carrying the technique name (original
            exception chained as ``__cause__``); ``run_experiment`` catches
            it and records NaN.
        NotImplementedError: while a V2 technique is still an architect
            stub (passes through unwrapped so builders see it verbatim;
            run_experiment's per-technique isolation still records NaN).
    """
    if name not in TECHNIQUES_V2:
        raise ValueError(
            f"unknown technique {name!r}; expected one of {TECHNIQUES_V2}"
        )
    try:
        if name == "raw":
            return float(executor(circuit, pauli))
        if name == "raw_plus":
            return _apply_raw_plus(circuit, pauli, backend_name, shots, seed)
        if name == "zne":
            return _apply_zne(circuit, pauli, executor, seed)
        if name == "cdr":
            return _apply_cdr(circuit, pauli, executor, seed)
        if name == "rem":
            return _apply_rem(circuit, pauli, executor)
        if name == "zne_fr":
            return _apply_zne_fr(
                circuit, pauli, executor, backend_name, shots, seed
            )
        # remaining validated names: 'cdr_ridge', 'cdr_rf'
        return _apply_cdr_sklearn(circuit, pauli, executor, seed, regressor=name)
    except (MitigationError, NotImplementedError):
        raise
    except Exception as exc:
        raise MitigationError(name, f"{type(exc).__name__}: {exc}") from exc


def shots_consumed(name: str, base_shots: int) -> int:
    """Total quantum-resource cost (shots) of running technique ``name`` once.

    Returns ``base_shots * SHOT_MULTIPLIER_V2[name]`` (V2 is a strict
    superset of the frozen V1 ``SHOT_MULTIPLIER`` with identical values on
    the V1 keys, so V1 behavior is byte-identical). Used by
    experiment/report for cost-normalized comparisons between techniques.

    Args:
        name: one of ``TECHNIQUES_V2``.
        base_shots: shots per single circuit execution.

    Returns:
        int total shots.

    Raises:
        ValueError: if ``name`` not in ``TECHNIQUES_V2``.
    """
    if name not in TECHNIQUES_V2:
        raise ValueError(
            f"unknown technique {name!r}; expected one of {TECHNIQUES_V2}"
        )
    return int(base_shots) * SHOT_MULTIPLIER_V2[name]


# ---------------------------------------------------------------------------
# Private per-technique helpers.
# ---------------------------------------------------------------------------


def _apply_raw_plus(
    circuit: QuantumCircuit,
    pauli: str,
    backend_name: str,
    base_shots: int,
    seed: int,
) -> float:
    """Equal-budget baseline: one raw execution at 11x (RAW_PLUS_MULTIPLIER)
    the base shots.

    Why a REBUILT executor instead of reusing the passed one: the uniform
    executor is bound to ``base_shots`` and is fully seeded, so calling it
    ``RAW_PLUS_MULTIPLIER`` times on the same circuit returns the SAME value
    every time — averaging those is a 1x-shot measurement pretending to be
    11x. A fresh executor at ``RAW_PLUS_MULTIPLIER * base_shots`` shots is
    the honest "spend CDR's whole budget on plain averaging" control the
    cost-aware analysis needs (stats review 2026-07-21, previously deferred).

    ``make_executor`` is looked up as a module attribute so tests can
    monkeypatch ``qemsel.backends.make_executor``. If the built executor
    exposes ``close()`` (the real-hardware executor's shared Batch) it is
    closed in a ``finally``.

    Executor invocations: 1 (at ``RAW_PLUS_MULTIPLIER * base_shots`` shots);
    the PASSED executor is invoked 0 times.
    """
    boosted = _backends.make_executor(
        backend_name, int(base_shots) * RAW_PLUS_MULTIPLIER, seed
    )
    try:
        return float(boosted(circuit, pauli))
    finally:
        close = getattr(boosted, "close", None)
        if callable(close):
            close()


def _apply_zne(
    circuit: QuantumCircuit,
    pauli: str,
    executor: Callable[[QuantumCircuit, str], float],
    seed: int,
) -> float:
    """Zero-noise extrapolation: Richardson over ``ZNE_SCALE_FACTORS``.

    Uses seeded ``fold_gates_at_random`` so results are reproducible
    (unseeded folding re-randomizes every call, which combined with shot
    noise makes Richardson extrapolation flaky run-to-run). The executor
    transpiles internally with optimization_level=0, so folded G Gdag G
    sequences are not simplified away.

    Executor invocations: ``len(ZNE_SCALE_FACTORS)``.
    """
    factory = RichardsonFactory(scale_factors=list(ZNE_SCALE_FACTORS))
    scale_noise = functools.partial(fold_gates_at_random, seed=seed)
    mitigated = zne.execute_with_zne(
        circuit.copy(),  # defensive copy: caller's circuit is never mutated
        lambda folded: executor(folded, pauli),
        factory=factory,
        scale_noise=scale_noise,
    )
    return float(np.real(mitigated))


def _apply_cdr(
    circuit: QuantumCircuit,
    pauli: str,
    executor: Callable[[QuantumCircuit, str], float],
    seed: int,
) -> float:
    """Clifford data regression via ``mitiq.cdr.execute_with_cdr``.

    The circuit is first compiled to ``CDR_BASIS_GATES`` (all non-Clifford
    content as rz rotations — a hard CDR requirement). Training circuits are
    generated internally by mitiq; the noiseless training labels come from
    ``qemsel.ideal.ideal_expectation`` (exact statevector, no shot noise).

    Fail-loud guards (science review 2026-07-21) — CDR must never return a
    classically-simulated value dressed up as a mitigated measurement:

    1. If the compiled circuit is FULLY CLIFFORD, mitiq 1.0.0 short-circuits
       (``mitiq/cdr/cdr.py``) and returns the ideal simulator value directly
       — zero error by construction, target never executed on the noisy
       backend. We raise :class:`MitigationError` instead.
    2. The training set is PRE-GENERATED here with the same seed/settings
       mitiq would use, and if the spread (``np.ptp``) of the training
       ideals is below ``CDR_MIN_TRAINING_IDEAL_SPREAD`` the linear
       regression is degenerate (curve_fit collapses to a constant equal to
       the classical value, ignoring the noisy measurement entirely) — we
       raise :class:`MitigationError` instead. ``run_experiment`` converts
       both into NaN + an errors.log line, so the dataset honestly records
       "CDR not applicable" rather than a fake ~1e-16 win.

    Custom regressors: ``CDR_FIT_FUNCTION`` / ``CDR_NUM_FIT_PARAMETERS``
    are passed through to ``execute_with_cdr`` when set (see the module
    docstring for the sklearn-regressor route).

    Executor invocations: ``1 + CDR_NUM_TRAINING_CIRCUITS`` (default
    ``scale_factors=(1,)``, i.e. plain CDR without noise scaling).
    """
    from mitiq.cdr import generate_training_circuits
    from mitiq.cdr.clifford_utils import is_clifford

    compiled = transpile(
        circuit,
        basis_gates=list(CDR_BASIS_GATES),
        optimization_level=0,
        seed_transpiler=seed,
    )
    if is_clifford(compiled):
        raise MitigationError(
            "cdr",
            "circuit is fully Clifford after compilation — mitiq would "
            "short-circuit and return the classical simulator value (zero "
            "error by construction, no mitigation performed); recording "
            "this as a CDR result would be an artifact",
        )
    # Pre-generate the training set exactly as execute_with_cdr will (same
    # circuit, count, fraction and random_state) and check that the ideal
    # values actually vary; a constant training target makes the regression
    # degenerate (classical simulation in disguise).
    training_circuits = generate_training_circuits(
        compiled,
        num_training_circuits=CDR_NUM_TRAINING_CIRCUITS,
        fraction_non_clifford=CDR_FRACTION_NON_CLIFFORD,
        random_state=seed,
    )
    bad_width = [
        tc.num_qubits for tc in training_circuits if tc.num_qubits != len(pauli)
    ]
    if bad_width:
        raise MitigationError(
            "cdr",
            f"training circuits have {bad_width[0]} qubits but the pauli has "
            f"{len(pauli)} — mitiq's qiskit->cirq round trip drops idle "
            "qubits; CDR cannot run on circuits with idle wires",
        )
    training_ideals = [
        _ideal.ideal_expectation(tc, pauli) for tc in training_circuits
    ]
    spread = float(np.ptp(training_ideals))
    if spread < CDR_MIN_TRAINING_IDEAL_SPREAD:
        raise MitigationError(
            "cdr",
            f"all {len(training_ideals)} near-Clifford training circuits "
            f"have the same ideal value ({training_ideals[0]:+.6f}, spread "
            f"{spread:.2e} < {CDR_MIN_TRAINING_IDEAL_SPREAD}); the CDR "
            "regression would collapse to that constant and ignore the "
            "noisy measurement — classical simulation in disguise",
        )
    extra_kwargs: dict = {}
    if CDR_FIT_FUNCTION is not None:
        extra_kwargs["fit_function"] = CDR_FIT_FUNCTION
        extra_kwargs["num_fit_parameters"] = CDR_NUM_FIT_PARAMETERS
    mitigated = execute_with_cdr(
        compiled,
        lambda circ: executor(circ, pauli),
        simulator=lambda circ: _ideal.ideal_expectation(circ, pauli),
        num_training_circuits=CDR_NUM_TRAINING_CIRCUITS,
        fraction_non_clifford=CDR_FRACTION_NON_CLIFFORD,
        random_state=seed,
        **extra_kwargs,
    )
    return float(np.real(mitigated))


def _apply_rem(
    circuit: QuantumCircuit,
    pauli: str,
    executor: Callable[[QuantumCircuit, str], float],
) -> float:
    """Readout-error mitigation by calibrated parity-damping inversion.

    Why not ``mitiq.rem.execute_with_rem``: that API requires an executor
    returning ``mitiq.MeasurementResult`` raw bitstrings, but the qemsel
    executor contract returns a single float expectation, so the measured
    distribution is not available here. Instead we invert the tensored
    readout channel at the expectation level for the one Pauli we measure —
    the calibration-circuit REM variant proven in ``spikes/spike_rem.py``,
    specialized to a single observable.

    Math: per-qubit readout error maps the ideal measured parity via
    ``<Z_S>_meas = prod_{i in S} a_i * <Z_S>_true + cross-terms``, with
    ``a_i = 1 - p0_i - p1_i`` and cross-terms carrying factors of
    ``c_i = p1_i - p0_i``. Two calibration circuits run through the SAME
    executor estimate the damping directly:

    * ``f0``: <Z on support> with all qubits in |0>  -> ``prod (a_i + c_i)``
    * ``f1``: <Z on support> with support in |1>     -> ``(-1)^k prod (a_i - c_i)``

    ``damping = (f0 + (-1)^k f1) / 2 = prod a_i + O(c_i^2)`` and the
    mitigated value is ``raw / damping``. Exact for symmetric readout errors
    (p0 == p1); first-order accurate in the asymmetry otherwise. Because
    calibration goes through the executor, it captures the noise (and
    layout/transpilation) the executor actually applies — and a noiseless
    executor yields damping == 1, making REM an exact identity.

    Executor invocations: ``1 + REM_NUM_CALIBRATION_CIRCUITS``.

    Raises:
        MitigationError: if the calibrated damping factor is below
            ``REM_MIN_DAMPING`` in magnitude (near-singular readout,
            inversion would amplify noise unboundedly).
    """
    raw_value = float(executor(circuit, pauli))
    support = [q for q, p in enumerate(pauli) if p != "I"]
    if not support:
        # Identity observable: readout error cannot affect it.
        return raw_value
    n = circuit.num_qubits
    # Calibrate the Z-basis measurement (the executor's basis rotation for
    # X/Y factors happens before readout, so Z-calibration is the right one).
    cal_pauli = "".join("Z" if q in support else "I" for q in range(n))
    ground = QuantumCircuit(n)  # |0...0>
    excited = QuantumCircuit(n)  # |1> on every support qubit
    for q in support:
        excited.x(q)
    f0 = float(executor(ground, cal_pauli))
    f1 = float(executor(excited, cal_pauli))
    sign = (-1.0) ** len(support)
    damping = 0.5 * (f0 + sign * f1)
    if abs(damping) < REM_MIN_DAMPING:
        raise MitigationError(
            "rem",
            f"calibrated readout damping {damping!r} is too close to zero "
            f"to invert (support={support})",
        )
    return raw_value / damping


# ---------------------------------------------------------------------------
# V2 private helpers (builder-mitigation / B1 replaces the
# NotImplementedError bodies; signatures + docstrings are the contract).
# ---------------------------------------------------------------------------


def _apply_zne_fr(
    circuit: QuantumCircuit,
    pauli: str,
    executor: Callable[[QuantumCircuit, str], float],
    backend_name: str,
    base_shots: int,
    seed: int,
) -> float:
    """Fixed-Richardson ZNE aligned to Scavino's analyzed variant (2605.08251).

    Differences from ``_apply_zne`` (all three are the alignment the Angle 3
    boundary overlay requires — do not "improve" them):

    1. FIXED coefficients: the estimate is ``sum_k c_k * E_k`` with
       ``c_k = richardson_coefficients(ZNE_FR_SCALE_FACTORS)`` computed a
       priori from the nodes — never a refit ``RichardsonFactory`` (mitiq's
       factory solves the same linear system, but going through our own
       fixed coefficients keeps the implementation transparently identical
       to the K_q variance model in ``qemsel.boundary``).
    2. SHOT ALLOCATION per ``ZNE_FR_SHOT_ALLOCATION``: 'equal_split' means
       each level k runs at ``base_shots // len(ZNE_FR_SCALE_FACTORS)``
       shots. The PASSED executor (bound to base_shots) is therefore NOT
       invoked; per-level executors are rebuilt via
       ``_backends.make_executor(backend_name, level_shots, seed)`` exactly
       like ``_apply_raw_plus`` does (module-attribute lookup so tests can
       monkeypatch; ``close()`` in a ``finally`` when exposed). Total cost
       must equal ``SHOT_MULTIPLIER_V2['zne_fr'] * base_shots``.
    3. DETERMINISTIC noise amplification per ``ZNE_FR_FOLD_METHOD``:
       'global' = ``mitiq.zne.scaling.fold_global`` on a defensive
       ``circuit.copy()`` — no per-call randomness (the analytic variance
       side assumes deterministic amplification).

    Raises:
        MitigationError: internal failures, wrapped with technique name
            'zne_fr' (apply_technique's outer wrapper also guarantees this).
    """
    from mitiq.zne.scaling import fold_global

    # (1) Fixed a-priori coefficients from the nodes — never refit.
    coeffs = richardson_coefficients(ZNE_FR_SCALE_FACTORS)
    n_levels = len(ZNE_FR_SCALE_FACTORS)

    # (2) Per-level shot budget. 'equal_split' spends ONE base budget total
    #     (B/n per level) so the zne_fr-vs-raw comparison is the equal-budget
    #     ΔMSE(ε, B) the boundary formula describes; 'full' spends one base
    #     budget PER level. SHOT_MULTIPLIER_V2['zne_fr'] is derived from the
    #     same switch, so the reported cost can never drift from what runs.
    if ZNE_FR_SHOT_ALLOCATION == "equal_split":
        level_shots = int(base_shots) // n_levels
    elif ZNE_FR_SHOT_ALLOCATION == "full":
        level_shots = int(base_shots)
    else:
        raise MitigationError(
            "zne_fr",
            f"unknown ZNE_FR_SHOT_ALLOCATION {ZNE_FR_SHOT_ALLOCATION!r}; "
            "expected 'equal_split' or 'full'",
        )
    if level_shots < 1:
        raise MitigationError(
            "zne_fr",
            f"base_shots {base_shots} is too small to split over {n_levels} "
            f"levels ({ZNE_FR_SHOT_ALLOCATION}) — level budget rounds to 0",
        )

    # (3) Deterministic global folding only (the analytic variance side
    #     assumes deterministic amplification; random folding would inject
    #     folding variance the boundary formula does not model).
    if ZNE_FR_FOLD_METHOD != "global":
        raise MitigationError(
            "zne_fr",
            f"unknown ZNE_FR_FOLD_METHOD {ZNE_FR_FOLD_METHOD!r}; expected "
            "'global'",
        )

    estimate = 0.0
    for coeff, scale in zip(coeffs, ZNE_FR_SCALE_FACTORS):
        # Defensive copy: fold_global never sees the caller's circuit.
        folded = fold_global(circuit.copy(), scale)
        # Rebuild a per-level executor at the split budget (the PASSED
        # executor is bound to base_shots and must not be invoked). Same
        # module-attribute lookup + close()-in-finally contract as raw_plus.
        level_executor = _backends.make_executor(backend_name, level_shots, seed)
        try:
            e_k = float(level_executor(folded, pauli))
        finally:
            close = getattr(level_executor, "close", None)
            if callable(close):
                close()
        estimate += coeff * e_k
    return float(estimate)


def _apply_cdr_sklearn(
    circuit: QuantumCircuit,
    pauli: str,
    executor: Callable[[QuantumCircuit, str], float],
    seed: int,
    regressor: str,
) -> float:
    """CDR with a sklearn regressor ('cdr_ridge' | 'cdr_rf' — Angle 2).

    Route (module docstring "Plugging in your own CDR regressor", route 2 —
    the ``generate_training_circuits`` bypass; sklearn regressors cannot go
    through mitiq's curve_fit-style ``fit_function``; a spike may still
    choose the fit_function route FOR RIDGE ONLY if it proves equivalent,
    but the bypass is the contracted default):

    1. Compile to ``CDR_BASIS_GATES`` (optimization_level=0,
       seed_transpiler=seed) — identical to ``_apply_cdr``.
    2. Apply the SAME three fail-loud guards as ``_apply_cdr``, with the
       same trigger conditions and MitigationError texts adapted to the
       technique name: fully-Clifford short-circuit, idle-wire width
       mismatch, training-ideal spread < CDR_MIN_TRAINING_IDEAL_SPREAD.
       Identical refusal conditions are LOAD-BEARING: Angle 2 compares the
       cdr variants on the same accepted-row set.
    3. Training set: ``mitiq.cdr.generate_training_circuits(compiled,
       num_training_circuits=CDR_SKLEARN_NUM_TRAINING_CIRCUITS,
       fraction_non_clifford=CDR_FRACTION_NON_CLIFFORD, random_state=seed)``
       — same settings as 'cdr' so the variants differ ONLY in the
       regressor.
    4. Run every training circuit through the noisy ``executor`` AND
       ``qemsel.ideal.ideal_expectation``; fit regressor noisy -> ideal:
       'cdr_ridge': ``sklearn.linear_model.RidgeCV(alphas=CDR_RIDGE_ALPHAS)``
       (deterministic efficient-LOO alpha selection — no extra noisy calls);
       'cdr_rf': ``sklearn.ensemble.RandomForestRegressor(
       n_estimators=CDR_RF_N_ESTIMATORS, max_depth=CDR_RF_MAX_DEPTH,
       random_state=seed)``. Single feature: the noisy expectation value.
    5. Execute the compiled TARGET once through ``executor`` and return the
       regressor's prediction for it (float; not clipped).

    Executor invocations: ``1 + CDR_SKLEARN_NUM_TRAINING_CIRCUITS`` — must
    equal ``SHOT_MULTIPLIER_V2[regressor]``.

    Raises:
        ValueError: ``regressor`` not in {'cdr_ridge', 'cdr_rf'}.
        MitigationError: guard refusals and internal failures.
    """
    from mitiq.cdr import generate_training_circuits
    from mitiq.cdr.clifford_utils import is_clifford
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import RidgeCV

    # Build the sklearn estimator up front (validates the regressor name
    # BEFORE any transpile / execution).
    if regressor == "cdr_ridge":
        # RidgeCV with a deterministic efficient-LOO alpha grid (cv=None) —
        # NOT a fixed Ridge(alpha=1.0): with ~10 training points and one
        # unstandardized feature, Sxx is O(0.01-1), so a fixed alpha=1.0
        # shrinks the slope by 43-99% and collapses predictions toward the
        # training-ideal mean (findings-applier 2026-07-23; the pre-fix
        # variant was 3.6-6.7x worse than plain cdr and worse than raw).
        model = RidgeCV(alphas=CDR_RIDGE_ALPHAS)
    elif regressor == "cdr_rf":
        model = RandomForestRegressor(
            n_estimators=CDR_RF_N_ESTIMATORS,
            max_depth=CDR_RF_MAX_DEPTH,
            random_state=seed,
        )
    else:
        raise ValueError(
            f"unknown sklearn CDR regressor {regressor!r}; expected "
            "'cdr_ridge' or 'cdr_rf'"
        )

    # (1) Same compile as _apply_cdr: all non-Clifford content as rz.
    compiled = transpile(
        circuit,
        basis_gates=list(CDR_BASIS_GATES),
        optimization_level=0,
        seed_transpiler=seed,
    )
    # (2) The SAME three fail-loud guards as _apply_cdr, with identical
    #     trigger conditions (LOAD-BEARING: Angle 2 compares the cdr variants
    #     on the same accepted-row set), technique name adapted.
    if is_clifford(compiled):
        raise MitigationError(
            regressor,
            "circuit is fully Clifford after compilation — mitiq would "
            "short-circuit and return the classical simulator value (zero "
            "error by construction, no mitigation performed); recording "
            "this as a CDR result would be an artifact",
        )
    # (3) Training set: identical settings to 'cdr' so the variants differ
    #     ONLY in the regressor (the Angle 2 control).
    training_circuits = generate_training_circuits(
        compiled,
        num_training_circuits=CDR_SKLEARN_NUM_TRAINING_CIRCUITS,
        fraction_non_clifford=CDR_FRACTION_NON_CLIFFORD,
        random_state=seed,
    )
    bad_width = [
        tc.num_qubits for tc in training_circuits if tc.num_qubits != len(pauli)
    ]
    if bad_width:
        raise MitigationError(
            regressor,
            f"training circuits have {bad_width[0]} qubits but the pauli has "
            f"{len(pauli)} — mitiq's qiskit->cirq round trip drops idle "
            "qubits; CDR cannot run on circuits with idle wires",
        )
    training_ideals = [
        _ideal.ideal_expectation(tc, pauli) for tc in training_circuits
    ]
    spread = float(np.ptp(training_ideals))
    if spread < CDR_MIN_TRAINING_IDEAL_SPREAD:
        raise MitigationError(
            regressor,
            f"all {len(training_ideals)} near-Clifford training circuits "
            f"have the same ideal value ({training_ideals[0]:+.6f}, spread "
            f"{spread:.2e} < {CDR_MIN_TRAINING_IDEAL_SPREAD}); the CDR "
            "regression would collapse to that constant and ignore the "
            "noisy measurement — classical simulation in disguise",
        )
    # (4) Route B (the generate_training_circuits bypass — sklearn regressors
    #     cannot go through mitiq's curve_fit-style fit_function). Single
    #     feature = the noisy expectation value. Noisy executions happen ONLY
    #     after every guard has passed, so a refusal spends 0 noisy calls
    #     (mirrors _apply_cdr).
    x_noisy = np.array([[float(executor(tc, pauli))] for tc in training_circuits])
    y_ideal = np.asarray(training_ideals, dtype=float)
    # (5) Execute the compiled TARGET once, then predict its mitigated value.
    target_noisy = float(executor(compiled, pauli))
    model.fit(x_noisy, y_ideal)
    return float(model.predict(np.array([[target_noisy]]))[0])
