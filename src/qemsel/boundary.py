"""Analytic ZNE help-harm boundary (Scavino arXiv:2605.08251) + selector overlay.

V2 module (INTERFACES.md section V2; builder-boundary / B3 implements every
``NotImplementedError`` body). This module is the theory side of the paper's
Angle 3 centerpiece: the closed-form finite-shot ZNE help-harm boundary

    delta_mse(eps, B) = D_p * eps**(2*p)  -  K_q * eps**q / B

where ``eps`` is the noise strength, ``B`` the total shot budget, the first
term is the squared-bias IMPROVEMENT from Richardson extrapolation and the
second the EXCESS sampling variance from the fixed Richardson coefficients +
shot splitting. SIGN CONVENTION (frozen): ``delta_mse = MSE_raw - MSE_zne_fr``,
so POSITIVE means ZNE helps. The help-harm boundary is the zero crossing.

The exact exponents/coefficients (p, q, D_p, K_q) come from the SPIKE that
re-derives Scavino's formula — this module freezes the API shape, not the
physics numbers. Three regime shapes are possible (paper abstract): shrinking
power law eps*(B) ~ B**(-1/(2p-q)), budget threshold B* = K_q/D_p, or no
lower boundary; :func:`boundary_eps` / :func:`boundary_shots` must handle all
three (returning None where no crossing exists).

Alignment guarantees (do not break):
* The Richardson coefficients come from
  ``qemsel.mitigation.richardson_coefficients(ZNE_FR_SCALE_FACTORS)`` —
  boundary.py IMPORTS mitigation (never the reverse), so the theory curve
  and the ``zne_fr`` implementation share one set of nodes/coefficients.
* K_q is computable a priori (Mohammadipour & Li arXiv:2502.20673); D_p
  needs the ideal value mu_0 — computable in SIMULATION only (via
  ``qemsel.ideal``). That asymmetry is the paper's non-circularity argument:
  the feature-only selector never sees mu_0.
* The eps axis of the overlay uses REALIZED backend error rates from
  ``qemsel.backends.get_backend_info`` (never nominal '@x' suffixes) — the
  cap-compression caveat (PROJECT_STATUS section 4.10) applies.

Cross-module imports allowed here (public functions only): mitigation
(richardson_coefficients, ZNE_FR_* constants), backends.get_backend_info,
ideal.ideal_expectation, features.extract_features, circuits.generate_suite,
plus joblib for the model bundle. NEVER import model/report/experiment —
report consumes this module's OUTPUT dict (JSON), not this module.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

from qiskit import QuantumCircuit

from qemsel import mitigation as _mitigation

#: get_backend_info key used as the noise-strength axis of the overlay by
#: default. 2q gate error is the dominant eps driver for our families;
#: grid_spec may override via 'eps_feature'.
DEFAULT_EPS_FEATURE: str = "avg_2q_error"

#: Predicted classes counted as "the selector chooses ZNE" in the overlay.
#: 'zne' kept so V1-trained bundles can be overlaid too; grid_spec may
#: override via 'zne_labels'.
DEFAULT_ZNE_LABELS: tuple[str, ...] = ("zne", "zne_fr")

#: Filename of the overlay figure (written into the out_dir passed to
#: overlay_selector_vs_theory; report.py embeds it by this relative name).
OVERLAY_PNG: str = "boundary_overlay.png"


@dataclass(frozen=True)
class BoundaryParams:
    """Coefficients of the closed-form boundary (spike supplies the values).

    Attributes:
        d_p: squared-bias improvement coefficient D_p (> 0 in the help
            regime's interior; needs mu_0 — sim-side only).
        k_q: excess-variance coefficient K_q (>= 0; a priori computable
            from the fixed Richardson coefficients + shot allocation).
        p: bias exponent parameter (delta_mse bias term is eps**(2*p)).
        q: variance exponent parameter (variance term is eps**q / B).
        scale_factors: the nodes these coefficients were derived for
            (default: mitigation.ZNE_FR_SCALE_FACTORS at construction time).
        shot_allocation: allocation the K_q side assumes (must match
            mitigation.ZNE_FR_SHOT_ALLOCATION for a truthful overlay).
        source: free-text provenance ('analytic', 'estimated:<circuit_id>').
    """

    d_p: float
    k_q: float
    p: int
    q: int
    scale_factors: tuple[float, ...] = field(
        default_factory=lambda: tuple(_mitigation.ZNE_FR_SCALE_FACTORS)
    )
    shot_allocation: str = field(
        default_factory=lambda: _mitigation.ZNE_FR_SHOT_ALLOCATION
    )
    source: str = "analytic"


def variance_k_q(
    scale_factors: tuple[float, ...] | None = None,
    shot_allocation: str | None = None,
) -> float:
    """A-priori excess-variance coefficient K_q of the fixed-Richardson ZNE.

    Computed from ``qemsel.mitigation.richardson_coefficients(scale_factors)``
    and the per-level shot allocation (Mohammadipour & Li 2502.20673 give
    the construction; the SPIKE fixes the exact expression). Defaults (None)
    pull the live ``ZNE_FR_SCALE_FACTORS`` / ``ZNE_FR_SHOT_ALLOCATION`` from
    qemsel.mitigation so theory and implementation cannot diverge.

    Returns:
        float K_q >= 0.

    Raises:
        ValueError: unknown shot_allocation / invalid nodes.
    """
    if scale_factors is None:
        scale_factors = tuple(_mitigation.ZNE_FR_SCALE_FACTORS)
    if shot_allocation is None:
        shot_allocation = _mitigation.ZNE_FR_SHOT_ALLOCATION
    # A-priori K_q reported at the generic variational pole (q=0, nu=1): the
    # purely geometric Richardson variance-amplification factor
    # ``sum_j c_j^2 / pi_j - 1`` (spike: notes/spike-boundary.md section 1,
    # e.g. 4.0 for the (1,3) two-point rule). The FULL family K_q (with the
    # observable's q and nu) is built by estimate_params via _variance_penalty;
    # this convenience is the nu=1, q=0 baseline the paper tabulates.
    return _variance_penalty(tuple(scale_factors), shot_allocation, q=0, nu=1.0)


def estimate_params(
    circuit: QuantumCircuit,
    pauli: str,
    backend_name: str,
    *,
    seed: int = 0,
) -> BoundaryParams:
    """Simulation-side estimate of the boundary coefficients for one unit.

    Uses the KNOWN ideal value mu_0 (``qemsel.ideal.ideal_expectation``) and
    noisy executions on ``backend_name`` (simulated only — MUST raise
    ValueError for any 'ibm_*' name; this function never spends QPU) to
    estimate D_p; K_q comes from :func:`variance_k_q`. The exact estimation
    recipe (how many probe scales/shots) is the SPIKE's deliverable; it must
    be deterministic given ``seed``. ``source`` records provenance.

    Returns:
        BoundaryParams with source='estimated:<circuit metadata>'.
    """
    if backend_name.startswith("ibm_"):
        raise ValueError(
            "estimate_params is simulation-only (needs the ideal value mu_0); "
            f"refusing real-hardware backend {backend_name!r}"
        )
    # Deferred imports: keeps `import qemsel.boundary` cheap and avoids any
    # import cycle (these are the public seams INTERFACES.md permits here).
    from qemsel import backends as _backends
    from qemsel import ideal as _ideal

    scale_factors = tuple(_mitigation.ZNE_FR_SCALE_FACTORS)
    shot_allocation = _mitigation.ZNE_FR_SHOT_ALLOCATION

    # mu_0: the known ideal value — the non-circularity ingredient (the
    # feature-only selector never sees this).
    mu0 = _ideal.ideal_expectation(circuit, pauli)

    # eps of this backend = its realized (scaled+capped) noise coordinate,
    # the SAME axis the overlay plots on (DEFAULT_EPS_FEATURE).
    info = _backends.get_backend_info(backend_name)
    eps = float(info[DEFAULT_EPS_FEATURE])

    # Leading bias slope alpha = dE[mu_noisy]/d eps at eps->0, estimated by a
    # secant from the known origin (0, mu_0) to the one noisy probe at this
    # backend. Deterministic given (backend_name, seed) because make_executor
    # is deterministic per (name, shots, seed).
    executor = _backends.make_executor(backend_name, _ESTIMATE_PROBE_SHOTS, seed)
    try:
        mu_noisy = float(executor(circuit, pauli))
    finally:
        _close_executor(executor)
    alpha = (mu_noisy - mu0) / eps if eps > 0.0 else 0.0
    d_p = float(alpha * alpha)  # D_1 = alpha^2 for fixed-Richardson (p = 1)
    p = 1

    if abs(abs(mu0) - 1.0) <= _DETERMINISTIC_MU0_TOL:
        # Deterministic +/-1 observable (mirror/GHZ, |mu_0| = 1): the ideal
        # single-shot variance vanishes, v(eps) = 1 - mu^2 ~ 2*kappa*eps, so
        # q = 1 and nu = 2*kappa with kappa = |alpha| (spike Prop. 6/7).
        q = 1
        nu = 2.0 * abs(alpha)
    else:
        # Variational-energy-like: nonzero ideal single-shot variance
        # v0 = <P^2>_0 - <P>_0^2 = 1 - mu_0^2 (P a Pauli, P^2 = I); q = 0.
        q = 0
        nu = max(0.0, 1.0 - mu0 * mu0)

    k_q = _variance_penalty(scale_factors, shot_allocation, q=q, nu=nu)
    source = (
        f"estimated:{getattr(circuit, 'name', 'circuit')}:"
        f"n{circuit.num_qubits}:d{circuit.depth()}:{backend_name}"
    )
    return BoundaryParams(
        d_p=d_p,
        k_q=k_q,
        p=p,
        q=q,
        scale_factors=scale_factors,
        shot_allocation=shot_allocation,
        source=source,
    )


def delta_mse(eps: float, shots: float, params: BoundaryParams) -> float:
    """The closed-form MSE difference, POSITIVE = zne_fr helps.

    Frozen formula (spike supplies only the coefficient values):
    ``params.d_p * eps**(2*params.p) - params.k_q * eps**params.q / shots``.

    Args:
        eps: noise strength (> 0; the overlay feeds realized backend error
            rates here).
        shots: total shot budget B (> 0).
        params: coefficients from :func:`variance_k_q` /
            :func:`estimate_params` / the spike's analytic values.

    Raises:
        ValueError: eps <= 0 or shots <= 0.
    """
    if not eps > 0.0:
        raise ValueError(f"eps must be > 0, got {eps!r}")
    if not shots > 0.0:
        raise ValueError(f"shots must be > 0, got {shots!r}")
    return (
        params.d_p * eps ** (2 * params.p)
        - params.k_q * eps ** params.q / shots
    )


def regime(
    eps: float, shots: float, params: BoundaryParams, *, tol: float = 0.0
) -> str:
    """'help' if ``delta_mse(eps, shots, params) > tol`` else 'harm'.

    ``tol`` (>= 0) lets callers demand a margin before calling 'help';
    exactly-zero delta_mse is 'harm' (mitigation must EARN its cost).

    Returns:
        'help' | 'harm' (never anything else).
    """
    if tol < 0.0:
        raise ValueError(f"tol must be >= 0, got {tol!r}")
    return "help" if delta_mse(eps, shots, params) > tol else "harm"


def boundary_eps(shots: float, params: BoundaryParams) -> float | None:
    """The eps* zero-crossing at budget ``shots``; None if no crossing.

    Solves delta_mse(eps, shots) == 0 for eps > 0. Must handle all three
    regime shapes (power law / budget threshold / no lower boundary) and
    return None when the crossing does not exist at this budget.
    """
    if not shots > 0.0:
        raise ValueError(f"shots must be > 0, got {shots!r}")
    two_p = 2 * params.p
    # No isolated eps crossing exists when:
    #  * d_p <= 0  (bias term never improves -> harm at every eps), or
    #  * k_q <= 0  (variance term never penalizes -> help at every eps), or
    #  * 2p == q   (critical: delta_mse = eps^(2p)*(d_p - k_q/shots) has no
    #               eps zero crossing -- it is a shots threshold; see
    #               boundary_shots).
    if params.d_p <= 0.0 or params.k_q <= 0.0:
        return None
    if two_p == params.q:
        return None
    base = params.k_q / (params.d_p * shots)  # > 0 here
    # eps* = (K_q/(D_p B))^(1/(2p-q)). Subcritical q < 2p: lower boundary
    # (harm below, help above). Supercritical q > 2p: 1/(2p-q) < 0, an upper
    # crossing (help below, harm above). Both are the unique positive root.
    return float(base ** (1.0 / (two_p - params.q)))


def boundary_shots(eps: float, params: BoundaryParams) -> float | None:
    """The budget B* zero-crossing at noise ``eps``; None if no crossing.

    Closed form when 2p != q: ``B* = (k_q / d_p) * eps**(q - 2*p)``; None
    when d_p <= 0 (no budget makes zne_fr help at this eps).
    """
    if not eps > 0.0:
        raise ValueError(f"eps must be > 0, got {eps!r}")
    if params.d_p <= 0.0:
        return None
    # B* = (K_q/D_p) * eps^(q-2p). At the critical case 2p == q this reduces
    # to the constant budget threshold B* = K_q/D_p (eps^0), exactly Scavino's
    # Theorem 1 critical regime -- the single formula covers all three shapes.
    val = (params.k_q / params.d_p) * eps ** (params.q - 2 * params.p)
    if val <= 0.0:
        return None  # k_q <= 0: zne_fr never adds net variance -> no threshold
    return float(val)


def overlay_selector_vs_theory(
    model_bundle: Path | str | dict,
    grid_spec: dict,
    out_dir: Path,
) -> dict:
    """The Angle 3 centerpiece: learned ZNE-refusal region vs analytic curve.

    Args:
        model_bundle: path to a ``model.joblib`` bundle (or the already
            loaded bundle dict) from ``qemsel.model.train_and_eval``. V2
            bundles carry 'feature_version'; a bundle with feature_version 2
            REQUIRES a shots axis in its features (log2_shots), which is
            exactly what lets the selector's decision vary along B. V1
            bundles are allowed (shots-blind selector: its region is
            constant in B — render it, the flatness IS the finding).
        grid_spec: dict with keys:
            'backends'   (list[str], REQUIRED): backend names spanning the
                eps axis, e.g. Fake devices at scales 0.25..2.0. eps of a
                grid point = get_backend_info(name)[eps_feature] — REALIZED,
                capped values, never the nominal '@x' suffix.
            'shots_list' (list[int], REQUIRED): the budget axis.
            'circuits'   (dict, REQUIRED): generate_suite config for the
                representative circuits evaluated at every grid point.
            'pauli'      (str | dict, optional, default 'auto'): resolved
                per circuit exactly like experiment does.
            'params'     (BoundaryParams | 'estimate', optional, default
                'estimate'): 'estimate' derives per-grid-point params via
                estimate_params on the representative circuits.
            'eps_feature' (str, optional, default DEFAULT_EPS_FEATURE).
            'zne_labels' (list[str], optional, default DEFAULT_ZNE_LABELS).
        out_dir: created if missing; the overlay figure is written here as
            ``OVERLAY_PNG`` (matplotlib 'Agg', figure closed after save).

    Behaviour contract:
    * Selector decision at a grid point (backend, shots): for every
      representative circuit, features via qemsel.features.extract_features
      (version = bundle's feature_version, default 1; base_shots = the grid
      point's shots) -> bundle model predict; the point counts as
      "selector chooses ZNE" when the fraction of representative circuits
      predicted in zne_labels is >= 0.5. Abstaining bundles: an 'abstain'
      outcome counts as NOT choosing ZNE.
    * Theory decision at the same point: regime(eps, shots, params).
    * Agreement = fraction of grid points where (selector chooses ZNE)
      == (theory says 'help'), as a percentage.
    * Deterministic: no unseeded randomness anywhere.
    * SIM-ONLY: raises ValueError if any grid backend starts with 'ibm_'.

    Returns:
        JSON-serializable dict with keys EXACTLY:
            'agreement_pct'       (float, 0..100)
            'iou_help'            (float 0..1: |selector AND theory-help| /
                                   |selector OR theory-help|; 0.0 when the
                                   union is empty)
            'n_points'            (int, len(backends) * len(shots_list))
            'selector_help_share' (float 0..1)
            'theory_help_share'   (float 0..1)
            'eps_feature'         (str)
            'zne_labels'          (list[str])
            'plot_path'           (str, str(out_dir / OVERLAY_PNG))
            'grid'                (list of per-point dicts: 'backend',
                                   'eps', 'shots', 'selector_zne' (bool),
                                   'zne_vote_share' (float), 'theory_regime'
                                   ('help'|'harm'), 'delta_mse' (float))

    Raises:
        ValueError: malformed grid_spec / bundle, 'ibm_*' backend, or a
            feature_version-2 bundle used without usable shots_list.
    """
    out_dir = Path(out_dir)

    # ---- validate grid_spec -----------------------------------------------
    if not isinstance(grid_spec, dict):
        raise ValueError(
            f"grid_spec must be a dict, got {type(grid_spec).__name__}"
        )
    for key in ("backends", "shots_list", "circuits"):
        if key not in grid_spec:
            raise ValueError(f"grid_spec missing required key {key!r}")
    backends = [str(b) for b in grid_spec["backends"]]
    if not backends:
        raise ValueError("grid_spec['backends'] must be a non-empty list")
    ibm = [b for b in backends if b.startswith("ibm_")]
    if ibm:
        raise ValueError(
            "boundary overlay is simulation-only; refusing ibm_* backends "
            f"{ibm}"
        )
    shots_list = [int(s) for s in grid_spec["shots_list"]]
    if not shots_list:
        raise ValueError("grid_spec['shots_list'] must be a non-empty list")
    if any(s <= 0 for s in shots_list):
        raise ValueError(
            f"grid_spec['shots_list'] entries must be > 0, got {shots_list}"
        )

    eps_feature = str(grid_spec.get("eps_feature", DEFAULT_EPS_FEATURE))
    zne_labels = [str(x) for x in grid_spec.get("zne_labels", DEFAULT_ZNE_LABELS)]
    pauli_cfg = grid_spec.get("pauli", "auto")
    params_spec = grid_spec.get("params", "estimate")

    # ---- load + inspect the model bundle ----------------------------------
    bundle = _load_model_bundle(model_bundle)
    model = bundle["model"]
    feature_names = [str(n) for n in bundle["feature_names"]]
    feature_version = int(bundle.get("feature_version", 1))
    abstain_threshold = bundle.get("abstain_threshold", None)
    classes = [str(c) for c in getattr(model, "classes_", bundle.get("classes", []))]
    if feature_version != 1 and not shots_list:
        raise ValueError(
            "a feature_version-2 bundle needs a usable shots_list (its "
            "features include log2_shots)"
        )

    # ---- representative circuits + realized eps per backend ---------------
    from qemsel import backends as _backends
    from qemsel import circuits as _circuits

    suite = _circuits.generate_suite(grid_spec["circuits"])
    if not suite:
        raise ValueError("grid_spec['circuits'] produced no circuits")

    eps_by_backend: dict[str, float] = {}
    for name in backends:
        info = _backends.get_backend_info(name)
        if eps_feature not in info:
            raise ValueError(
                f"eps_feature {eps_feature!r} not in get_backend_info keys "
                f"{sorted(info)}"
            )
        eps_by_backend[name] = float(info[eps_feature])

    # Theory params: one BoundaryParams per backend (a fixed object reused, or
    # estimated once per backend from the representative circuits).
    params_by_backend = _resolve_params_by_backend(
        params_spec, backends, suite, pauli_cfg
    )

    # ---- sweep the (backend/eps x shots) grid -----------------------------
    grid: list[dict] = []
    n_selector_zne = n_theory_help = n_agree = n_intersect = n_union = 0
    for name in backends:
        eps = eps_by_backend[name]
        params = params_by_backend[name]
        for shots in shots_list:
            vote_share = _selector_zne_vote(
                model, feature_names, classes, abstain_threshold,
                feature_version, zne_labels, suite, name, shots, pauli_cfg,
            )
            selector_zne = vote_share >= 0.5
            dmse = delta_mse(eps, float(shots), params)
            theory_help = dmse > 0.0  # regime(..., tol=0): help iff dmse > 0
            agree = selector_zne == theory_help
            n_selector_zne += int(selector_zne)
            n_theory_help += int(theory_help)
            n_agree += int(agree)
            n_intersect += int(selector_zne and theory_help)
            n_union += int(selector_zne or theory_help)
            grid.append(
                {
                    "backend": name,
                    "eps": eps,
                    "shots": int(shots),
                    "selector_zne": bool(selector_zne),
                    "zne_vote_share": float(vote_share),
                    "theory_regime": "help" if theory_help else "harm",
                    "delta_mse": float(dmse),
                }
            )

    n_points = len(backends) * len(shots_list)
    agreement_pct = 100.0 * n_agree / n_points if n_points else 0.0
    iou_help = (n_intersect / n_union) if n_union else 0.0
    selector_help_share = n_selector_zne / n_points if n_points else 0.0
    theory_help_share = n_theory_help / n_points if n_points else 0.0

    out_dir.mkdir(parents=True, exist_ok=True)
    plot_path = out_dir / OVERLAY_PNG
    _render_overlay(
        grid, backends, eps_by_backend, params_by_backend, eps_feature,
        plot_path,
    )

    return {
        "agreement_pct": float(agreement_pct),
        "iou_help": float(iou_help),
        "n_points": int(n_points),
        "selector_help_share": float(selector_help_share),
        "theory_help_share": float(theory_help_share),
        "eps_feature": eps_feature,
        "zne_labels": list(zne_labels),
        "plot_path": str(plot_path),
        "grid": grid,
    }


# ===========================================================================
# Private helpers (B3). None are part of the public V2 surface; they translate
# the boundary spike (notes/spike-boundary.md, spikes/spike_boundary.py) into
# the module conventions and drive the overlay. Module-level names they call
# (BoundaryParams, delta_mse, boundary_shots, estimate_params) resolve at call
# time, so definition order below does not matter.
# ===========================================================================

#: Probe shot count used by estimate_params for the finite-difference bias
#: slope. Fixed (not a public parameter) so estimate_params is deterministic
#: given only (circuit, pauli, backend_name, seed).
_ESTIMATE_PROBE_SHOTS: int = 8192

#: |mu_0| this close to 1 => the observable is the deterministic +/-1 class
#: (q=1, nu=2*kappa); otherwise variational (q=0, nu=1-mu_0**2).
_DETERMINISTIC_MU0_TOL: float = 1e-6


def _per_level_allocation(
    scale_factors: tuple[float, ...], shot_allocation: str
) -> list[float]:
    """Per-level shot fractions pi_j for the variance penalty.

    'equal_split' (Scavino's analyzed allocation): the base budget B is split
    uniformly, pi_j = 1/n (n_j = B/n shots per level). 'full': every level
    runs at the full base budget, pi_j = 1 (n_j = B). These are exactly the
    two allocations ``mitigation.ZNE_FR_SHOT_ALLOCATION`` admits.
    """
    n = len(scale_factors)
    if n < 2:
        raise ValueError(
            f"need >= 2 scale factors for Richardson extrapolation, got {n}"
        )
    if shot_allocation == "equal_split":
        return [1.0 / n] * n
    if shot_allocation == "full":
        return [1.0] * n
    raise ValueError(
        f"unknown shot_allocation {shot_allocation!r}; expected 'equal_split' "
        "or 'full'"
    )


def _variance_penalty(
    scale_factors: tuple[float, ...],
    shot_allocation: str,
    q: int,
    nu: float,
) -> float:
    """K_q = nu * (sum_j c_j^2 * s_j^q / pi_j - 1)  (spike variance_penalty_K).

    The bracket is the ZNE-vs-raw variance amplification at equal total budget
    (the -1 credits back raw's own single-shot variance); nu is the leading
    coefficient of the observable's single-shot variance v(eps) = nu*eps^q.
    c_j come from ``mitigation.richardson_coefficients`` (the shared source of
    truth) so the theory curve and the ``zne_fr`` implementation can never
    disagree on the coefficients. K_q >= 0 for the Richardson rules we use.
    """
    coeffs = _mitigation.richardson_coefficients(tuple(scale_factors))
    pis = _per_level_allocation(scale_factors, shot_allocation)
    if len(coeffs) != len(scale_factors):
        raise ValueError(
            f"richardson_coefficients returned {len(coeffs)} coefficients for "
            f"{len(scale_factors)} nodes"
        )
    bracket = sum(
        (c_j * c_j) * (float(s_j) ** q) / pi_j
        for c_j, s_j, pi_j in zip(coeffs, scale_factors, pis)
    )
    return float(nu * (bracket - 1.0))


def _close_executor(executor) -> None:
    """Best-effort ``executor.close()`` (the simulated executor has none)."""
    close = getattr(executor, "close", None)
    if callable(close):
        try:
            close()
        except Exception:  # noqa: BLE001 - cleanup must never raise
            pass


def _strip_feat_prefix(name: str) -> str:
    """Map a model feature column (``feat_depth``) to the extract_features key
    (``depth``); pass through names without the prefix unchanged."""
    return name[len("feat_"):] if name.startswith("feat_") else name


def _resolve_pauli(pauli_cfg, family: str, n_qubits: int) -> str:
    """Resolve a pauli spec for one circuit — identical rule to experiment.py
    (inlined because boundary must not import experiment): 'auto' => 'Z'*n, a
    single char is repeated to width, a per-family dict routes by family."""
    if isinstance(pauli_cfg, dict):
        spec = pauli_cfg.get(family, pauli_cfg.get("default", "auto"))
    else:
        spec = pauli_cfg
    if spec == "auto":
        return "Z" * n_qubits
    if len(spec) == 1:
        return spec * n_qubits
    return spec


def _load_model_bundle(model_bundle) -> dict:
    """Return the bundle dict from a path/str or an already-loaded dict."""
    if isinstance(model_bundle, dict):
        bundle = model_bundle
    else:
        import joblib

        path = Path(model_bundle)
        if not path.exists():
            raise ValueError(f"model bundle file not found: {path}")
        bundle = joblib.load(path)
    if not isinstance(bundle, dict):
        raise ValueError(
            "model bundle must be a dict (or a path to one), got "
            f"{type(bundle).__name__}"
        )
    for key in ("model", "feature_names"):
        if key not in bundle:
            raise ValueError(f"malformed model bundle: missing key {key!r}")
    model = bundle["model"]
    if not (hasattr(model, "predict_proba") or hasattr(model, "predict")):
        raise ValueError(
            "malformed model bundle: 'model' has neither predict_proba nor "
            "predict"
        )
    return bundle


def _predict_labels(model, X, classes, abstain_threshold) -> list[str]:
    """Predicted class label per row, with 'abstain' where max proba falls
    below the bundle's abstain_threshold (abstain counts as NOT choosing ZNE)."""
    if hasattr(model, "predict_proba"):
        import numpy as np

        proba = np.asarray(model.predict_proba(X), dtype=float)
        cls = [str(c) for c in getattr(model, "classes_", classes)]
        idx = proba.argmax(axis=1)
        maxp = proba.max(axis=1)
        labels: list[str] = []
        for i, k in enumerate(idx):
            if abstain_threshold is not None and maxp[i] < float(abstain_threshold):
                labels.append("abstain")
            else:
                labels.append(cls[int(k)])
        return labels
    return [str(p) for p in model.predict(X)]


def _selector_zne_vote(
    model,
    feature_names: list[str],
    classes: list[str],
    abstain_threshold,
    feature_version: int,
    zne_labels: list[str],
    suite,
    backend_name: str,
    shots: int,
    pauli_cfg,
) -> float:
    """Fraction of representative circuits the selector routes to a ZNE label
    at one grid point (backend, shots). Features are built in the bundle's
    ``feature_names`` order (feat_ prefix stripped) exactly like recommend.py."""
    import pandas as pd

    from qemsel.features import extract_features

    rows: list[list[float]] = []
    for circuit, _spec in suite:
        feats = extract_features(
            circuit, backend_name, version=feature_version, base_shots=shots
        )
        missing = [
            fn for fn in feature_names if _strip_feat_prefix(fn) not in feats
        ]
        if missing:
            raise ValueError(
                "feature mismatch between bundle and extract_features: bundle "
                f"expects {missing} but only {sorted(feats)} are available "
                "(bundle trained with a different features version?)"
            )
        rows.append([float(feats[_strip_feat_prefix(fn)]) for fn in feature_names])
    X = pd.DataFrame(rows, columns=feature_names)
    labels = _predict_labels(model, X, classes, abstain_threshold)
    zne_set = set(zne_labels)
    votes = sum(1 for lab in labels if lab in zne_set)
    return votes / len(labels)


def _resolve_params_by_backend(
    params_spec, backends: list[str], suite, pauli_cfg
) -> dict[str, BoundaryParams]:
    """Map each backend name -> the BoundaryParams the theory side uses.

    A fixed ``BoundaryParams`` is reused across all backends; ``'estimate'``
    derives one per backend by averaging estimate_params over the
    representative circuits (a convenience — the paper fits per family; see
    notes). Anything else is a ValueError.
    """
    if isinstance(params_spec, BoundaryParams):
        return {name: params_spec for name in backends}
    if params_spec == "estimate":
        return {
            name: _estimate_params_for_backend(name, suite, pauli_cfg)
            for name in backends
        }
    raise ValueError(
        "grid_spec['params'] must be a BoundaryParams or 'estimate', got "
        f"{params_spec!r}"
    )


def _estimate_params_for_backend(
    backend_name: str, suite, pauli_cfg
) -> BoundaryParams:
    """Representative BoundaryParams for one backend: mean D_p/K_q over the
    suite, modal q, p fixed at 1."""
    d_ps: list[float] = []
    k_qs: list[float] = []
    qs: list[int] = []
    for circuit, spec in suite:
        pauli = _resolve_pauli(pauli_cfg, spec.family, circuit.num_qubits)
        params = estimate_params(circuit, pauli, backend_name)
        d_ps.append(params.d_p)
        k_qs.append(params.k_q)
        qs.append(params.q)
    q = max(set(qs), key=qs.count)
    return BoundaryParams(
        d_p=sum(d_ps) / len(d_ps),
        k_q=sum(k_qs) / len(k_qs),
        p=1,
        q=q,
        scale_factors=tuple(_mitigation.ZNE_FR_SCALE_FACTORS),
        shot_allocation=_mitigation.ZNE_FR_SHOT_ALLOCATION,
        source=f"estimated-mean:{backend_name}:n{len(suite)}",
    )


def _render_overlay(
    grid: list[dict],
    backends: list[str],
    eps_by_backend: dict[str, float],
    params_by_backend: dict[str, BoundaryParams],
    eps_feature: str,
    plot_path: Path,
) -> None:
    """Write the Angle-3 overlay PNG, guarded.

    The figure is SECONDARY to the returned dict (the load-bearing output), so
    a rendering failure (e.g. a matplotlib/Agg allocation error under memory
    pressure) must never propagate out of overlay_selector_vs_theory: on
    failure a small placeholder is written to ``plot_path`` (so the file the
    contract promises exists) and a RuntimeWarning is emitted.
    """
    try:
        _render_overlay_impl(
            grid, backends, eps_by_backend, params_by_backend, eps_feature,
            plot_path,
        )
    except Exception as exc:  # noqa: BLE001 - the dict, not the PNG, is the output
        import warnings

        warnings.warn(
            f"boundary overlay figure could not be rendered ({exc!r}); wrote a "
            f"placeholder to {plot_path}",
            RuntimeWarning,
            stacklevel=2,
        )
        try:
            plot_path.write_bytes(
                b"boundary overlay figure unavailable (render error)\n"
            )
        except OSError:
            pass


def _render_overlay_impl(
    grid: list[dict],
    backends: list[str],
    eps_by_backend: dict[str, float],
    params_by_backend: dict[str, BoundaryParams],
    eps_feature: str,
    plot_path: Path,
) -> None:
    """Draw + save the overlay: grid points colored by theory regime and shaped
    by the selector's ZNE choice, with the analytic delta_mse=0 boundary
    curve(s) over the eps span."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    help_color = "#2a9d8f"
    harm_color = "#e76f51"

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    try:
        _draw_overlay_axes(
            ax, grid, eps_by_backend, params_by_backend, eps_feature,
            help_color, harm_color, Line2D,
        )
        fig.tight_layout()
        fig.savefig(plot_path, dpi=110)
    finally:
        plt.close(fig)


def _draw_overlay_axes(
    ax, grid, eps_by_backend, params_by_backend, eps_feature,
    help_color, harm_color, Line2D,
) -> None:
    for pt in grid:
        theory_help = pt["theory_regime"] == "help"
        color = help_color if theory_help else harm_color
        if pt["selector_zne"]:
            ax.scatter(
                pt["eps"], pt["shots"], marker="o", s=95, facecolors=color,
                edgecolors="black", linewidths=0.6, zorder=3,
            )
        else:
            ax.scatter(
                pt["eps"], pt["shots"], marker="X", s=95, facecolors="none",
                edgecolors=color, linewidths=1.8, zorder=3,
            )

    # Analytic boundary curve(s): B*(eps) = boundary_shots(eps) over the eps
    # span, one per distinct params object. Drawn defensively.
    eps_vals = [e for e in eps_by_backend.values() if e > 0.0]
    if eps_vals:
        lo, hi = min(eps_vals), max(eps_vals)
        if lo == hi:
            lo, hi = lo * 0.5, hi * 2.0
        n_line = 60
        eps_line = [
            lo * (hi / lo) ** (i / (n_line - 1)) for i in range(n_line)
        ]
        seen: list[int] = []
        for params in params_by_backend.values():
            if id(params) in seen:
                continue
            seen.append(id(params))
            xs: list[float] = []
            ys: list[float] = []
            for e in eps_line:
                try:
                    b = boundary_shots(e, params)
                except ValueError:
                    b = None
                if b is not None and b > 0.0 and math.isfinite(b):
                    xs.append(e)
                    ys.append(b)
            if len(xs) >= 2:
                ax.plot(
                    xs, ys, color="#264653", lw=1.8, ls="--", zorder=2,
                    label="analytic boundary  ΔMSE=0",
                )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(f"realized noise ε  (backend {eps_feature})")
    ax.set_ylabel("shot budget B")
    ax.set_title("Selector ZNE choice vs analytic help–harm boundary")
    ax.grid(True, which="both", ls=":", alpha=0.4)

    legend_handles = [
        Line2D([0], [0], marker="o", ls="none", markerfacecolor=help_color,
               markeredgecolor="black", markersize=10,
               label="selector: ZNE"),
        Line2D([0], [0], marker="X", ls="none", markerfacecolor="none",
               markeredgecolor="#555555", markersize=10,
               label="selector: not-ZNE"),
        Line2D([0], [0], marker="s", ls="none", markerfacecolor=help_color,
               markeredgecolor="none", markersize=10, label="theory: help"),
        Line2D([0], [0], marker="s", ls="none", markerfacecolor=harm_color,
               markeredgecolor="none", markersize=10, label="theory: harm"),
        Line2D([0], [0], color="#264653", lw=1.8, ls="--",
               label="analytic ΔMSE=0"),
    ]
    ax.legend(handles=legend_handles, loc="best", fontsize=8, framealpha=0.9)
