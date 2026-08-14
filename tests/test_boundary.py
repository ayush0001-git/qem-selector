"""Unit tests for qemsel.boundary (B3) — the analytic ZNE help-harm boundary.

Covers the closed-form spike math (delta_mse sign asymptotics, the three
regime shapes of the zero crossings, the a-priori variance coefficient) and
the Angle-3 overlay against a fake model bundle. The pure-math tests build
``BoundaryParams`` directly and are independent of every other builder.

``mitigation.richardson_coefficients`` is a B1 stub at the time of writing, so
tests that reach it (variance_k_q / estimate_params) monkeypatch it with the
exact Lagrange-at-zero rule from the spike (notes/spike-boundary.md); when B1
lands its implementation is the same deterministic rule, so the patch supplies
identical values and these tests keep passing.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from qemsel import boundary as B
from qemsel import mitigation as _mit
from qemsel.boundary import BoundaryParams
from qemsel.features import FEATURE_NAMES


# ---------------------------------------------------------------------------
# Shared helpers / fixtures
# ---------------------------------------------------------------------------


def _lagrange_at_zero(scale_factors):
    """Fixed Richardson coefficients = Lagrange basis at 0 (spike rule)."""
    lam = [float(s) for s in scale_factors]
    coeffs = []
    for j in range(len(lam)):
        prod = 1.0
        for m in range(len(lam)):
            if m != j:
                prod *= lam[m] / (lam[m] - lam[j])
        coeffs.append(prod)
    return tuple(coeffs)


@pytest.fixture()
def patched_richardson(monkeypatch):
    """Supply the exact Lagrange coefficients while B1's stub is unimplemented."""
    monkeypatch.setattr(
        "qemsel.mitigation.richardson_coefficients", _lagrange_at_zero
    )
    return _lagrange_at_zero


class _FakeModel:
    """Minimal sklearn-like classifier: broadcasts one fixed proba vector.

    Satisfies the bundle 'model' contract used by the overlay (predict_proba +
    classes_). ``predict`` returns the argmax class for every row.
    """

    def __init__(self, classes, proba_vector):
        self.classes_ = np.array(classes, dtype=object)
        self._p = np.asarray(proba_vector, dtype=float)

    def predict_proba(self, X):
        return np.tile(self._p, (len(X), 1))

    def predict(self, X):
        idx = int(self._p.argmax())
        return np.array([self.classes_[idx]] * len(X), dtype=object)


_CLASSES = ["raw", "zne", "cdr", "rem"]


def _bundle(proba_vector, *, abstain_threshold=None, feature_version=1):
    """A fake V1-style model bundle for the overlay."""
    bundle = {
        "model": _FakeModel(_CLASSES, proba_vector),
        "feature_names": [f"feat_{n}" for n in FEATURE_NAMES],
        "classes": _CLASSES,
        "feature_version": feature_version,
    }
    if abstain_threshold is not None:
        bundle["abstain_threshold"] = abstain_threshold
    return bundle


# proba favoring each interesting class (order raw, zne, cdr, rem)
_P_ZNE = [0.05, 0.85, 0.05, 0.05]   # argmax -> 'zne'
_P_RAW = [0.85, 0.05, 0.05, 0.05]   # argmax -> 'raw'
_P_ZNE_WEAK = [0.30, 0.40, 0.15, 0.15]  # argmax 'zne' but max < 0.5 threshold


def _tiny_circuits_cfg():
    return {
        "families": ["ghz_plus", "layered_random"],
        "n_qubits": [2],
        "depths": [4],
        "seeds": [0],
    }


# ---------------------------------------------------------------------------
# delta_mse — frozen formula + sign asymptotics
# ---------------------------------------------------------------------------


def test_delta_mse_matches_frozen_formula():
    params = BoundaryParams(d_p=2.0, k_q=3.0, p=1, q=0)
    eps, shots = 0.1, 1000.0
    expected = 2.0 * eps ** 2 - 3.0 * eps ** 0 / shots
    assert B.delta_mse(eps, shots, params) == pytest.approx(expected)


def test_delta_mse_q1_formula():
    params = BoundaryParams(d_p=5.0, k_q=7.0, p=1, q=1)
    eps, shots = 0.02, 4096.0
    expected = 5.0 * eps ** 2 - 7.0 * eps ** 1 / shots
    assert B.delta_mse(eps, shots, params) == pytest.approx(expected)


def test_delta_mse_rejects_nonpositive():
    params = BoundaryParams(d_p=1.0, k_q=1.0, p=1, q=0)
    with pytest.raises(ValueError):
        B.delta_mse(0.0, 1000.0, params)
    with pytest.raises(ValueError):
        B.delta_mse(-0.1, 1000.0, params)
    with pytest.raises(ValueError):
        B.delta_mse(0.1, 0.0, params)
    with pytest.raises(ValueError):
        B.delta_mse(0.1, -5.0, params)


def test_delta_mse_harm_at_tiny_eps_help_at_large_eps():
    # Subcritical q=1 < 2p=2, d_p,k_q > 0: ZNE HARMS below eps*, HELPS above
    # (spike section 4.3). At fixed budget, tiny eps -> variance term wins.
    params = BoundaryParams(d_p=100.0, k_q=5.0, p=1, q=1)
    shots = 1024.0
    assert B.delta_mse(1e-6, shots, params) < 0.0   # harm at tiny eps
    assert B.delta_mse(0.5, shots, params) > 0.0     # help at large eps


def test_delta_mse_help_grows_with_shots():
    # More shots shrinks the variance penalty -> pushes toward help.
    params = BoundaryParams(d_p=100.0, k_q=5.0, p=1, q=1)
    eps = 1e-3
    d_small = B.delta_mse(eps, 256.0, params)
    d_large = B.delta_mse(eps, 4096.0, params)
    assert d_large > d_small


# ---------------------------------------------------------------------------
# regime
# ---------------------------------------------------------------------------


def test_regime_help_and_harm():
    params = BoundaryParams(d_p=100.0, k_q=5.0, p=1, q=1)
    assert B.regime(0.5, 1024.0, params) == "help"
    assert B.regime(1e-6, 1024.0, params) == "harm"


def test_regime_exact_zero_is_harm():
    # delta_mse == 0 must read 'harm' (mitigation must EARN its cost).
    params = BoundaryParams(d_p=1.0, k_q=1.0, p=1, q=0)
    # choose eps, shots so that d_p*eps^2 == k_q/shots exactly:
    # eps^2 = 1/shots -> shots = 1/eps^2
    eps = 0.1
    shots = 1.0 / eps ** 2  # delta_mse == 0
    assert B.delta_mse(eps, shots, params) == pytest.approx(0.0, abs=1e-12)
    assert B.regime(eps, shots, params) == "harm"


def test_regime_tol_margin():
    params = BoundaryParams(d_p=1.0, k_q=0.0, p=1, q=0)
    # delta_mse = eps^2 = 0.01 at eps=0.1
    assert B.delta_mse(0.1, 1e9, params) == pytest.approx(0.01)
    assert B.regime(0.1, 1e9, params, tol=0.005) == "help"
    assert B.regime(0.1, 1e9, params, tol=0.02) == "harm"
    with pytest.raises(ValueError):
        B.regime(0.1, 1e9, params, tol=-0.1)


# ---------------------------------------------------------------------------
# boundary_eps / boundary_shots — the three regime shapes
# ---------------------------------------------------------------------------


def test_boundary_eps_is_a_zero_crossing():
    params = BoundaryParams(d_p=50.0, k_q=3.0, p=1, q=1)
    shots = 2048.0
    eps_star = B.boundary_eps(shots, params)
    assert eps_star is not None and eps_star > 0.0
    assert B.delta_mse(eps_star, shots, params) == pytest.approx(0.0, abs=1e-12)


def test_boundary_eps_subcritical_shrinks_with_shots_at_predicted_slope():
    # q=1 < 2p=2 -> eps*(B) ~ B^(-1/(2p-q)) = B^-1
    params = BoundaryParams(d_p=50.0, k_q=3.0, p=1, q=1)
    e1 = B.boundary_eps(256.0, params)
    e4 = B.boundary_eps(4.0 * 256.0, params)
    assert e4 < e1
    assert e4 / e1 == pytest.approx(4.0 ** (-1.0 / (2 * 1 - 1)))  # 0.25


def test_boundary_eps_q0_slope():
    # q=0 -> eps*(B) ~ B^(-1/2)
    params = BoundaryParams(d_p=4.0, k_q=1.0, p=1, q=0)
    e1 = B.boundary_eps(1000.0, params)
    e4 = B.boundary_eps(4000.0, params)
    assert e4 / e1 == pytest.approx(4.0 ** (-0.5))  # 0.5


def test_boundary_eps_none_when_dp_nonpositive():
    params = BoundaryParams(d_p=0.0, k_q=1.0, p=1, q=0)
    assert B.boundary_eps(1024.0, params) is None
    params_neg = BoundaryParams(d_p=-1.0, k_q=1.0, p=1, q=0)
    assert B.boundary_eps(1024.0, params_neg) is None


def test_boundary_eps_none_when_kq_nonpositive():
    params = BoundaryParams(d_p=1.0, k_q=0.0, p=1, q=0)
    assert B.boundary_eps(1024.0, params) is None


def test_boundary_eps_none_in_critical_regime():
    # q == 2p (critical): no eps crossing, only a budget threshold.
    params = BoundaryParams(d_p=1.0, k_q=8.0, p=1, q=2)
    assert B.boundary_eps(1024.0, params) is None


def test_boundary_eps_rejects_nonpositive_shots():
    params = BoundaryParams(d_p=1.0, k_q=1.0, p=1, q=0)
    with pytest.raises(ValueError):
        B.boundary_eps(0.0, params)


def test_boundary_shots_closed_form_and_roundtrip():
    params = BoundaryParams(d_p=50.0, k_q=3.0, p=1, q=1)
    eps = 0.01
    b_star = B.boundary_shots(eps, params)
    expected = (3.0 / 50.0) * eps ** (1 - 2)
    assert b_star == pytest.approx(expected)
    # delta_mse at (eps, B*) is a zero crossing
    assert B.delta_mse(eps, b_star, params) == pytest.approx(0.0, abs=1e-12)
    # roundtrip: boundary_eps at that budget recovers eps
    assert B.boundary_eps(b_star, params) == pytest.approx(eps)


def test_boundary_shots_critical_regime_is_constant_threshold():
    # q == 2p: B* = K_q/D_p independent of eps (Scavino critical regime).
    params = BoundaryParams(d_p=2.0, k_q=8.0, p=1, q=2)
    assert B.boundary_shots(0.01, params) == pytest.approx(4.0)
    assert B.boundary_shots(0.2, params) == pytest.approx(4.0)


def test_boundary_shots_none_when_dp_nonpositive():
    params = BoundaryParams(d_p=0.0, k_q=1.0, p=1, q=0)
    assert B.boundary_shots(0.01, params) is None


def test_boundary_shots_none_when_kq_nonpositive():
    params = BoundaryParams(d_p=1.0, k_q=0.0, p=1, q=0)
    assert B.boundary_shots(0.01, params) is None


def test_boundary_shots_rejects_nonpositive_eps():
    params = BoundaryParams(d_p=1.0, k_q=1.0, p=1, q=0)
    with pytest.raises(ValueError):
        B.boundary_shots(0.0, params)


def test_supercritical_has_upper_crossing():
    # q=3 > 2p=2: help BELOW the crossing, harm ABOVE it (an upper boundary).
    params = BoundaryParams(d_p=1.0, k_q=1.0, p=1, q=3)
    shots = 1000.0
    eps_star = B.boundary_eps(shots, params)
    assert eps_star is not None and eps_star > 0.0
    assert B.delta_mse(eps_star, shots, params) == pytest.approx(0.0, abs=1e-12)
    assert B.delta_mse(eps_star * 0.5, shots, params) > 0.0   # help below
    assert B.delta_mse(eps_star * 2.0, shots, params) < 0.0   # harm above


# ---------------------------------------------------------------------------
# variance_k_q — a-priori variance coefficient (needs richardson_coefficients)
# ---------------------------------------------------------------------------


def test_variance_k_q_two_point_matches_spike(patched_richardson):
    # (1,3) uniform split -> q=0 penalty factor 4.0 (spike hand calc).
    assert B.variance_k_q((1.0, 3.0), "equal_split") == pytest.approx(4.0)


def test_variance_k_q_two_point_full_allocation(patched_richardson):
    # 'full' (pi_j = 1): sum c^2 - 1 = 2.5 - 1 = 1.5 for (1,3).
    assert B.variance_k_q((1.0, 3.0), "full") == pytest.approx(1.5)


def test_variance_k_q_default_matches_manual(patched_richardson):
    # Robust to B1's node/allocation choice: recompute the q=0 penalty from the
    # LIVE ZNE_FR constants (the paper's (1,3) equal-split rule gives 4.0).
    sf = tuple(_mit.ZNE_FR_SCALE_FACTORS)
    alloc = _mit.ZNE_FR_SHOT_ALLOCATION
    c = _lagrange_at_zero(sf)
    pi = (1.0 / len(sf)) if alloc == "equal_split" else 1.0
    expected = sum(cj * cj / pi for cj in c) - 1.0
    assert B.variance_k_q() == pytest.approx(expected)


def test_variance_k_q_nonnegative(patched_richardson):
    for sf in [(1.0, 3.0), (1.0, 2.0, 3.0), (1.0, 2.0, 3.0, 4.0)]:
        assert B.variance_k_q(sf, "equal_split") >= 0.0


def test_variance_k_q_unknown_allocation_raises(patched_richardson):
    with pytest.raises(ValueError):
        B.variance_k_q((1.0, 3.0), "nonsense")


# ---------------------------------------------------------------------------
# estimate_params
# ---------------------------------------------------------------------------


def test_estimate_params_refuses_ibm():
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    with pytest.raises(ValueError, match="simulation-only"):
        B.estimate_params(qc, "ZZ", "ibm_brisbane")


@pytest.mark.slow
def test_estimate_params_deterministic_class(patched_richardson, tiny_identity_circuit):
    # 3q identity-equivalent, ideal <ZZZ> = +1 exactly -> deterministic class.
    params = B.estimate_params(tiny_identity_circuit, "ZZZ", "FakeManilaV2", seed=0)
    assert isinstance(params, BoundaryParams)
    assert params.p == 1
    assert params.q == 1  # |mu_0| == 1 -> deterministic
    assert params.d_p >= 0.0
    assert params.k_q >= 0.0
    assert params.source.startswith("estimated:")
    assert params.scale_factors == tuple(_mit.ZNE_FR_SCALE_FACTORS)
    # determinism: same seed -> identical estimate
    again = B.estimate_params(tiny_identity_circuit, "ZZZ", "FakeManilaV2", seed=0)
    assert again.d_p == params.d_p
    assert again.k_q == params.k_q


@pytest.mark.slow
def test_estimate_params_variational_class(patched_richardson, tiny_circuit):
    # Bell state, <ZI> = 0 exactly -> |mu_0| < 1 -> variational class q=0.
    params = B.estimate_params(tiny_circuit, "ZI", "FakeManilaV2", seed=0)
    assert params.q == 0
    assert params.p == 1
    assert params.d_p >= 0.0
    assert params.k_q >= 0.0


# ---------------------------------------------------------------------------
# overlay_selector_vs_theory
# ---------------------------------------------------------------------------

_OVERLAY_KEYS = {
    "agreement_pct",
    "iou_help",
    "n_points",
    "selector_help_share",
    "theory_help_share",
    "eps_feature",
    "zne_labels",
    "plot_path",
    "grid",
}
_GRID_KEYS = {
    "backend",
    "eps",
    "shots",
    "selector_zne",
    "zne_vote_share",
    "theory_regime",
    "delta_mse",
}


def test_overlay_smoke_structure_and_png(out_dir):
    bundle = _bundle(_P_ZNE)
    grid_spec = {
        "backends": ["FakeManilaV2@x0.5", "FakeManilaV2", "FakeManilaV2@x2.0"],
        "shots_list": [256, 4096],
        "circuits": _tiny_circuits_cfg(),
        # a params with a real crossing so the analytic curve is drawn too
        "params": BoundaryParams(d_p=1.0, k_q=2.0, p=1, q=0),
    }
    result = B.overlay_selector_vs_theory(bundle, grid_spec, out_dir)

    assert set(result) == _OVERLAY_KEYS
    assert result["n_points"] == 3 * 2
    assert 0.0 <= result["agreement_pct"] <= 100.0
    assert 0.0 <= result["iou_help"] <= 1.0
    assert 0.0 <= result["selector_help_share"] <= 1.0
    assert 0.0 <= result["theory_help_share"] <= 1.0
    assert result["eps_feature"] == "avg_2q_error"
    assert result["zne_labels"] == ["zne", "zne_fr"]
    assert len(result["grid"]) == 6
    for pt in result["grid"]:
        assert set(pt) == _GRID_KEYS
        assert pt["theory_regime"] in ("help", "harm")
        assert isinstance(pt["selector_zne"], bool)
    # PNG written to out_dir under the canonical name, and is non-empty.
    from pathlib import Path

    assert result["plot_path"] == str(Path(out_dir) / B.OVERLAY_PNG)
    assert (Path(out_dir) / B.OVERLAY_PNG).exists()
    assert (Path(out_dir) / B.OVERLAY_PNG).stat().st_size > 0

    # JSON-serializable
    import json

    json.dumps(result)


def test_overlay_perfect_agreement_all_help_all_zne(out_dir):
    # theory: help everywhere (d_p huge, k_q=0); selector: always zne.
    bundle = _bundle(_P_ZNE)
    grid_spec = {
        "backends": ["FakeManilaV2", "FakeManilaV2@x2.0"],
        "shots_list": [256, 4096],
        "circuits": _tiny_circuits_cfg(),
        "params": BoundaryParams(d_p=1e6, k_q=0.0, p=1, q=0),
    }
    result = B.overlay_selector_vs_theory(bundle, grid_spec, out_dir)
    assert result["agreement_pct"] == pytest.approx(100.0)
    assert result["theory_help_share"] == pytest.approx(1.0)
    assert result["selector_help_share"] == pytest.approx(1.0)
    assert result["iou_help"] == pytest.approx(1.0)
    assert all(pt["theory_regime"] == "help" for pt in result["grid"])
    assert all(pt["selector_zne"] for pt in result["grid"])


def test_overlay_perfect_agreement_all_harm_all_raw(out_dir):
    # theory: harm everywhere (d_p=0, k_q>0); selector: always raw.
    bundle = _bundle(_P_RAW)
    grid_spec = {
        "backends": ["FakeManilaV2", "FakeManilaV2@x2.0"],
        "shots_list": [256, 4096],
        "circuits": _tiny_circuits_cfg(),
        "params": BoundaryParams(d_p=0.0, k_q=1.0, p=1, q=0),
    }
    result = B.overlay_selector_vs_theory(bundle, grid_spec, out_dir)
    assert result["agreement_pct"] == pytest.approx(100.0)
    assert result["theory_help_share"] == pytest.approx(0.0)
    assert result["selector_help_share"] == pytest.approx(0.0)
    # empty union -> iou defined as 0.0
    assert result["iou_help"] == pytest.approx(0.0)


def test_overlay_total_disagreement(out_dir):
    # theory: help everywhere; selector: always raw -> agreement 0.
    bundle = _bundle(_P_RAW)
    grid_spec = {
        "backends": ["FakeManilaV2", "FakeManilaV2@x2.0"],
        "shots_list": [256, 4096],
        "circuits": _tiny_circuits_cfg(),
        "params": BoundaryParams(d_p=1e6, k_q=0.0, p=1, q=0),
    }
    result = B.overlay_selector_vs_theory(bundle, grid_spec, out_dir)
    assert result["agreement_pct"] == pytest.approx(0.0)
    assert result["theory_help_share"] == pytest.approx(1.0)
    assert result["selector_help_share"] == pytest.approx(0.0)
    assert result["iou_help"] == pytest.approx(0.0)


def test_overlay_abstain_counts_as_not_zne(out_dir):
    # max proba 0.85 < threshold 0.99 -> every prediction abstains -> not-zne.
    bundle = _bundle(_P_ZNE, abstain_threshold=0.99)
    grid_spec = {
        "backends": ["FakeManilaV2", "FakeManilaV2@x2.0"],
        "shots_list": [256, 4096],
        "circuits": _tiny_circuits_cfg(),
        "params": BoundaryParams(d_p=1e6, k_q=0.0, p=1, q=0),  # help everywhere
    }
    result = B.overlay_selector_vs_theory(bundle, grid_spec, out_dir)
    assert result["selector_help_share"] == pytest.approx(0.0)
    assert all(not pt["selector_zne"] for pt in result["grid"])
    # theory says help everywhere but selector abstains -> 0 agreement
    assert result["agreement_pct"] == pytest.approx(0.0)


def test_overlay_vote_share_half_threshold(out_dir):
    # 2 circuits: engineer a per-circuit split is hard with a fixed proba, so
    # here confirm a unanimous zne vote gives share 1.0 and selector_zne True.
    bundle = _bundle(_P_ZNE)
    grid_spec = {
        "backends": ["FakeManilaV2"],
        "shots_list": [1024],
        "circuits": _tiny_circuits_cfg(),  # 2 circuits
        "params": BoundaryParams(d_p=1.0, k_q=1.0, p=1, q=0),
    }
    result = B.overlay_selector_vs_theory(bundle, grid_spec, out_dir)
    assert result["n_points"] == 1
    pt = result["grid"][0]
    assert pt["zne_vote_share"] == pytest.approx(1.0)
    assert pt["selector_zne"] is True


def test_overlay_refuses_ibm_backend(out_dir):
    bundle = _bundle(_P_ZNE)
    grid_spec = {
        "backends": ["FakeManilaV2", "ibm_brisbane"],
        "shots_list": [1024],
        "circuits": _tiny_circuits_cfg(),
        "params": BoundaryParams(d_p=1.0, k_q=1.0, p=1, q=0),
    }
    with pytest.raises(ValueError, match="ibm_"):
        B.overlay_selector_vs_theory(bundle, grid_spec, out_dir)


def test_overlay_missing_keys_raise(out_dir):
    bundle = _bundle(_P_ZNE)
    with pytest.raises(ValueError, match="backends"):
        B.overlay_selector_vs_theory(bundle, {"shots_list": [1], "circuits": {}}, out_dir)
    with pytest.raises(ValueError, match="shots_list"):
        B.overlay_selector_vs_theory(
            bundle, {"backends": ["FakeManilaV2"], "circuits": {}}, out_dir
        )
    with pytest.raises(ValueError, match="circuits"):
        B.overlay_selector_vs_theory(
            bundle, {"backends": ["FakeManilaV2"], "shots_list": [1]}, out_dir
        )


def test_overlay_bad_params_spec_raises(out_dir):
    bundle = _bundle(_P_ZNE)
    grid_spec = {
        "backends": ["FakeManilaV2"],
        "shots_list": [1024],
        "circuits": _tiny_circuits_cfg(),
        "params": "not-a-valid-spec",
    }
    with pytest.raises(ValueError, match="params"):
        B.overlay_selector_vs_theory(bundle, grid_spec, out_dir)


def test_overlay_bad_eps_feature_raises(out_dir):
    bundle = _bundle(_P_ZNE)
    grid_spec = {
        "backends": ["FakeManilaV2"],
        "shots_list": [1024],
        "circuits": _tiny_circuits_cfg(),
        "params": BoundaryParams(d_p=1.0, k_q=1.0, p=1, q=0),
        "eps_feature": "no_such_feature",
    }
    with pytest.raises(ValueError, match="eps_feature"):
        B.overlay_selector_vs_theory(bundle, grid_spec, out_dir)


def test_overlay_estimate_mode_wires_through(out_dir, monkeypatch):
    # 'estimate' path: patch estimate_params so no noisy sim runs (fast), then
    # confirm the estimate branch is exercised and produces a valid overlay.
    fixed = BoundaryParams(d_p=10.0, k_q=1.0, p=1, q=0)

    def _fake_estimate(circuit, pauli, backend_name, *, seed=0):
        return fixed

    monkeypatch.setattr(B, "estimate_params", _fake_estimate)
    bundle = _bundle(_P_ZNE)
    grid_spec = {
        "backends": ["FakeManilaV2", "FakeManilaV2@x2.0"],
        "shots_list": [256, 4096],
        "circuits": _tiny_circuits_cfg(),
        # params omitted -> defaults to 'estimate'
    }
    result = B.overlay_selector_vs_theory(bundle, grid_spec, out_dir)
    assert result["n_points"] == 4
    assert set(result) == _OVERLAY_KEYS


def test_overlay_accepts_custom_zne_labels(out_dir):
    # If 'zne' is not in zne_labels, an always-'zne' selector counts as not-ZNE.
    bundle = _bundle(_P_ZNE)
    grid_spec = {
        "backends": ["FakeManilaV2"],
        "shots_list": [1024],
        "circuits": _tiny_circuits_cfg(),
        "params": BoundaryParams(d_p=1.0, k_q=1.0, p=1, q=0),
        "zne_labels": ["zne_fr"],  # only the fixed-Richardson label
    }
    result = B.overlay_selector_vs_theory(bundle, grid_spec, out_dir)
    assert result["zne_labels"] == ["zne_fr"]
    assert result["selector_help_share"] == pytest.approx(0.0)
