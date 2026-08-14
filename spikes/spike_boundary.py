"""Spike: analytic finite-shot ZNE help-harm boundary (Scavino Alfaro, arXiv:2605.08251).

Implements the paper's closed-form leading-order MSE difference between the raw
(unmitigated) estimator and fixed-Richardson ZNE, and its zero-crossing boundary,
exactly as extracted from the full text (arxiv.org/html/2605.08251v1, verified
against ar5iv on 2026-07-22):

    Delta_MSE(eps, B) = MSE_raw - MSE_ZNE
                      = D_p * eps^(2p)  -  K_q * eps^q / B  +  R_p(eps, B)

    Delta_MSE > 0  <=>  ZNE helps.   (sign convention quoted from the paper)

Symbols (paper -> here):
    eps   : physical noise strength (per-location Pauli/depolarizing rate);
            scaled multiplicatively by lambda_j at ZNE level j.
    B     : TOTAL shot budget. Raw spends all B at eps; ZNE SPLITS the same B
            across levels, n_j = pi_j * B  (equal-budget comparison).
    p     : bias exponent; p = 1 for fixed Richardson with linear leading bias.
    D_p   : squared-bias improvement coefficient. For p = 1, D_1 = alpha^2 with
            alpha = d E[mu_hat_noisy]/d eps at eps = 0 (needs mu_0 -> sim-only /
            known-answer circuits).
    q     : variance exponent of the effective single-shot variance
            v(eps) = nu * eps^q + O(eps^(q+delta)).  Property of the OBSERVABLE
            + measurement protocol, not of the extrapolation rule.
                q = 1, nu = 2*kappa  : deterministic +/-1 observable with
                                       mu(eps) = 1 - kappa*eps (GHZ / mirror).
                q = 0, nu = v0       : variational energy, v0 = <H^2>_0 - <H>_0^2.
    K_q   : excess-variance penalty of ZNE over raw at equal total budget:
                K_{q,k} = nu * [ sum_j c_j^2 * lambda_j^q / pi_j  -  1 ]
            (the "-1" is raw's own variance nu*eps^q/B being credited back).
    c_j   : FIXED Richardson coefficients for nodes lambda_0 < ... < lambda_k:
            sum c_j = 1, sum c_j lambda_j^m = 0 (m = 1..k)  ->  Lagrange at 0.
            Two-point rule (1, 3):  c_0 = 3/2, c_1 = -1/2.  Paper's main setup:
            k = 1, lambdas = (1, 3), UNIFORM split pi = (1/2, 1/2).

Boundary (Theorem 1), the LOWER perturbative zero crossing:
    0 <= q < 2p : eps*(B) ~ (K_q / D_p)^(1/(2p-q)) * B^(-1/(2p-q))
                  ZNE HARMS below eps* (small eps / small B), HELPS above it.
    q == 2p    : no shrinking boundary; budget threshold B* = K_q / D_p
                  (B > B* -> helps for all small eps; B < B* -> harms).
    q  > 2p    : no leading-order lower boundary.

This spike is pure numpy — no qiskit/mitiq — and cross-checks the asymptotic
boundary against an EXACT finite-shot MSE zero crossing for the deterministic
observable model mu(eps) = (1 - eps)^ell, where the closed form has no
approximation beyond binomial shot statistics.

Run:
  & "E:\\quatum  computiiing\\qem-selector\\.venv\\Scripts\\python.exe" spikes\\spike_boundary.py
"""

from __future__ import annotations

import math

import numpy as np

# ----------------------------------------------------------------------------
# Fixed-Richardson machinery (paper Sec. 2-3)
# ----------------------------------------------------------------------------

LAMBDAS = (1.0, 3.0)          # paper's main two-point rule (k = 1)
PI = (0.5, 0.5)               # uniform shot allocation across levels
SHOT_BUDGETS = (256, 1024, 4096, 16384)
EPS_GRID = np.geomspace(1e-4, 0.5, 200)


def richardson_coeffs(lambdas: tuple[float, ...]) -> np.ndarray:
    """Fixed Richardson coefficients = Lagrange basis at 0 over the nodes.

    c_j = prod_{m != j} lambda_m / (lambda_m - lambda_j).
    Satisfies sum c_j = 1 and sum c_j lambda_j^m = 0 for m = 1..k.
    """
    lam = np.asarray(lambdas, dtype=float)
    coeffs = np.empty_like(lam)
    for j in range(lam.size):
        others = np.delete(lam, j)
        coeffs[j] = np.prod(others / (others - lam[j]))
    return coeffs


def variance_penalty_K(
    nu: float,
    q: float,
    lambdas: tuple[float, ...] = LAMBDAS,
    pi: tuple[float, ...] = PI,
) -> float:
    """K_{q,k} = nu * [ sum_j c_j^2 lambda_j^q / pi_j - 1 ]  (paper Sec. 3)."""
    c = richardson_coeffs(lambdas)
    lam = np.asarray(lambdas, dtype=float)
    w = np.asarray(pi, dtype=float)
    return float(nu * (np.sum(c**2 * lam**q / w) - 1.0))


def delta_mse(eps: np.ndarray, B: float, D_p: float, K_q: float, p: float, q: float) -> np.ndarray:
    """Leading-order Delta_MSE(eps, B) = D_p eps^(2p) - K_q eps^q / B.

    Positive => ZNE helps; negative => ZNE harms. Remainder R_p dropped.
    """
    return D_p * eps ** (2.0 * p) - K_q * eps**q / B


def boundary_eps(B: float, D_p: float, K_q: float, p: float, q: float) -> float:
    """Closed-form lower boundary eps*(B) for the subcritical regime q < 2p."""
    if not q < 2.0 * p:
        raise ValueError("closed-form eps*(B) only exists for q < 2p")
    return (K_q / (D_p * B)) ** (1.0 / (2.0 * p - q))


def budget_threshold(D_p: float, K_q: float) -> float:
    """Critical regime q = 2p: B* = K_q / D_p."""
    return K_q / D_p


# ----------------------------------------------------------------------------
# Two observable classes from the paper (Prop. 6)
# ----------------------------------------------------------------------------


def deterministic_class(ell: int, gamma: float = 1.0) -> dict:
    """GHZ / mirror-like: +/-1 observable, mu(eps) = (1 - gamma*eps)^ell.

    ell = number of noisy Pauli-active locations (Prop. 7's local Pauli
    contraction). Leading bias slope alpha = kappa = gamma*ell; single-shot
    variance v(eps) = 1 - mu^2 ~ 2*kappa*eps  ->  q = 1, nu = 2*kappa.
    In OUR codebase this is the mirror_circuit family (ideal exactly +1).
    """
    kappa = gamma * ell
    return {
        "name": f"deterministic (mirror/GHZ-like, ell={ell})",
        "p": 1.0,
        "q": 1.0,
        "alpha": kappa,
        "D_p": kappa**2,
        "nu": 2.0 * kappa,
        "ell": ell,
        "gamma": gamma,
    }


def variational_class(alpha: float, v0: float) -> dict:
    """Variational energy: ideal single-shot variance v0 > 0  ->  q = 0, nu = v0.

    In OUR codebase: hw_efficient_ansatz / layered_random Pauli expectations
    with |<P>| < 1 (nonzero ideal variance).
    """
    return {
        "name": f"variational (energy-like, alpha={alpha}, v0={v0})",
        "p": 1.0,
        "q": 0.0,
        "alpha": alpha,
        "D_p": alpha**2,
        "nu": v0,
    }


# ----------------------------------------------------------------------------
# Exact finite-shot MSE cross-check (deterministic model, no expansion)
# ----------------------------------------------------------------------------


def exact_delta_mse_deterministic(
    eps: np.ndarray, B: float, ell: int, gamma: float = 1.0
) -> np.ndarray:
    """Exact Delta_MSE for the +/-1 observable with mu(eps) = (1-gamma*eps)^ell.

    Raw:  bias^2 + (1 - mu(eps)^2)/B                 (all B shots at eps)
    ZNE:  (sum c_j mu(l_j eps) - 1)^2 + sum c_j^2 (1 - mu(l_j eps)^2)/(pi_j B)
    Only binomial shot statistics assumed — validates the local expansion.
    """
    c = richardson_coeffs(LAMBDAS)
    mu0 = 1.0

    def mu(e: np.ndarray) -> np.ndarray:
        return np.clip(1.0 - gamma * e, -1.0, 1.0) ** ell

    mse_raw = (mu(eps) - mu0) ** 2 + (1.0 - mu(eps) ** 2) / B
    bias_zne = sum(c[j] * mu(LAMBDAS[j] * eps) for j in range(len(LAMBDAS))) - mu0
    var_zne = sum(
        c[j] ** 2 * (1.0 - mu(LAMBDAS[j] * eps) ** 2) / (PI[j] * B)
        for j in range(len(LAMBDAS))
    )
    return mse_raw - (bias_zne**2 + var_zne)


def zero_crossing(eps: np.ndarray, dmse: np.ndarray) -> float | None:
    """First sign change - -> + on the grid (the LOWER boundary), interpolated."""
    sign = np.sign(dmse)
    for i in range(len(eps) - 1):
        if sign[i] < 0 and sign[i + 1] > 0:
            # log-linear interpolation
            x0, x1 = math.log(eps[i]), math.log(eps[i + 1])
            y0, y1 = dmse[i], dmse[i + 1]
            return math.exp(x0 - y0 * (x1 - x0) / (y1 - y0))
    return None


# ----------------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------------


def main() -> None:
    np.set_printoptions(precision=4, suppress=True)
    c = richardson_coeffs(LAMBDAS)
    print("=" * 78)
    print("Scavino arXiv:2605.08251 — finite-shot ZNE help-harm boundary (spike)")
    print("=" * 78)
    print(f"fixed-Richardson nodes lambda = {LAMBDAS}, coefficients c = {c}")
    assert np.isclose(c.sum(), 1.0), "sum c_j must be 1"
    assert np.isclose(np.dot(c, np.asarray(LAMBDAS)), 0.0), "sum c_j lambda_j must be 0"
    assert np.allclose(c, [1.5, -0.5]), "two-point rule must give (3/2, -1/2)"
    print(f"shot split pi = {PI}  (n_j = pi_j * B; raw uses all B)")
    c3 = richardson_coeffs((1.0, 3.0, 5.0))
    print(f"(generic check, k=2 nodes (1,3,5): c = {c3}, sum = {c3.sum():.4f})")
    print()

    # K_q for the paper's two observable classes under the (1,3) uniform rule:
    #   q=0: sum c^2 l^0 / pi = 4.5 + 0.5 = 5    -> K = 4 nu
    #   q=1: sum c^2 l^1 / pi = 4.5 + 1.5 = 6    -> K = 5 nu
    assert np.isclose(variance_penalty_K(1.0, 0.0), 4.0)
    assert np.isclose(variance_penalty_K(1.0, 1.0), 5.0)
    print("K_{q,k} penalty factors (nu=1): q=0 -> 4.000, q=1 -> 5.000   [verified]")
    print()

    classes = [
        deterministic_class(ell=20),   # ~20 noisy locations: small mirror circuit
        variational_class(alpha=2.0, v0=1.0),
    ]

    overall_ok = True
    for cls in classes:
        p, q, D_p, nu = cls["p"], cls["q"], cls["D_p"], cls["nu"]
        K_q = variance_penalty_K(nu, q)
        print("-" * 78)
        print(f"CLASS: {cls['name']}")
        print(f"  p={p:g}  q={q:g}  alpha={cls['alpha']:g}  D_p={D_p:g}  nu={nu:g}  K_q={K_q:g}")
        print(f"  regime: q < 2p (subcritical) -> eps*(B) = (K_q/(D_p B))^(1/{2*p - q:g})"
              f"  ~ B^(-1/{2*p - q:g})")
        print()
        print(f"  {'B':>7} | {'eps* (closed form)':>18} | {'dMSE(eps*/2)':>13} | "
              f"{'dMSE(2 eps*)':>13} | verdicts")
        prev_star = None
        for B in SHOT_BUDGETS:
            star = boundary_eps(B, D_p, K_q, p, q)
            below = float(delta_mse(np.array([star / 2.0]), B, D_p, K_q, p, q)[0])
            above = float(delta_mse(np.array([star * 2.0]), B, D_p, K_q, p, q)[0])
            ok_sign = below < 0 < above
            ok_mono = prev_star is None or star < prev_star
            overall_ok &= ok_sign and ok_mono
            prev_star = star
            print(f"  {B:>7} | {star:>18.6g} | {below:>13.4g} | {above:>13.4g} | "
                  f"harm-below={'OK' if ok_sign else 'FAIL'} "
                  f"shrinks-with-B={'OK' if ok_mono else 'FAIL'}")
        # Scaling-law check: eps*(4B)/eps*(B) must equal 4^(-1/(2p-q))
        r_meas = boundary_eps(4 * 256, D_p, K_q, p, q) / boundary_eps(256, D_p, K_q, p, q)
        r_theo = 4.0 ** (-1.0 / (2 * p - q))
        ok_slope = np.isclose(r_meas, r_theo)
        overall_ok &= ok_slope
        print(f"  slope check: eps*(4B)/eps*(B) = {r_meas:.4f} vs 4^(-1/(2p-q)) = "
              f"{r_theo:.4f}  [{'OK' if ok_slope else 'FAIL'}]")
        print()

        # Boundary curve table over the requested grid
        print(f"  Delta_MSE sign over eps in [1e-4, 0.5] (+ = ZNE helps, - = harms):")
        eps_marks = np.geomspace(1e-4, 0.5, 8)
        header = "  ".join(f"{e:>9.2e}" for e in eps_marks)
        print(f"  {'B':>7} | {header}")
        for B in SHOT_BUDGETS:
            vals = delta_mse(eps_marks, B, D_p, K_q, p, q)
            row = "  ".join(f"{'+' if v > 0 else '-':>9}" for v in vals)
            print(f"  {B:>7} | {row}")
        print()

        # Exact finite-shot cross-check for the deterministic class
        if q == 1.0:
            print("  EXACT finite-shot cross-check (binomial statistics, no expansion):")
            print(f"  {'B':>7} | {'eps* asymptotic':>15} | {'eps* exact':>12} | rel.diff")
            for B in SHOT_BUDGETS:
                star = boundary_eps(B, D_p, K_q, p, q)
                grid = np.geomspace(1e-7, 0.2, 4000)
                exact = zero_crossing(grid, exact_delta_mse_deterministic(grid, B, cls["ell"]))
                rel = abs(exact - star) / star if exact else float("nan")
                ok = exact is not None and rel < 0.15
                overall_ok &= ok
                print(f"  {B:>7} | {star:>15.6g} | {exact:>12.6g} | {rel:>7.2%} "
                      f"[{'OK' if ok else 'FAIL'}]")
            print("  (agreement tightens as B grows: eps* -> 0 = the perturbative regime)")
            print()

    # Critical-regime demo (q = 2p): budget threshold, no shrinking boundary
    print("-" * 78)
    print("CRITICAL REGIME demo (q = 2p = 2): budget threshold B* = K_q/D_p")
    nu_c, alpha_c = 1.0, 1.0
    K_c = variance_penalty_K(nu_c, 2.0)  # sum c^2 l^2 / pi = 4.5 + 4.5 = 9 -> K = 8
    Bstar = budget_threshold(alpha_c**2, K_c)
    print(f"  nu={nu_c} alpha={alpha_c} -> K_2={K_c:g}, B* = {Bstar:g} shots")
    for B in (int(Bstar // 2), int(2 * Bstar)):
        v = delta_mse(np.array([1e-3]), B, alpha_c**2, K_c, 1.0, 2.0)[0]
        verdict = "helps" if v > 0 else "harms"
        expect = "harms" if B < Bstar else "helps"
        ok = verdict == expect
        overall_ok &= ok
        print(f"  B={B}: dMSE(1e-3) = {v:.3e} -> ZNE {verdict} for ALL small eps "
              f"[{'OK' if ok else 'FAIL'}]")
    print()

    print("=" * 78)
    print(f"ALL SANITY CHECKS: {'PASS' if overall_ok else 'FAIL'}")
    print("=" * 78)
    if not overall_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
