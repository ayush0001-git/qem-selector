# Spike: Scavino's finite-shot ZNE help-harm boundary (arXiv:2605.08251)

Written 2026-07-22 by the boundary-spike agent. Sources: arXiv abs page,
full text at arxiv.org/html/2605.08251v1, cross-verified against
ar5iv.labs.arxiv.org/html/2605.08251 (two independent fetches, all seven
key claims below confirmed with quoted passages). Runnable implementation:
`spikes/spike_boundary.py` — **runs green, all sanity checks PASS** with the
project venv python.

Paper: **"The finite-shot help-harm boundary of zero-noise extrapolation"**,
Vicenzo Scavino Alfaro, arXiv:2605.08251 (May 2026).

---

## 1. The exact closed-form boundary (verbatim structure)

Theorem 1 / Eq. (3):

```
Delta_MSE(eps, B) = MSE_raw(eps, B) - MSE_ZNE(eps, B)
                  = D_p * eps^(2p)  -  K_q * eps^q / B  +  R_p(eps, B)
```

- **Sign convention (quoted): "Delta_MSE > 0 <=> ZNE helps."** The judge's
  summary in RESEARCH_ANGLES.md had the same formula; the sign convention is
  now pinned: positive = help, and the boundary is the **lower** zero crossing
  — **ZNE harms BELOW eps\* (small eps), helps ABOVE it** ("ZNE harms below
  the lower local boundary and helps above it within the perturbative
  regime").
- `R_p = O(eps^(2p+delta_b)) + O(eps^(q+delta_v)/B)` is a controlled
  higher-order remainder (dropped in the spike; the exact-MSE cross-check
  below quantifies what dropping it costs).

### Symbols, precisely

| Symbol | Paper meaning | Value in the paper's main setup |
|---|---|---|
| `eps` | physical noise strength per noisy location (local Pauli contraction / depolarizing rate), scaled multiplicatively to `lambda_j * eps` at ZNE level j | swept axis |
| `B` | **TOTAL** shot budget. Raw spends all `B` at `eps`; ZNE **splits the same B**: `n_j = pi_j * B`, `sum pi_j = 1` — an equal-total-budget comparison (quoted: "The unmitigated estimator, using all B shots at noise eps") | swept axis |
| `p` | bias exponent of the leading uncancelled bias | **p = 1** for fixed Richardson with linear leading bias |
| `D_p` | squared-bias-improvement coefficient; for p=1, `D_1 = alpha^2`, `alpha = d E[mu_hat_noisy]/d eps` at eps=0 | needs the ideal value mu_0 → sim / known-answer only (the Angle 3 "feature, not bug") |
| `q` | variance exponent of the effective single-shot variance `v(eps) = nu*eps^q + ...` — a property of the **observable + measurement protocol**, NOT the extrapolation rule | q=1 deterministic, q=0 variational (Prop. 6) |
| `nu` | leading coefficient of `v(eps)` | `2*kappa` (deterministic), `v0 = <H^2>_0 - <H>_0^2` (variational) |
| `K_q` | excess-variance penalty: `K_{q,k} = nu * [ sum_j c_j^2 * lambda_j^q / pi_j - 1 ]` (the −1 credits back raw's own variance) | q=0 → 4·nu; q=1 → 5·nu under the (1,3) uniform rule |
| `c_j` | **fixed** Richardson coefficients: `sum c_j = 1`, `sum c_j lambda_j^m = 0` for m=1..k (= Lagrange basis at 0) | (3/2, −1/2) for nodes (1,3) |
| `lambda_j` | noise-scale nodes `1 = lambda_0 < ... < lambda_k` | **(1, 3)**, k=1 |
| `pi_j` | per-level shot fractions | **uniform (1/2, 1/2)** in main results |

### The three boundary regimes (Theorem 1, names from the abstract)

1. **Subcritical `0 <= q < 2p` — shrinking power law:**
   `eps*(B) ~ C_{p,q} * B^(-1/(2p-q))`, `C_{p,q} = (K_q/D_p)^(1/(2p-q))`.
   Equivalently `eps*(B) = (K_q/(D_p*B))^(1/(2p-q))`.
   - q=1 (deterministic/mirror): `eps* ∝ B^(-1)`
   - q=0 (variational energy): `eps* ∝ B^(-1/2)`
   (Paper's Aer fits: GHZ slopes cluster near −1, QAOA near −0.5. ✓)
2. **Critical `q = 2p` — budget threshold:** no shrinking boundary;
   `B* = K_q/D_p`. `B > B*` → ZNE helps for all small eps; `B < B*` → harms.
3. **Supercritical `q > 2p`:** no leading-order lower boundary.

### The two observable classes (Prop. 6) — what q to use

- **Deterministic ±1 observable** (GHZ `X^⊗n`, our `mirror_circuit` with
  ideal exactly +1): `mu(eps) = 1 - kappa*eps + O(eps^2)` ⇒
  `v(eps) = 1 - mu^2 = 2*kappa*eps + ...` ⇒ **q = 1, nu = 2*kappa**.
  Under the local Pauli contraction model (Prop. 7)
  `mu(lambda_j*eps) = (1 - gamma*lambda_j*eps)^ell_n` with `ell_n` = number of
  noisy Pauli-active locations, so `alpha = kappa = gamma*ell_n`.
- **Variational energy** (nonzero ideal variance — our
  `hw_efficient_ansatz`/`layered_random` Pauli expectations with |ideal|<1):
  `v(0) = v0 > 0` ⇒ **q = 0, nu = v0**.

---

## 2. What "fixed-Richardson ZNE" means EXACTLY (for `zne_fr`)

The variant the boundary is derived for — and what our `zne_fr` must
implement to make the overlay apples-to-apples:

1. **Scale factors (1, 3)** — two-point, first-order Richardson (k=1).
   (Robustness checks use k=2 with (1,3,5); the headline theory/sims are k=1.)
2. **FIXED coefficients c = (3/2, −1/2)** — Lagrange at 0 over the nodes,
   general two-point rule `c_0 = a/(a-1), c_1 = -1/(a-1)` for nodes (1,a).
   NOT a fitted/least-squares factory: the coefficients are constants fixed
   in advance by the nodes, never re-fit from data.
3. **Shot splitting: total budget B split across the two levels, uniform
   `pi = (1/2, 1/2)`** — i.e. B/2 shots at scale 1 and B/2 at scale 3.
   The raw comparator uses ALL B shots at scale 1. (Paper also derives the
   optimal allocation `pi_j ∝ |c_j|*sqrt(v(lambda_j*eps))` but uses uniform
   for the main results.)
4. **Noise amplification in the paper's Aer sims scales the depolarizing
   parameter directly** (multiplies eps), not gate folding. Primary noise
   model: depolarizing; amplitude damping as robustness check. Aer circuits:
   GHZ n=3/5/7 measured as X^⊗n (q≈1) and QAOA/MaxCut n=6/8 (q≈0).

### Delta vs our current `zne` (all three differences matter)

| | our `zne` (mitiq) | Scavino fixed-Richardson (`zne_fr` target) |
|---|---|---|
| scale factors | (1, 2, 3) | (1, 3) |
| extrapolation | RichardsonFactory over 3 nodes = k=2 quadratic fit | fixed 2-point coefficients (3/2, −1/2), k=1 |
| shots | **base_shots at EACH factor** (3× total budget; SHOT_MULTIPLIER=3) | **total B split B/2 + B/2** (1× total budget) |
| amplification | `fold_gates_at_random` (unitary folding) | noise-parameter scaling (direct eps multiply) |

Implementation notes for `zne_fr`:
- Easiest faithful sim path: run the executor at `n_shots = B/2` per scale
  factor and combine `1.5*mu_hat(1) - 0.5*mu_hat(3)` ourselves (no mitiq
  factory needed — it is a two-term linear combination). `SHOT_MULTIPLIER`
  for `zne_fr` should be **1** (it spends the same total budget as raw) —
  this also makes it the first technique that is cost-neutral by design.
- Amplification: for the sim overlay we have TWO options: (a) our existing
  `@x<scale>` noise dial (multiplies calibrated error rates — closest to the
  paper's "scale eps directly"; but note our x1.0→x1.5 noise-character caveat),
  or (b) global folding at factor 3. The paper's theory treats amplification
  abstractly as `eps → lambda*eps`; PROJECT_STATUS caveat 4.10 (synthetic
  depolarizing+readout at scaled settings) is actually *closer* to the
  paper's model than folding is. On hardware only folding is available —
  disclose the mismatch there.

---

## 3. Mapping to OUR quantities

| Paper | Ours |
|---|---|
| `eps` | per-location error rate ≈ backend calibrated gate error × noise-scale dial. For the overlay, the natural x-axis is the **realized** average error rate of the scaled backend (report §5 prints realized-vs-nominal) — NOT the nominal x1.0/1.5/2.0 dial. `alpha ≈ gamma * ell_n` with `ell_n ≈` transpiled noisy-op count (features already carry gate counts). |
| `B` | `base_shots` per unit (256/1024/4096/16384 planned axis). CAREFUL: our existing `zne` burns 3×B; Scavino's B is the TOTAL. `zne_fr` at budget B must use B/2 + B/2. |
| Richardson order | k=1 fixed (1,3) two-point — see §2. |
| `D_p` | needs `mu_0` (ideal) — computable in sim via `ideal.py`, and on hardware only for known-answer circuits (`mirror_circuit`, ideal=+1). This is the non-circularity argument of Angle 3: the selector never sees `mu_0`. |
| `q` | per-family: `mirror_circuit` (and ghz_plus stabilizer-like readouts near |ideal|=1) → q=1; `hw_efficient_ansatz` / `layered_random` (|ideal|<1, nonzero single-shot variance) → q=0-ish. NOTE our suite conditions on |ideal| ≥ 0.25, so no family is exactly at the q=0 "generic" pole; fit the variance exponent empirically per family like the paper does (variance-exponent fits) rather than asserting it. |
| `K_q` | computable **a priori** from (c_j, lambda_j, pi_j, nu): 4·nu (q=0) or 5·nu (q=1) for the (1,3) uniform rule. `nu` from ideal-state variance (sim) or `2*kappa` with kappa fit from the small-eps bias slope. |

Numbers to expect (from the spike, deterministic ell=20, gamma=1):
eps\* = 0.00195 / 4.9e-4 / 1.2e-4 / 3.1e-5 at B = 256/1024/4096/16384
(∝ 1/B). Variational (alpha=2, v0=1): eps\* = 0.0625 / 0.0313 / 0.0156 /
0.0078 (∝ B^(-1/2)). Our backends' per-gate eps ~1e-3..1e-2 × dial lands
right in the interesting band for the variational class at small B — good
news for the overlay's dynamic range.

---

## 4. What the spike verified (all PASS)

`spikes/spike_boundary.py` (pure numpy, no qiskit/mitiq):

1. Richardson coefficients from the Lagrange rule reproduce (3/2, −1/2) for
   (1,3) and satisfy both constraint sets; generic k=2 nodes (1,3,5) give
   (15/8, −5/4, 3/8), sum 1.
2. `K_{q,k}` penalty factors: 4·nu (q=0), 5·nu (q=1) — match hand
   calculation.
3. Sign structure: dMSE < 0 at eps\*/2 and > 0 at 2·eps\* for every B in
   {256, 1024, 4096, 16384} — **harm at low eps, help at high eps**. ✓
4. Monotonicity: eps\*(B) strictly shrinks with B; measured
   eps\*(4B)/eps\*(B) equals 4^(−1/(2p−q)) exactly (0.25 for q=1, 0.5 for
   q=0). **Harm region grows as shots shrink.** ✓
5. **Exact finite-shot cross-check** (no local expansion; binomial
   statistics on mu(eps) = (1−eps)^ell): exact zero crossing vs closed form
   agrees to 1.92% / 0.56% / 0.15% / 0.04% at B = 256/1024/4096/16384 —
   the asymptotic boundary is trustworthy over our whole planned shots axis,
   tightening exactly as the theory says (eps\* → 0 is the perturbative
   regime).
6. Critical-regime (q=2p) budget threshold B\* = K/D flips help/harm as
   predicted.

## 5. Honesty / faithfulness statement

- The formula was extracted **faithfully** — Delta_MSE structure, K_{q,k},
  the three regimes, the (1,3)/(3/2,−1/2)/uniform-split setup, and the sign
  and equal-budget conventions were each confirmed by two independent
  fetches with quoted passages. No approximation was needed to state it.
- What the spike does NOT reproduce: the remainder term `R_p` (dropped —
  but the exact-MSE cross-check bounds the cost at <2% on eps\* at B=256,
  <0.1% at 4096+), the paper's bootstrap machinery, and the amplitude-damping
  robustness variant. The exact-MSE cross-check itself assumes the
  paper's Prop. 7 local Pauli contraction model `mu = (1−gamma*eps)^ell`;
  real backends have mixed gate/readout channels, so on our data `alpha`,
  `nu`, `q` should be FIT (small-eps slope + variance-exponent fits, as the
  paper itself does) rather than derived from ell counts.
- One caveat carried from PROJECT_STATUS §4.10: our noise dial changes noise
  character at the first scaling step and caps readout at 0.45 (Lagos
  realized ~x1.28/x1.44) — the overlay's eps axis must use *realized* rates.

## 6. Recommended next steps (for the builder agents)

1. Add `zne_fr` per §2: scales (1,3), fixed (3/2,−1/2), B/2+B/2 split,
   SHOT_MULTIPLIER 1, behind a NEW technique name (backward compat: existing
   `zne` untouched).
2. New boundary module implementing exactly the spike's four functions
   (`richardson_coeffs`, `variance_penalty_K`, `delta_mse`, `boundary_eps`)
   + per-family (alpha, nu, q) fitting from sweep data.
3. The shots axis {256, 1024, 4096, 16384} spans the boundary for both
   observable classes at our backends' eps range — confirmed by the spike's
   sign tables (the q=0 class crosses inside [1e-4, 0.5] at every planned B).
