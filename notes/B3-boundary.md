# B3-boundary — src/qemsel/boundary.py

Implemented all 7 stubs + private helpers, from the boundary spike's exact
Scavino formula. `delta_mse` = frozen `d_p*eps^(2p) - k_q*eps^q/shots`
(positive = zne_fr helps). `regime` strictly `> tol` (exact 0 => harm).
`boundary_eps`/`boundary_shots` solve the zero crossing and cover all three
regime shapes (subcritical power law, critical `2p==q` budget threshold,
supercritical upper crossing), returning None when `d_p<=0`, `k_q<=0`, or no
eps crossing exists.

`variance_k_q` = a-priori nu=1, q=0 geometry `sum c_j^2/pi_j - 1` via
`mitigation.richardson_coefficients` (single source of truth). `estimate_params`
(sim-only, ValueError on ibm_*) gets mu_0 from `ideal`, alpha by secant to one
noisy probe; deterministic class (|mu0|=1) => q=1,nu=2|alpha|, else q=0,
nu=1-mu0^2. `overlay_selector_vs_theory` sweeps backends(eps)×shots, votes
selector ZNE-share vs theory regime, returns the 9-key dict + robust PNG.

B1 landed mid-task: `ZNE_FR_SCALE_FACTORS=(1,3)` => `variance_k_q()`=4.0.
Tests: 41 (39 fast + 2 slow), all green. Full suite unweakened.
