# Adversarial tester T-boundary — round 1 (2026-07-23)

Scope: `src/qemsel/boundary.py` formula vs arXiv:2605.08251 (re-fetched fresh),
3 hand-derived boundary points, asymptotics, and `overlay_selector_vs_theory`
fed a SYNTHETIC bundle with a known rule (zne iff eps > 0.1). All verification
by RUNNING code. Scripts live in the session scratchpad
(`t_boundary_formula.py`, `t_boundary_overlay.py`, `t_estimate_probe.py`);
repro = run each with the venv python.

## Verdict: PASS — zero critical/major findings. 83 + 26 + 4 checks, all green.

## 1. Paper re-fetch (independent of the spike)

Fetched arxiv.org/html/2605.08251v1 AND ar5iv.labs.arxiv.org/html/2605.08251.
Confirmed verbatim: Delta_MSE = MSE_noisy − MSE_ZNE, > 0 <=> ZNE helps;
K_{q,k} = nu[Σ c_j² λ_j^q/π_j − 1]; p=1, D_1 = α²; subcritical
eps*(B) = (K/(D·B))^{1/(2p−q)}; critical B* = K/D; supercritical no shrinking
lower boundary. Paper's own worked value for the (1,3) uniform rule:
K = ν[7/2 + 3^q/2] → 4ν (q=0), 5ν (q=1). Code's `_variance_penalty`
reproduces ν(3.5 + 3^q/2) EXACTLY for q∈{0,1,2,3} × ν∈{1,2.5,40};
`variance_k_q()` = 4.0. (Caution for future fetchers: the first WebFetch
summarizer mis-multiplied to "6ν/7ν" while quoting the correct formula —
always recompute from the quoted expression.)

## 2. Hand-derived points (computed before running code) — all match

- A (q=0, D=4, K=4): eps*(1024) = 1/32 = 0.03125; inverse B*(0.03125)=1024;
  delta_mse(eps*)=0 exactly.
- B (q=1, D=400, K=200 = 5·2·20): eps*(4096) = 1/8192; eps*(256) = 1/512.
- C (critical q=2p, D=2, K=8): boundary_eps → None at every B;
  boundary_shots → 4.0 at every eps; help/harm flips across B*=4; delta==0
  counts as harm (regime's "earn its cost" rule) — verified.

## 3. Asymptotics — all match

eps*(4B)/eps*(B) exactly 4^{−1/(2p−q)} (0.5 q=0, 0.25 q=1) across
B=256..4096; eps* strictly ↓ 0 as B→∞; sign structure harm-below/help-above
at eps*/2 and 2eps* for B∈{256..16384}; supercritical q=3: help at all small
eps (paper) with the code's returned root the correct upper crossing of the
truncated formula (documented in-code; not a defect). Degenerate guards
(d_p≤0, k_q≤0, eps/shots≤0, tol<0) all behave per contract.

## 4. Overlay with synthetic known-rule bundle — agreement math correct

Grid: 7 scaled Manila/Lagos backends (realized avg_2q_error 0.005–0.199,
straddling 0.1) × shots {64,256,4096}; theory params FIXED (D=4,K=4,p=1,q=0
→ help iff eps > 1/√B). Bundle = dict with a predict-only model implementing
zne_fr iff feat_backend_avg_2q_error > 0.1 (verified feat == overlay eps axis
to 1e-15). Hand-computed: agree 17/21 = 80.952%, IoU 8/12, sel share 9/21,
theory share 11/21 — overlay returned ALL FOUR exactly, plus all 21 grid rows
(selector_zne / theory_regime / delta_mse) matching an independent
reimplementation. Also verified: real PNG written, dict JSON-serializable,
deterministic across runs, predict_proba+abstain_threshold bundle counts
abstain as NOT-zne (x8 backend, p=0.8 < 0.9), exact 0.5 vote share counts as
ZNE (>=0.5 contract), ibm_* refused by both overlay and estimate_params,
empty shots_list and malformed bundle refused.

## 5. estimate_params probe

Deterministic same-seed; mirror (ideal +1) routes to q=1 with
k_q = 5ν = 10|α| exactly (paper Prop 6/7); different seed → different probe.

## Minor observations (no action required)

- Supercritical boundary_eps returns the formula's upper crossing rather than
  None; deliberate and documented, delta-sign behavior matches the paper.
- Overlay computes theory as `dmse > 0.0` inline rather than calling
  `regime()`; identical at tol=0.
