# B1 — mitigation.py V2 (zne_fr, cdr_ridge, cdr_rf)

Implemented the three stubs + `richardson_coefficients`. V1 dispatch untouched
→ byte-identical (captured FIRST via scratchpad, pinned in
`test_mitigation_v2.py`: raw 0.203125, raw_plus 0.19176…, zne 0.21875,
cdr 0.16611…, rem 0.23111… on FakeManilaV2@256/seed3/ZZ).

Key decision: **ZNE_FR_SCALE_FACTORS retuned (1.0,2.0,3.0)→(1.0,3.0)** — the
Scavino k=1 two-point rule the boundary spike faithfully extracted (coeffs
(3/2,-1/2), B/2+B/2 split, K_q 4ν/5ν). Sanctioned by the stub ("SPIKE MAY
ADJUST node values; boundary.py reads this constant"). boundary.py (B3) imports
both this constant and `richardson_coefficients`, so theory and impl can't drift.

- `zne_fr`: fixed coeffs × equal_split rebuilt executors (base//n per level,
  same seed, close() in finally) × deterministic `fold_global`. Passed executor
  NOT invoked. Multiplier 1 (cost-neutral vs raw). Also supports 'full'.
- `cdr_ridge`/`cdr_rf`: Route B (generate_training_circuits bypass), SAME 3
  guards as `_apply_cdr` (Clifford / idle-wire / spread), guards run before any
  noisy call. `Ridge(alpha=1.0)` / `RandomForestRegressor(100,random_state=seed)`,
  single noisy feature. Cost 1+N=11.

⚠ FLAG for B4/paper: `Ridge(alpha=CDR_RIDGE_ALPHA=1.0)` shrinks HARD on this
1-feature/~11-point scale — cdr_ridge=0.776 vs linear cdr=0.166 on the capture
circuit. Unlike the cdr-nl spike's RidgeCV (auto near-0 alpha ≈ linear). Value
is the architect default (implemented verbatim, not my call to change); revisit
CDR_RIDGE_ALPHA (or switch to RidgeCV) before the Angle-2 run.

Tests: 48 in test_mitigation_v2.py (constants, richardson known-values+Lagrange
constraints+validation, zne_fr coeff/split/close/determinism/wrapping, cdr
call-count/guards/determinism/name-validation, non-mutation, slow byte-identical
regression). Full suite: green.
