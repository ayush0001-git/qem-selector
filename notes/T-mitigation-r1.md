# T-mitigation round 1 — adversarial module test (2026-07-23)

Verdict: **PASS — zero critical/major findings.** All verification done by RUNNING
instrumented code (scripts in scratchpad: t_zne_fr.py, t_cdr_variants.py,
t_beat_raw.py, t_byte_identical.py, research_subset.yaml).

## 1. zne_fr vs the spike's fixed-Richardson definition — MATCHES EXACTLY
Instrumented `qemsel.backends.make_executor` with a spy factory + stub executors:
- Nodes `ZNE_FR_SCALE_FACTORS == (1.0, 3.0) == spike_boundary.LAMBDAS`;
  `richardson_coefficients` agrees with the spike's `richardson_coeffs` on
  (1,3), (1,2,3), (1,3,5), (1,1.5,2,3); two-point rule is exactly (1.5, -0.5).
- Call pattern (base_shots=4096, seed=7, FakeLagosV2): passed executor invoked
  **0** times; exactly **2** per-level executors built at **2048 shots each**
  (uniform B/2+B/2 split of ONE budget = spike PI=(0.5,0.5); total spent 4096 =
  1x base, consistent with SHOT_MULTIPLIER_V2['zne_fr']==1); each invoked once.
- Estimate == `1.5*E(l=1) - 0.5*E(l=3)` as an EXACT float identity on poisoned
  stub values (fixed coefficients, no refit; passed-executor poison value 999.0
  never leaked in).
- Level circuits are op-identical (qasm2 compare) to independently computed
  deterministic `mitiq.zne.scaling.fold_global` at 1.0 / 3.0 (ops 32 -> 32 / 96);
  caller's circuit not mutated; fold_global deterministic call-to-call.
- base_shots=1 refuses loudly (MitigationError, level budget rounds to 0).
- Bitwise deterministic with a real executor; seed changes the value.

## 2. cdr_ridge / cdr_rf — call counts, determinism, guard parity
- Executor call counts (counting wrapper around the real executor):
  cdr 11, cdr_ridge 11, cdr_rf 11 == SHOT_MULTIPLIER_V2 (1 + 10 training);
  shots_consumed == 11 * base.
- Bitwise seeded determinism for both (two circuits/backends); different seed
  gives different value.
- Guard PARITY with 'cdr' (load-bearing for Angle 2's shared accepted-row set):
  fully-Clifford GHZ, idle-wire circuit, and degenerate-spread near_clifford
  (q2 d4 s0) all refuse identically across cdr/cdr_ridge/cdr_rf with
  MitigationError and **0 noisy executor calls** spent on refusal.
- Unknown regressor name -> ValueError before any execution.

## 3. All three beat raw on a noisy circuit — CONFIRMED
mirror q3 d16 s0 on FakeLagosV2@x2.0 @4096 shots: raw_err 1.0029 vs
zne_fr 0.9751, cdr_ridge 0.3766, cdr_rf 0.2780 (all three < raw). Per-unit
losses on other probed units (e.g. cdr_ridge on layered_random q3 d8) are the
expected help/harm variation the selector studies, not defects.

## 4. Old techniques byte-identical to captured references — CONFIRMED
- Fresh `run_experiment --config configs/tiny.yaml` -> results.csv SHA256
  `cbdf857d...3d33` == results/tiny/results.csv (raw/zne/cdr/rem).
- Fresh EXACT-SUBSET of configs/research.yaml (3 families x n{2,3} x d4 x s0 x
  {FakeManilaV2, FakeLagosV2@x1.5}, techniques incl. **raw_plus**): all 12 rows
  x 35 cells STRING-identical to results/research/results.csv (also proves
  per-unit seeding is grid-position independent).
- tests/test_mitigation.py + test_mitigation_v2.py: 103 passed.

## Minor observation (no action required this round)
Both zne_fr level executors are built with the SAME seed (contract says so).
The two levels run different (folded) circuits so draws differ in practice,
but strictly the spike's variance model assumes independent shot noise across
levels; if a reviewer probes this, deriving per-level seeds (seed, seed+1)
would be cleaner — behavior change, so only with a coordinated decision.
