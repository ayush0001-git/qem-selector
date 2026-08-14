# Independent verifier — research-run headline audit (2026-07-22)

Owner: INDEPENDENT VERIFIER agent (wrote none of the audited code/results).
Method: fresh pandas/sklearn scripts in the session scratchpad, replicating the
model.py protocol from its documented contract (StratifiedGroupKFold 5-fold,
shuffle=True, random_state=0, grouped by family_q{n}_d{depth}; RF/GB
random_state=0; best by CV macro-F1; LOFO/LOBO/LODO on all usable rows; LODO
pools @x-scale siblings per base device). venv sklearn 1.9.0, pandas 2.3.3.

## Verdict: PASSED — zero discrepancies

### 1. Model headlines (all 4 runs re-executed from scratch)

Every number matches the trainer's metrics.json / per_seed metrics.json to 4
decimals (deterministic protocol, exact reproduction):

| run | CV acc±std | F1 | baseline | LOFO | LOBO | LODO |
|---|---|---|---|---|---|---|
| AGG best (GB) | 0.7963±0.0525 | 0.4172 | 0.5944 | 0.7870/0.4397 | 0.8926/0.6069 | 0.8648/0.3567 |
| AGG cost (GB) | 0.7278±0.1333 | 0.5835 | 0.4370 | 0.7019/0.5131 | 0.7833/0.5732 | 0.7037/0.4285 |
| SEED best (RF) | 0.7716±0.0370 | 0.3819 | 0.6222 | 0.7117/0.3343 | 0.8247/0.4707 | 0.8160/0.3368 |
| SEED cost (GB) | 0.6333±0.0293 | 0.4670 | 0.4870 | 0.5981/0.4336 | 0.7562/0.5748 | 0.6883/0.4288 |

Per-device LODO folds match (AGG best: Jakarta 0.9667 / Lagos 0.6944 / Manila
0.9333; AGG cost Lagos 0.4222 = weakest number, confirmed). Model-family picks
match (GB/GB/RF/GB). dropped_classes=[] in all 4. n_splits=5,
stratified_group in all 4. Seed-averaging deltas confirmed: LOFO +0.075/+0.104,
LODO +0.049/+0.016.

### 2. Insight spot-checks (all reproduce)

- Integrity: best_technique == NaN-aware argmin over 5 abs_error cols on
  1620/1620 rows (0 mismatches; also 0 on a 30-row random subsample).
- CDR refusal-adjusted share: 1205 accepted rows, 1008 wins = 83.65%
  (79.6/83.0/88.3% by scale); 415 refused rows -> rem 352 / zne 33 /
  raw_plus 22 / raw 8.
- Seed-flip rate: 338/1620 = 20.86% (best), 350/1620 = 21.60% (cost);
  unanimous groups 313/540 = 57.96% and 285/540 = 52.78%; ghz_plus 32.4%,
  mirror 5.25%; cdr<->rem 117+72; 0/540 agg winners on partial seed coverage.
- Q1 shares: cdr 59.26/61.67/65.74%, rem 33.70->25.37%; x2.0 CDR wins
  Jakarta 131 / Manila 124 / Lagos 100; all 49 raw+raw_plus wins on Lagos;
  raw_plus wins 37 (rem refused 18, cdr 22, both 13), raw wins 12 (rem
  refused 11); cost raw 13.3->20.4% pooled, Lagos 31.7->49.4%.
- Q4 magnitudes: raw 0.4205 / raw_plus 0.4202 / zne 0.3728 / rem 0.1963 /
  cdr 0.0893 (median 0.0249); raw 0.3656->0.4752, cdr 0.0839->0.0959 by
  scale; reduction factors 4.29x/2.00x/1.13x.
- Q2 ZNE: 78 wins, 30 full-menu (28/30 at depth 8/16; Jakarta x1.0=11,
  x1.5=6, Manila x1.0=6 as claimed), Jakarta 15->11->5, Manila 6->1->0,
  Lagos 7->13->20 with all 20 Lagos@x2.0 wins REM-refused;
  worse-than-raw 28.52/14.44/15.19% by depth, improvement
  0.0285/0.0498/0.0649; Lagos impr +0.0061, worse-than-raw 41.11%
  (Jakarta 7.96%, Manila 9.07%); win-row median clifford_fraction 0.789 vs
  0.274; agg zne wins 25/540.
- Q3: raw_plus better than raw on 49.69%, paired diff mean -0.0003, median
  +0.0001; beats cdr 5.56% / rem 6.35% of valid rows; win rows mean raw
  error 0.758, median margin 0.0115.

### 3. Sanity

- 1620 rows = 540 groups x exactly 3 seeds; aggregated.csv = 540 rows,
  9 backends, 180 circuits; aggregated has all 10 feat_* cols, 0 NaN.
- errors.log 571 lines = 415 cdr + 156 rem, exactly matching NaN counts in
  results.csv (cdr_abs_error 415 NaN, rem_abs_error 156 NaN, others 0).
  CDR reasons: 253 degenerate-training-spread + 162 fully-Clifford.
  Per-env CDR refusals 46 everywhere + 47 on Jakarta@x1.5 (the +1 runtime
  extra). REM refusals 8/46/102 at x1.0/1.5/2.0, all FakeLagosV2.
- No impossible values: no negative abs errors, clifford_fraction in [0,1],
  all raw_shots=4096, no nonpositive shots, 0 rows with cdr_abs_error<1e-12,
  0 empty labels. Max abs_error 2.836 (plausible for a ratio-style observable
  under heavy noise; not flagged).

### Nuances (not discrepancies)

1. The Q2 sentence "Full-menu wins sit at Jakarta x1.0 (11) + x1.5 (6),
   Manila x1.0 (6)" names 23 of the 30 full-menu wins; the remaining 7 are
   Jakarta@x2.0 (1), Lagos@x1.0 (3), Lagos@x1.5 (2), Manila@x1.5 (1). The
   named counts are exact; the sentence is a summary, not exhaustive.
2. Headline rounding is honest everywhere (e.g. 83.65% quoted as 83.7%,
   1.85% as 1.9%).

Scripts: scratchpad verify_model.py / verify_insights.py / verify_extra.py
(session-local, not committed).
