# Adversarial tester T-stats — round 1 (src/qemsel/stats.py)

Date: 2026-07-23. Verified by RUNNING simulations (scratchpad scripts `t_stats_adv.py`, `t_stats_adv2.py`), not by reading. 88 checks, 0 failures. No src/tests files touched.

## What was exercised

1. **bootstrap_ci coverage (simulated draws)**
   - Bernoulli p=0.3, n=100, 400 reps, n_boot=2000: coverage **0.948** (target ~0.95).
   - Normal mu=1, sd=2, n=60, 300 reps: coverage 0.920.
   - `win_share_ci` true share 0.4 with empty/NaN labels mixed in: coverage 0.970 (empty/NaN correctly excluded from denominator).
   - ci=0.5 interval covers ~50% (0.475). Determinism (same seed -> identical dict), NaN drop counts, custom statistic (median), constant input (lo=hi=estimate), JSON-serializable, ValueError on <2 non-NaN and ci outside (0,1): all correct.

2. **paired_permutation_test**
   - Null (symmetric N(0,1) diffs, 300 reps, n_perm=499): p uniform-ish — frac(p<=.05)=0.047, frac(p<=.25)=0.230, frac(p<=.5)=0.487, mean p=0.505; min p respects add-one floor 1/(n_perm+1).
   - Power: shift 0.5, sd 1, n=50 -> 0.900 (theory ~0.93).
   - Exact-p cross-check: 6 identical +1 diffs, two-sided exact p=1/32; MC mean over 30 seeds = 0.03118. Large-magnitude (1e12) diffs: p matches exact 2/2^12 — the 1e-15 epsilon does not distort.
   - Alternatives 'less'/'greater'/'two-sided' point the right way; identical arrays give p=1.0; pairwise NaN drop counted; deterministic; ValueErrors on length mismatch / bad alternative / <2 pairs.

3. **cliffs_delta**
   - Known cases: all-greater=+1, all-less=-1, identical=0, tie-symmetric=0, [0,0,1] vs [0,1,1] = -1/3, NaN drop, antisymmetry, ValueError on all-NaN array.
   - Brute-force cross-check on 20 random tie-heavy arrays: exact agreement. 1620x1620 runs fine.

4. **koester_checklist (constructed pathological frames)**
   - Per-seed frame with planted pathologies: value overshoot fires ({'t1':1}), error > 1+|ideal| fires, nan_rate=1/6, wrong best_technique -> n_mismatch=1 and passed=False, tiny-margin row flagged by k-sigma check (fraction 1/5; NaN-error row correctly excluded from the margin denominator). k_sigma=0 flags nothing; k_sigma=1e9 flags all 5 checked rows. Clean frame -> passed=True. All-NaN-error row skipped, not a mismatch.
   - Aggregated frame: coverage-rule argmin honored (global argmin on a partial-coverage tech is NOT a mismatch when best is the max-coverage argmin); partial-coverage winner fires (count 1, passed=False); overshoot/error_beyond/winner_margin correctly None on this schema.
   - Argmin tie -> first (config-order) column wins, matching docstring. Partial value-column presence -> partial overshoot dict, margin None. ValueError on missing error columns / missing best_technique. Output JSON-serializable.

5. **Helpers**: sigma_shot (value, clamp at |v|>1 -> 0, NaN/shots<=0 raise), win_shares (empty/NaN excluded, absent tech -> 0.0, all-empty -> {}), summarize_folds (single fold std=0, ddof=1, empty/NaN raise).

## Informational observations (no action required, paper-caveat material)

- **Percentile bootstrap small-p regime**: at p=0.05, n=40 coverage drops to ~0.87 — inherent percentile-bootstrap behavior, not a bug; matters only if a paper CI is quoted for a technique with a near-zero win share on a small grid. (At the paper's n>=100 and moderate shares, coverage is nominal.)
- **Sign-flip null is symmetry, not mean-zero**: with mean-zero but skewed diffs (exp(1)-1, n=40), type I ~0.07 at alpha=.05. The docstring states the symmetric null explicitly, so this is per-spec; worth one sentence of caveat if paired error differences are heavily skewed.

## Verdict

Zero critical/major findings. Module behaves as documented under adversarial inputs.

Repro:
```
& "E:\quatum  computiiing\qem-selector\.venv\Scripts\python.exe" <scratchpad>\t_stats_adv.py   # 78 checks
& "E:\quatum  computiiing\qem-selector\.venv\Scripts\python.exe" <scratchpad>\t_stats_adv2.py  # 10 checks
```
(scratchpad = C:\Users\ayush\AppData\Local\Temp\claude\E--quatum--computiiing\13852342-c2be-4d47-b461-7c0561ff460a\scratchpad)
