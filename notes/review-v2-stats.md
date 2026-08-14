# Review V2 — Statistician (stats.py / report §8 / checklist faithfulness)

Scope: `src/qemsel/stats.py`, report's CI/effect-size presentation, multiple-comparison
risk, faithfulness of `koester_checklist` to arXiv:2605.29872. All suspicions verified
by running code against `results/research/results.csv` (1620 rows) and the
`results/boundary_smoke` artifacts with the project venv. No files outside this note touched.

---

## HIGH-1 — All row-level inference ignores the sweep's cluster structure (anti-conservative)

The research sweep is a crossed design: 180 circuits x 9 backends = 1620 rows, ONE seed
per cell (verified: `df.groupby(["circuit_id","backend"]).size()` is all 1). Rows sharing a
circuit are strongly correlated, but `paired_permutation_test`, `bootstrap_ci`, and
`win_share_ci` all treat rows as i.i.d.

Measured on the frozen sweep:

- ICC of the paired difference (raw_plus - raw) by circuit_id: **0.25**.
- ICC of the "cdr wins" indicator by circuit_id: **0.635**.
- raw_plus_vs_raw two-sided p: **0.369** row-level vs **0.610** with circuit-level sign
  flips (180 clusters, 20k perms) — the row-level test roughly halves the p-value.
- cdr win-share 95% CI: **[0.598, 0.646]** row bootstrap vs **[0.564, 0.680]** circuit-cluster
  bootstrap — the shipped CI is ~2.5x too narrow.
- Illustration of how bad it can get: the shipped `boundary_smoke/stats.json` reports
  cdr_vs_rem p = 0.0002 on 24 rows — but the smoke run contains only **3 distinct circuits**;
  a circuit-level sign-flip test has only 2^3 = 8 sign patterns and cannot produce p below
  ~0.125. The 0.0002 is borrowed entirely from treating 24 correlated rows as independent.

This is precisely the Köster–Mauerer point that correlated observations "drastically reduce
the effective number of independent observations" — the paper's own hygiene citation
invalidates its current p-values/CIs.

**Fix (additive, backward-compatible):** optional `groups=` parameter on
`paired_permutation_test` (flip signs per cluster) and on `bootstrap_ci`/`win_share_ci`
(resample clusters); default None keeps byte-identical current behavior. The paper should
cluster on circuit_id (ICC by backend is only 0.002 — backend clustering is negligible).
Report §8 must state the resampling/exchangeability unit.

## HIGH-2 — `sigma_shot` on mitigated estimates with budget-total shots understates sigma ~3–8x

`winner_margin_below_k_sigma` (stats.py) and `derive_significant_label` (model.py, same
helper) compute a technique's sigma as `sigma_shot(value, <tech>_shots)`. But
`<tech>_shots` is the **cost ledger** (total budget consumed), and the formula
`sqrt((1-v^2)/shots)` is only the estimator sigma for a direct measurement (raw/raw_plus).

For `zne` (verified: RichardsonFactory over scale factors (1,2,3), `zne_shots` = 12288 =
3x4096): Richardson coefficients at 0 are (3, -3, 1), so
Var = (9+9+1) * sigma_node^2 = 19 * (1-v^2)/4096, i.e. true sigma ≈ 4.36 * sigma_node.
The code computes sqrt((1-v^2)/12288) = 0.577 * sigma_node — an underestimate of ~**7.6x**.
`zne_fr` (nodes (1,3), coeffs (1.5,-0.5)) is ~2.2x; `cdr`/`rem` use 45056/12288 total shots
in the denominator while the target-circuit estimate rests on 4096-shot measurements plus
fit/inversion noise — also several-fold underestimates.

Consequences (both anti-conservative, the artefact direction 2605.29872 warns about):
- research checklist "n_flagged = 229 ties (14.1%)" is a substantial undercount of true
  statistical ties;
- the significance-aware labels (`model_significant.joblib`, `metrics_significant.json`)
  declare "winner beats runner-up by >= 2 sigma" at margins that are really ~0.3 sigma
  when ZNE is winner or runner-up.

The variance-amplification factor is already documented in-repo (LITERATURE.md,
Mohammadipour & Li: Var <= (sigma^2/N_S) * ||gamma||_1^2; Krebsbach: Lambda^2 overhead).

**Fix (additive):** per-technique sigma model — for zne/zne_fr multiply the per-node
shot sigma by sqrt(sum gamma_j^2) using `richardson_coefficients` (already exported by
mitigation.py); for cdr/rem either use the target-circuit shots with a documented
amplification floor or report the margin check as approximate-lower-bound. New config
key / keyword so existing outputs stay byte-identical.

## HIGH-3 — Degenerate [0, 0] "95% CIs" at boundary win shares (already shipped)

Percentile bootstrap of a 0/n (or n/n) proportion collapses to a zero-width interval.
`results/boundary_smoke/report.md` §8 already publishes `zne 0 [0, 0]`, `zne_fr 0 [0, 0]`,
`cdr_ridge 0 [0, 0]` as 95% CIs from n = 24 — an interval with 0% coverage presented as 95%.
A Wilson 95% interval for 0/24 is [0, 0.138]. This will recur exactly where the paper's
Angle-3 claim lives (ZNE win share ~= 0 in low-noise/low-shot slices), and tests even pin
the degenerate behavior (`test_win_share_ci_all_wins_is_one` asserts lo == hi == 1.0).

**Fix (additive):** add a `wilson_interval(k, n, ci)` helper (or BCa bootstrap) and have
`win_share_ci` include it alongside the percentile bounds; report renders Wilson for
proportions. Note the module currently implements only the percentile bootstrap (no BCa
anywhere despite skewed error distributions); report.py correctly says "percentile
bootstraps", so no false BCa claim — the issue is method adequacy at the boundary, not
labeling.

## MEDIUM-1 — Refusal-conditioning bias in paired tests and Cliff's delta

`paired_permutation_test` drops pairs where either value is NaN; `cliffs_delta` drops NaNs
**per array** (samples become different populations). CDR refuses on 415/1620 rows, and it
refuses on the *hard* rows: mean raw error is 0.544 on dropped rows vs 0.360 on kept rows
(verified). So every cdr-vs-X comparison is conditioned on CDR's easy region, and the
per-array drop in `cliffs_delta` is worse: rem's sample keeps rows where CDR refused
(rem mean err 0.177 there), asymmetrically flattering CDR. The report's paired-tests table
shows `n_pairs` but omits `n_dropped_nan` (532 pairs would vanish silently for cdr_vs_rem
on the research data) and never states the conditioning.

**Fix:** report `n_dropped_nan` in the §8 paired table + one sentence stating comparisons
are conditional on joint success; for effect sizes on paired columns, drop pairwise (not
per-array) or add a `paired=` flag; optionally add a worst-case sensitivity line
(refusals counted as max error).

## MEDIUM-2 — `koester_checklist` is not the arXiv:2605.29872 checklist

Re-fetched the paper (abstract; full text/HTML returns 404). Its actual proposals: an
**eight-criterion review framework** (statistical rigour / reproducibility / reporting
quality) and **minimum reporting standards** — explicit parameter documentation,
parameter-sensitivity robustness checks (their 132-config sweep), longitudinal drift
assessment, and inferential testing **with effect sizes**. The qemsel `koester_checklist`
is a six-item internal data-integrity gate (argmin consistency, physical-range overshoot,
NaN rate, shot-noise margin, seed coverage) — none of these items appear in the paper;
the overshoot check is closer to their *other* paper (2607.09360). Rendering
"### Koester-Mauerer statistical checklist ... **PASS**" claims compliance with a
checklist it does not implement — an easy reviewer hit (plausibly by the authors
themselves). Items the paper asks for that the codebase does NOT yet cover: conclusion
robustness across ZNE knobs (scale factors / extrapolant), and effective-independent-N
(HIGH-1 is its sim-side analogue; drift itself is n/a sim-side and should be *stated* as
out of scope, not silently passed).

**Fix:** rename the rendered section to "Internal integrity checklist (motivated by
Köster & Mauerer 2605.29872/2607.09360)" and add a short mapping table: which of their
reporting standards the pipeline meets (inferential tests + effect sizes + CIs + full
parameter logging in run_meta.json), which are N/A (drift, sim-only), which are open
(knob-robustness sweep, clustered effective-N).

## MEDIUM-3 — No multiple-comparison handling; `top2` contest is post-hoc

Grep confirms no Holm/Bonferroni/BH anywhere in src. Current §8 runs 2 tests, but the
paper plan implies many more: 7 techniques = 21 pairwise contests, times slices (5
families x 9 backends x shots axis for Angle 3; CDR-variant overfitting map for Angle 2).
Additionally the `top2_<a>_vs_<b>` contest selects the two highest win-share techniques
from the SAME data it then tests — selective inference; its nominal p-value is invalid
as stated. `raw_plus_vs_raw` is fine (pre-specified control).

**Fix (additive):** `adjust_pvalues(dict, method='holm'|'bh')` helper in stats.py; report
prints adjusted alongside raw p-values and states the test family size; the paper
pre-registers its confirmatory contrasts and labels top2 as descriptive.

## LOW-1 — inf bypasses the stated NaN policy

`_clean_1d` masks NaN only. `bootstrap_ci([1,2,inf,3])` returns estimate=inf, hi=**nan**
(verified) — silently producing NaN output, which the module docstring forbids;
`cliffs_delta` counts inf-inf pairs as ties. Use `np.isfinite` (and count dropped
non-finite) or raise.

## LOW-2 — §8 presentation polish

- "A CI that straddles a rival's estimate means the ordering is not resolved" is the
  overlapping-CI fallacy; win shares are also correlated multinomial components. Better:
  bootstrap the share *difference* of the top-2, or soften the sentence.
- Effect-size table: add the conventional Cliff's-delta magnitude bands
  (|d|<0.147 negligible / <0.33 small / <0.474 medium / else large) and a bootstrap CI on
  delta, so "-0.365" is interpretable.

## Verified-OK (no findings)

- `paired_permutation_test` mechanics: sign-flip = within-pair A/B swap, correct
  exchangeability under H0 *given independent pairs*; add-one p-correction; two-sided /
  less / greater tolerances; determinism.
- `cliffs_delta` arithmetic: ties, full separation, sign convention, hand-checked
  fraction, 1620^2 memory fine.
- `koester_checklist` internals: coverage-restricted argmin exactly mirrors experiment.py
  (0 mismatches on both frozen schemas), stable-sort tie-break = config order,
  aggregated-schema margin honestly None instead of fabricated.
- `win_shares` empty/NaN denominator policy; `summarize_folds` ddof=1 with n=1 guard;
  `bootstrap_ci` determinism and key contract.
- report.py reads every stats field with .get (partial dicts degrade gracefully);
  no false "BCa" claim anywhere (method is stated as percentile).
